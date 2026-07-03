"""Episode recorder for HumanoidEnv data collection.

Extracted from HumanoidEnv so the env stays a thin coordinator: this owns the
in-progress recording session (with its OWN lock) and flushes one mp4 per
RECORD_CAMERA plus a robot_states.npz + metadata.json per episode. The env drives
it from the collect loop (``tick``) and from lifecycle (``start`` / ``stop`` /
``finalize``); the env also pins/unpins the record cameras, since it owns the
camera subscriptions. The recorder is purely about capturing and writing rows.

Thread-safety: all session state is guarded by an internal lock. ``tick`` runs on
the env's collect thread; start/stop/finalize on the caller (GUI) thread.
``stop`` clears the in-progress session under the lock and HANDS IT BACK, so the
collect thread stops touching it immediately and finalize (disk I/O) happens
outside the lock without blocking streaming.
"""

import json
import pathlib
import threading
from datetime import datetime

import cv2
import numpy as np


# Cameras captured to disk during data collection (mirrors the old
# RobotDataCollector.CAMERA_NAMES). build_dataset.py only consumes head +
# hand_left; hand_right is recorded for future use.
RECORD_CAMERAS = ["hand_left", "hand_right", "head"]


class Recorder:
    """Owns an in-progress recording session and flushes it to disk.

    Dependencies are injected so the recorder never touches the SDK or the env's
    camera/lock internals directly:
      * ``read_arm_joints`` — () -> (vals, ts); e.g. ``robot.arm_joint_states``. Its
        result is COPIED per row (the SDK hands back a reused internal buffer; see
        _record_tick) so recorded joints don't all alias the final value.
      * ``extract_pose``    — status link-frame dict -> (x, y, z, qx, qy, qz, qw).
    """

    def __init__(self, output_dir, record_hz, read_arm_joints, extract_pose,
                 record_cameras=RECORD_CAMERAS):
        self.output_dir = pathlib.Path(output_dir)
        self.record_hz = record_hz
        self._read_arm_joints = read_arm_joints
        self._extract_pose = extract_pose
        self.record_cameras = list(record_cameras)
        self._lock = threading.Lock()
        self._rec = None
        self._is_recording = False

    @property
    def is_recording(self):
        return self._is_recording

    def start(self, episode_name=None):
        """Begin a session. Returns True if it actually started, False if one was
        already running. The caller pins ``record_cameras`` on a True return so a
        tick never misses one to idle-eviction."""
        with self._lock:
            if self._is_recording:
                print("[Recorder] Already recording.")
                return False

            if episode_name is None:
                episode_name = f'episode_{sum(1 for _ in self.output_dir.glob("episode_*")):03d}'

            episode_dir = self.output_dir / episode_name
            cameras_dir = episode_dir / 'cameras'
            cameras_dir.mkdir(parents=True, exist_ok=True)

            self._rec = {
                'episode_dir':   episode_dir,
                'cameras_dir':   cameras_dir,
                'writers':       {},
                'needs_bgr':     {},
                'timestamps':    [],
                'left':          [],
                'right':         [],
                'arm_joints':    [],
                'arm_joints_ts': [],
                'gripper':       [],
                'cam_ready':     {name: False for name in self.record_cameras},
                'start_time':    datetime.now().isoformat(),
            }
            self._is_recording = True

        print(f"[Recorder] Recording started -> {episode_dir}")
        return True

    def stop(self):
        """End the active session and hand back the in-progress rec dict (or None
        if not recording) for the caller to ``finalize``. Clearing ``_rec`` under the
        lock first means the collect thread sees it gone and won't touch the session,
        so finalize (disk writes) can run outside the lock without blocking streaming."""
        with self._lock:
            if not self._is_recording:
                return None
            print("[Recorder] Stopping recording...")
            self._is_recording = False
            rec = self._rec
            self._rec = None
        return rec

    def tick(self, t, status, grip, frames):
        """Append one row if a session is active; no-op otherwise. Called every
        collect-loop tick. Takes the session lock internally."""
        if status is None:
            return
        with self._lock:
            if not self._is_recording or self._rec is None:
                return
            self._record_tick(t, status, grip, frames)

    def _record_tick(self, t, status, grip, frames):
        """Append one row to the active session. Caller holds the session lock.

        ``frames`` is the collect loop's per-camera rolling [prev, cur] pairs; the
        current frame is frames[name][-1] (absent for a camera that dropped this tick).
        """
        rec = self._rec

        # Wait until every record camera has produced at least one frame before
        # writing anything, so row counts stay in sync.
        for name in self.record_cameras:
            if name in frames:
                rec['cam_ready'][name] = True
        if not all(rec['cam_ready'].values()):
            missing = [n for n, r in rec['cam_ready'].items() if not r]
            print(f"[Recorder]   waiting for cameras: {missing}")
            return

        # --- Record robot state ---
        rec['left'].append(self._extract_pose(status['frames']['arm_left_link7']))
        rec['right'].append(self._extract_pose(status['frames']['arm_right_link7']))
        # arm_joints sourced from arm_joint_states(). COPY the returned array before
        # storing it: arm_joint_states() hands back a handle to one REUSED internal SDK
        # buffer, so appending it by reference makes every recorded row alias the same
        # object — at finalize they all collapse to the buffer's final value (frozen
        # joints across the whole episode, while the scalar arm_joints_ts stays live).
        # np.array() snapshots the current values. arm_joints_ts records each sample's
        # timestamp for staleness checks.
        arm_vals, arm_ts = self._read_arm_joints()
        rec['arm_joints'].append(np.array(arm_vals))
        rec['arm_joints_ts'].append(arm_ts)
        rec['gripper'].append(grip)
        rec['timestamps'].append(t)

        # --- Write camera frames (skip cameras that dropped this tick) ---
        for name in self.record_cameras:
            pair = frames.get(name)
            if pair is None:
                continue
            frame = pair[-1]
            if name not in rec['writers']:
                h, w   = frame.shape[:2]
                path   = str(rec['cameras_dir'] / f'{name}.mp4')
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                rec['writers'][name]   = cv2.VideoWriter(path, fourcc, self.record_hz, (w, h))
                rec['needs_bgr'][name] = frame.ndim == 3
                print(f"[Recorder]   opened cameras/{name}.mp4  ({w}x{h})")

            rec['writers'][name].write(
                cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if rec['needs_bgr'][name] else frame
            )

    def finalize(self, rec):
        """Flush video files and write npz/metadata for a finished session. Runs
        outside the lock (``rec`` is already detached from the live session)."""
        for w in rec['writers'].values():
            w.release()

        timestamps = rec['timestamps']
        n = len(timestamps)
        if n == 0:
            print(f"[Recorder]   no frames recorded -> {rec['episode_dir']}")
            return

        left_a  = np.array(rec['left'],  dtype=np.float32)   # (N, 7)
        right_a = np.array(rec['right'], dtype=np.float32)   # (N, 7)

        # robot_states.npz — one array per signal (build_dataset.py reads
        # timestamps/left_pos/left_quat/gripper). arm_joints_ts is additive.
        np.savez(
            rec['episode_dir'] / 'robot_states.npz',
            timestamps    = np.array(timestamps,            dtype=np.float64),  # (N,)    seconds
            left_pos      = left_a[:, :3],                                      # (N, 3)  metres
            left_quat     = left_a[:, 3:],                                      # (N, 4)  [qx,qy,qz,qw]
            right_pos     = right_a[:, :3],                                     # (N, 3)
            right_quat    = right_a[:, 3:],                                     # (N, 4)
            arm_joints    = np.array(rec['arm_joints'],    dtype=np.float32),   # (N, 14) radians
            arm_joints_ts = np.array(rec['arm_joints_ts'], dtype=np.float64),   # (N,)    freeze diagnostic
            gripper       = np.array(rec['gripper'],       dtype=np.float32),   # (N, 2)  0=open 1=closed
        )

        with open(rec['episode_dir'] / 'metadata.json', 'w') as f:
            json.dump({
                'fps':          self.record_hz,
                'n_frames':     n,
                'start_time':   rec['start_time'],
                'camera_names': self.record_cameras,
            }, f, indent=2)

        print(f"[Recorder]   saved {n} frames -> {rec['episode_dir']}")
