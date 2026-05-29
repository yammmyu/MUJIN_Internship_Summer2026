import json
import pathlib
import signal
import threading
import time
from datetime import datetime

import cv2
import numpy as np
from a2d_sdk.robot import RobotDds as Robot, CosineCamera as Camera, RobotController

CAMERA_NAMES = ["hand_left", "hand_right", "head"]
RECORD_HZ    = 30


class RobotDataCollector:
    def __init__(self, output_dir="recordings", record_hz=RECORD_HZ):
        self.robot            = Robot()
        self.robot_controller = RobotController()
        self.camera           = Camera(CAMERA_NAMES)
        self._get_camera_frame = self._sdk_get_frame
        time.sleep(1.0)

        self.output_dir = pathlib.Path(output_dir)
        self.record_hz  = record_hz
        self._interval  = 1.0 / record_hz

        # Recording state — guarded by _rec_lock. _rec holds the in-progress
        # recording session (None when not recording).
        self._is_recording = False
        self._rec_lock     = threading.Lock()
        self._rec          = None

        # Latest streamed values, refreshed every tick by the stream loop.
        self._frame_lock  = threading.Lock()
        self.latest_frames = {name: None for name in CAMERA_NAMES}

        self.latest_left_pos   = None  # (x, y, z)
        self.latest_right_pos  = None  # (x, y, z)
        self.latest_left_quat  = None  # (qx, qy, qz, qw)
        self.latest_right_quat = None  # (qx, qy, qz, qw)

        # Single always-on stream loop: get_motion_status + camera frames at
        # RECORD_HZ, whether or not we are recording.
        # _stop_event is set by shutdown() to wake the sleeping thread immediately.
        self._stop_event  = threading.Event()
        self._stream_thread = threading.Thread(target=self._stream_loop, daemon=True)
        self._stream_thread.start()

    # ------------------------------------------------------------------ #
    #  Camera frame accessor                                               #
    # ------------------------------------------------------------------ #

    def _sdk_get_frame(self, name):
        """Get camera feed live"""
        image, _ = self.camera.get_latest_image(name)
        return image

    def get_latest_frame(self, name):
        """Latest streamed frame for a camera (None until a frame arrives)."""
        with self._frame_lock:
            return self.latest_frames.get(name)

    # ------------------------------------------------------------------ #
    #  End-effector snapshot — single SDK call returns both arms          #
    # ------------------------------------------------------------------ #

    def _get_hand_statuses(self):
        """Return (left_7, right_7) from a single get_motion_status() call."""
        status = self.robot_controller.get_motion_status()
        frames = status['frames']

        def _extract(link):
            f    = frames[link]
            pos  = f['position']
            quat = f['orientation']['quaternion']
            return (
                pos['x'],  pos['y'],  pos['z'],
                quat['x'], quat['y'], quat['z'], quat['w'],
            )

        left, right = _extract('arm_left_link7'), _extract('arm_right_link7')
        self.latest_left_pos   = left[:3]
        self.latest_right_pos  = right[:3]
        self.latest_left_quat  = left[3:]
        self.latest_right_quat = right[3:]
        return left, right

    # ------------------------------------------------------------------ #
    #  Single stream loop — paced at RECORD_HZ, always running            #
    # ------------------------------------------------------------------ #

    def _stream_loop(self):
        next_tick = time.monotonic()

        while not self._stop_event.is_set():
            t = time.time()

            # --- End-effector poses (keeps latest_*_pos/quat fresh) ---
            try:
                left_status, right_status = self._get_hand_statuses()
            except Exception as e:
                print(f"  [stream] get_motion_status failed: {e}")
                left_status = right_status = None

            # --- Camera frames ---
            snapshot = {name: self._get_camera_frame(name) for name in CAMERA_NAMES}
            with self._frame_lock:
                for name, frame in snapshot.items():
                    if frame is not None:
                        self.latest_frames[name] = frame

            # --- Save from the streamed data while recording ---
            with self._rec_lock:
                if self._is_recording and self._rec is not None and left_status is not None:
                    self._record_tick(t, left_status, right_status, snapshot)

            # --- Pace to RECORD_HZ ---
            # Use Event.wait so shutdown() wakes us immediately.
            next_tick += self._interval
            sleep_for  = next_tick - time.monotonic()
            if sleep_for > 0:
                self._stop_event.wait(sleep_for)
            else:
                next_tick = time.monotonic()

    def _record_tick(self, t, left_status, right_status, snapshot):
        """Append one row to the active recording session. Caller holds _rec_lock."""
        rec = self._rec

        # Wait until every camera has produced at least one frame before
        # writing anything, so row counts stay in sync.
        for name, frame in snapshot.items():
            if frame is not None:
                rec['cam_ready'][name] = True
        if not all(rec['cam_ready'].values()):
            missing = [n for n, r in rec['cam_ready'].items() if not r]
            print(f"  waiting for cameras: {missing}")
            return

        # --- Record robot state ---
        rec['left'].append(left_status)
        rec['right'].append(right_status)
        rec['arm_joints'].append(self.robot.arm_joint_states()[0])
        rec['gripper'].append(self.robot.gripper_states()[0])
        rec['timestamps'].append(t)

        # --- Write camera frames (skip cameras that dropped this tick) ---
        for name in CAMERA_NAMES:
            frame = snapshot[name]
            if frame is None:
                continue
            if name not in rec['writers']:
                h, w   = frame.shape[:2]
                path   = str(rec['cameras_dir'] / f'{name}.mp4')
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                rec['writers'][name]    = cv2.VideoWriter(path, fourcc, self.record_hz, (w, h))
                rec['needs_bgr'][name]  = frame.ndim == 3
                print(f"  opened cameras/{name}.mp4  ({w}x{h})")

            rec['writers'][name].write(
                cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if rec['needs_bgr'][name] else frame
            )

    def _finalize(self, rec):
        """Flush video files and write npz/metadata for a finished session."""
        for w in rec['writers'].values():
            w.release()

        timestamps = rec['timestamps']
        n       = len(timestamps)
        left_a  = np.array(rec['left'],  dtype=np.float32)   # (N, 7)
        right_a = np.array(rec['right'], dtype=np.float32)   # (N, 7)

        if n == 0:
            print(f"  no frames recorded → {rec['episode_dir']}")
            return

        # robot_states.npz — one array per signal for easy per-key filtering
        np.savez(
            rec['episode_dir'] / 'robot_states.npz',
            timestamps = np.array(timestamps,           dtype=np.float64),  # (N,)     seconds
            left_pos   = left_a[:, :3],                                      # (N, 3)   metres
            left_quat  = left_a[:, 3:],                                      # (N, 4)   [qx,qy,qz,qw]
            right_pos  = right_a[:, :3],                                     # (N, 3)
            right_quat = right_a[:, 3:],                                     # (N, 4)
            arm_joints = np.array(rec['arm_joints'], dtype=np.float32),     # (N, 14)  radians
            gripper    = np.array(rec['gripper'],    dtype=np.float32),     # (N, 2)   0=open 1=closed
        )

        with open(rec['episode_dir'] / 'metadata.json', 'w') as f:
            json.dump({
                'fps':          self.record_hz,
                'n_frames':     n,
                'start_time':   rec['start_time'],
                'camera_names': CAMERA_NAMES,
            }, f, indent=2)

        print(f"  saved {n} frames → {rec['episode_dir']}")

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def start(self, episode_name=None):
        with self._rec_lock:
            if self._is_recording:
                print("Already recording.")
                return

            if episode_name is None:
                episode_name = f'episode_{sum(1 for _ in self.output_dir.glob("episode_*")):03d}'

            episode_dir = self.output_dir / episode_name
            cameras_dir = episode_dir / 'cameras'
            cameras_dir.mkdir(parents=True, exist_ok=True)

            self._rec = {
                'episode_dir': episode_dir,
                'cameras_dir': cameras_dir,
                'writers':     {},
                'needs_bgr':   {},
                'timestamps':  [],
                'left':        [],
                'right':       [],
                'arm_joints':  [],
                'gripper':     [],
                'cam_ready':   {name: False for name in CAMERA_NAMES},
                'start_time':  datetime.now().isoformat(),
            }
            self._is_recording = True

        print(f"Recording started → {episode_dir}")

    def stop(self):
        with self._rec_lock:
            if not self._is_recording:
                return
            print("Stopping recording...")
            self._is_recording = False
            rec = self._rec
            self._rec = None

        # Finalise outside the lock — the stream loop won't touch this session
        # anymore (it sees _rec is None), so writing files can't block streaming.
        self._finalize(rec)
        print("Recording stopped.")

    def shutdown(self):
        # 1. Signal the stream loop to stop — wakes it from sleep immediately.
        self._stop_event.set()

        # 2. Grab the active recording session under the lock before the thread
        #    can touch it for one more tick, then mark recording as stopped.
        rec = None
        with self._rec_lock:
            if self._is_recording:
                self._is_recording = False
                rec  = self._rec
                self._rec = None

        # 3. Wait for the stream thread to finish its current tick and exit.
        #    It will see _stop_event set and return from _stream_loop.
        if self._stream_thread and self._stream_thread.is_alive():
            self._stream_thread.join(timeout=5.0)
            if self._stream_thread.is_alive():
                print("[shutdown] stream thread did not stop in time — proceeding anyway")
            self._stream_thread = None

        # 4. Finalize the recording now that the stream thread is gone.
        #    No concurrency risk here — only we hold rec.
        if rec is not None:
            self._finalize(rec)

        # 5. Close SDK resources last.
        try:
            self.camera.close()
        except Exception as e:
            print(f"[shutdown] camera.close: {e}")
        try:
            self.robot.shutdown()
        except Exception as e:
            print(f"[shutdown] robot.shutdown: {e}")


if __name__ == '__main__':
    collector = RobotDataCollector(output_dir='recordings')

    def _on_sigint(*_):
        collector.stop()
        collector.shutdown()
        raise SystemExit(0)

    signal.signal(signal.SIGINT, _on_sigint)
    collector.start()
    print("Press Ctrl+C to stop.")
    signal.pause()
