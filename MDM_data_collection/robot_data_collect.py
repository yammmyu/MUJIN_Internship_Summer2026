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
    def __init__(self, output_dir="recordings", record_hz=RECORD_HZ,
                 robot=None, camera=None, robot_controller=None):
        self.robot            = robot            or Robot()
        self.camera           = camera           or Camera(CAMERA_NAMES)
        self.robot_controller = robot_controller or RobotController()
        if robot is None or camera is None:
            time.sleep(1.0)

        self.output_dir = pathlib.Path(output_dir)
        self.record_hz  = record_hz
        self._interval  = 1.0 / record_hz

        self._latest_frames = {name: None for name in CAMERA_NAMES}
        self._frame_lock    = threading.Lock()

        self._is_recording  = False
        self._camera_thread = None
        self._record_thread = None

        # Latest end-effector positions — updated by _get_hand_statuses() at
        # RECORD_HZ while recording; None when no data has been fetched yet.
        self.latest_left_pos  = None  # (x, y, z)
        self.latest_right_pos = None  # (x, y, z)

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
        self.latest_left_pos  = left[:3]
        self.latest_right_pos = right[:3]
        return left, right

    # ------------------------------------------------------------------ #
    #  Camera background thread — keeps _latest_frames fresh at ~100 Hz  #
    # ------------------------------------------------------------------ #

    def _camera_loop(self):
        while self._is_recording:
            for name in CAMERA_NAMES:
                image, _ = self.camera.get_latest_image(name)
                if image is not None:
                    with self._frame_lock:
                        self._latest_frames[name] = image.copy()
            time.sleep(0.01)

    # ------------------------------------------------------------------ #
    #  Recording loop — paced at RECORD_HZ                                #
    # ------------------------------------------------------------------ #

    def _record_loop(self, episode_dir: pathlib.Path):
        cameras_dir = episode_dir / 'cameras'
        cameras_dir.mkdir(parents=True, exist_ok=True)

        writers         = {}
        timestamps      = []
        left_list       = []
        right_list      = []
        arm_joints_list = []
        gripper_list    = []

        next_tick  = time.monotonic()
        start_time = datetime.now().isoformat()

        while self._is_recording:
            t = time.time()

            # --- Camera frames (snapshot first) ---
            with self._frame_lock:
                snapshot = {name: self._latest_frames[name] for name in CAMERA_NAMES}

            # Skip this tick if any camera hasn't delivered its first frame yet.
            # This keeps robot_states.npz row count in sync with video frame count.
            if any(frame is None for frame in snapshot.values()):
                time.sleep(0.001)
                continue

            # --- End-effector poses (single SDK call for both arms) ---
            left_status, right_status = self._get_hand_statuses()
            left_list.append(left_status)    # (7,): x,y,z,qx,qy,qz,qw
            right_list.append(right_status)  # (7,): x,y,z,qx,qy,qz,qw

            # --- Joint and gripper states ---
            arm_joints_list.append(list(self.robot.arm_joint_states()[0]))  # 14 floats (rad)
            gripper_list.append(list(self.robot.gripper_states()[0]))        # 2 floats (0=open,1=closed)

            timestamps.append(t)

            for name, frame in snapshot.items():
                if frame is None:
                    continue
                if name not in writers:
                    h, w   = frame.shape[:2]
                    path   = str(cameras_dir / f'{name}.mp4')
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writers[name] = cv2.VideoWriter(path, fourcc, self.record_hz, (w, h))
                    print(f"  opened cameras/{name}.mp4  ({w}x{h})")

                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if frame.ndim == 3 else frame
                writers[name].write(bgr)

            # --- Pace to RECORD_HZ ---
            next_tick += self._interval
            sleep_for  = next_tick - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_tick = time.monotonic()

        # --- Flush video files ---
        for w in writers.values():
            w.release()

        n       = len(timestamps)
        left_a  = np.array(left_list,  dtype=np.float32)   # (N, 7)
        right_a = np.array(right_list, dtype=np.float32)   # (N, 7)

        # robot_states.npz — one array per signal for easy per-key filtering
        np.savez(
            episode_dir / 'robot_states.npz',
            timestamps = np.array(timestamps,        dtype=np.float64),  # (N,)     seconds
            left_pos   = left_a[:, :3],                                   # (N, 3)   metres
            left_quat  = left_a[:, 3:],                                   # (N, 4)   [qx,qy,qz,qw]
            right_pos  = right_a[:, :3],                                  # (N, 3)
            right_quat = right_a[:, 3:],                                  # (N, 4)
            arm_joints = np.array(arm_joints_list, dtype=np.float32),    # (N, 14)  radians
            gripper    = np.array(gripper_list,    dtype=np.float32),    # (N, 2)   0=open 1=closed
        )

        with open(episode_dir / 'metadata.json', 'w') as f:
            json.dump({
                'fps':          self.record_hz,
                'n_frames':     n,
                'start_time':   start_time,
                'camera_names': CAMERA_NAMES,
            }, f, indent=2)

        print(f"  saved {n} frames → {episode_dir}")

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def start(self, episode_name=None):
        if self._is_recording:
            print("Already recording.")
            return

        if episode_name is None:
            existing     = sorted(self.output_dir.glob('episode_*'))
            episode_name = f'episode_{len(existing):03d}'

        episode_dir        = self.output_dir / episode_name
        self._is_recording = True

        self._camera_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self._camera_thread.start()

        self._record_thread = threading.Thread(
            target=self._record_loop, args=(episode_dir,), daemon=False
        )
        self._record_thread.start()
        print(f"Recording started → {episode_dir}")

    def stop(self):
        if not self._is_recording:
            return
        print("Stopping recording...")
        self._is_recording = False
        if self._camera_thread:
            self._camera_thread.join()
            self._camera_thread = None
        if self._record_thread:
            self._record_thread.join()
            self._record_thread = None
        print("Recording stopped.")

    def shutdown(self):
        self.stop()
        self.robot.shutdown()


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
