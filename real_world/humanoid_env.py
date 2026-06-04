"""Env layer for deploying the left_arm_ee_image policy on the G1 robot.

This mirrors diffusion_policy/real_world/real_env.py (RealEnv): a single object
that owns the SDK resources and their background threads, and exposes a small
verb-based API so the caller never touches locks, buffers, or the SDK directly.

Async producer/consumer design (two threads owned by the env):
  * collection thread (producer) -> keeps the latest two head/hand frames and
    the latest two left-EE states fresh, paced at RECORD_HZ.
  * execution  thread (consumer) -> drains the most recent predicted action
    chunk and commands the left arm + gripper, paced at RECORD_HZ.

The caller owns only the inference thread:
    obs = env.get_obs(); resp = <server>; env.submit_actions(resp)

Data-collection / streaming logic is copied from
MDM_data_collection/robot_data_collect.py, which is the tested, reliable path.
"""

import copy
import threading
import time

import numpy as np
from a2d_sdk.robot import RobotDds as Robot, CosineCamera as Camera, RobotController

RECORD_HZ = 30

# A camera auto-switches OFF if no consumer has requested it within this window.
CAMERA_IDLE_TIMEOUT = 2.0

# Camera roles for the left_arm_ee_image policy obs.
AGENT_CAMERA = "head"        # -> agentview_image
HAND_CAMERA = "hand_left"    # -> robot0_eye_in_hand_image
INFERENCE_CAMERAS = [AGENT_CAMERA, HAND_CAMERA]


# Fallback intrinsics when the SDK can't supply them (ported from
# gui/camera_panel.py:get_default_camera_intrinsics so the env owns intrinsics).
_DEFAULT_INTRINSICS = {
    "head": {'width': 1280, 'height': 800, 'fx': 900.0, 'fy': 900.0,
             'cx': 640.0, 'cy': 400.0, 'distortion': [0.0, 0.0, 0.0, 0.0, 0.0]},
    "head_depth": {'width': 1280, 'height': 800, 'fx': 900.0, 'fy': 900.0,
                   'cx': 640.0, 'cy': 400.0, 'distortion': [0.0, 0.0, 0.0, 0.0, 0.0]},
    "hand_left": {'width': 320, 'height': 240, 'fx': 300.0, 'fy': 300.0,
                  'cx': 160.0, 'cy': 120.0, 'distortion': [0.0, 0.0, 0.0, 0.0, 0.0]},
    "hand_right": {'width': 320, 'height': 240, 'fx': 300.0, 'fy': 300.0,
                   'cx': 160.0, 'cy': 120.0, 'distortion': [0.0, 0.0, 0.0, 0.0, 0.0]},
    "head_center_fisheye": {'width': 640, 'height': 480, 'fx': 400.0, 'fy': 400.0,
                            'cx': 320.0, 'cy': 240.0, 'distortion': [0.1, -0.1, 0.0, 0.0, 0.0]},
}


def _default_camera_intrinsics(name):
    """Default intrinsics for a camera (falls back to 'head' for unknown names)."""
    return _DEFAULT_INTRINSICS.get(name, _DEFAULT_INTRINSICS["head"])


def rot6d_to_quat(rot6d):
    """6D rotation (first two rotation-matrix columns) -> quaternion [x, y, z, w].

    Mirrors diffusion_policy/scripts/build_left_arm_ee_replay_buffer.py:_6d_rot_to_quat
    for a single sample, so the live decode matches how the training data was built.
    """
    rot6d = np.asarray(rot6d, dtype=np.float64)
    c1 = rot6d[:3]
    c2 = rot6d[3:6]
    # Gram-Schmidt orthonormalisation
    c1 = c1 / (np.linalg.norm(c1) + 1e-8)
    c2 = c2 - np.dot(c2, c1) * c1
    c2 = c2 / (np.linalg.norm(c2) + 1e-8)
    c3 = np.cross(c1, c2)
    m = np.stack([c1, c2, c3], axis=-1)  # rotation matrix (columns = c1,c2,c3)
    t = m[0, 0] + m[1, 1] + m[2, 2]
    if t > 0:
        s = 0.5 / np.sqrt(t + 1.0)
        q = [(m[2, 1] - m[1, 2]) * s, (m[0, 2] - m[2, 0]) * s,
             (m[1, 0] - m[0, 1]) * s, 0.25 / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = [0.25 * s, (m[0, 1] + m[1, 0]) / s,
             (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s]
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = [(m[0, 1] + m[1, 0]) / s, 0.25 * s,
             (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s]
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = [(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s,
             0.25 * s, (m[0, 1] - m[1, 0]) / s]
    return [float(v) for v in q]


class HumanoidEnv:
    """Owns SDK resources + the collection/execution threads for live inference."""

    def __init__(self,
                 robot=None, robot_controller=None, camera=None,
                 cameras=INFERENCE_CAMERAS,
                 frequency=RECORD_HZ,
                 pos_alpha=0.5,
                 command_lifetime=2.0,
                 idle_timeout=CAMERA_IDLE_TIMEOUT):
        # Reuse injected SDK handles if given (e.g. the GUI already holds them),
        # otherwise create our own — same construction as RobotDataCollector.
        # We only tear down resources we created; injected handles are the
        # caller's to manage (the GUI shares them across panels).
        self._owns_robot = robot is None
        self._owns_camera = camera is None
        self.robot = robot if robot is not None else Robot()
        self.robot_controller = robot_controller if robot_controller is not None else RobotController()
        self.camera = camera if camera is not None else Camera(list(cameras))
        if self._owns_robot or self._owns_camera:
            time.sleep(1.0)  # let freshly-created DDS / camera resources come up

        # Superset of names the SDK Camera was built with (what can be fetched).
        # NOT the active set — cameras are switched ON on demand (see request()).
        self.cameras = list(cameras)
        self.dt = 1.0 / frequency
        self.pos_alpha = pos_alpha          # position low-pass (1.0 = no smoothing)
        self.command_lifetime = command_lifetime
        self.idle_timeout = idle_timeout    # auto-off a camera idle longer than this

        # ---- shared state (replaces robot_info.lock + scattered GUI buffers) ----
        self._lock = threading.Lock()
        # Per-camera on/off switch: a camera is fetched only while ACTIVE; it goes
        # active when a consumer request()s it and is evicted after idle_timeout.
        self._active = set()                                   # cameras currently ON
        self._last_requested = {}                              # name -> time.monotonic()
        self._frames = {}                                      # name -> rolling [prev, cur]
        self._intrinsics = {}                                  # name -> intrinsics dict (lazy)
        self._last_two_ee_states = None                        # [s_{t-1}, s_t]
        self._obs_timestamp = 0.0

        self._action_queue = []            # most recent predicted chunk, FIFO drained
        self._smoothed_pos = None

        self._stop_event = threading.Event()
        self._collect_thread = None
        self._exec_thread = None

    # ===================== lifecycle (RealEnv.start/stop/__enter__) =====================
    @property
    def is_ready(self):
        """Ready for inference once the two policy cameras + EE state are populated."""
        with self._lock:
            frames_ready = all(
                self._frames.get(n) is not None for n in (AGENT_CAMERA, HAND_CAMERA))
            return frames_ready and self._last_two_ee_states is not None

    # ===================== consumer-facing camera switch =====================
    def request(self, name):
        """Mark a camera as wanted this cycle and switch it ON.

        Idempotent and cheap — consumers call this every loop tick to keep a
        camera alive. Names outside the SDK's constructed set are ignored.
        """
        if name not in self.cameras:
            return
        with self._lock:
            self._last_requested[name] = time.monotonic()
            self._active.add(name)

    def get_frames(self, name):
        """request(name) + return a copy of the rolling [prev, cur] pair, or None.

        Returns None until the collect loop has fetched the camera at least once
        (it was just switched on, or it is still warming up).
        """
        self.request(name)
        with self._lock:
            pair = self._frames.get(name)
            return copy.deepcopy(pair) if pair else None

    def get_frame(self, name):
        """request(name) + return the latest single frame (copy), or None."""
        pair = self.get_frames(name)
        return pair[-1] if pair else None

    def get_intrinsics(self, name):
        """Camera intrinsics, fetched once from the SDK then cached (env-owned)."""
        with self._lock:
            cached = self._intrinsics.get(name)
        if cached is not None:
            return cached
        # SDK call done outside the lock (it may block).
        info = None
        try:
            if hasattr(self.camera, 'get_camera_info'):
                info = self.camera.get_camera_info(name)
            elif hasattr(self.camera, 'get_intrinsics'):
                info = self.camera.get_intrinsics(name)
        except Exception as e:
            print(f"[HumanoidEnv.get_intrinsics] {name}: {e}")
            info = None
        if not info:
            info = _default_camera_intrinsics(name)
        with self._lock:
            self._intrinsics[name] = info
        return info

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def start(self):
        self._stop_event.clear()
        self._collect_thread = threading.Thread(target=self._collect_loop, daemon=True)
        self._exec_thread = threading.Thread(target=self._exec_loop, daemon=True)
        self._collect_thread.start()
        self._exec_thread.start()
        print("HumanoidEnv: collection + execution threads started.")

    def stop(self):
        self._stop_event.set()
        for t in (self._collect_thread, self._exec_thread):
            if t is not None and t.is_alive():
                t.join(timeout=5.0)
        self._collect_thread = self._exec_thread = None
        # Only close resources we created; injected handles belong to the caller.
        if self._owns_camera:
            try:
                self.camera.close()
            except Exception as e:
                print(f"[HumanoidEnv.stop] camera.close: {e}")
        if self._owns_robot:
            try:
                self.robot.shutdown()
            except Exception as e:
                print(f"[HumanoidEnv.stop] robot.shutdown: {e}")

    # ===================== producer: collection loop =====================
    # Pacing + SDK reads copied from RobotDataCollector._stream_loop.
    def _collect_loop(self):
        next_tick = time.monotonic()
        while not self._stop_event.is_set():
            now = time.time()

            # Evict cameras idle longer than the timeout, then snapshot the active
            # set so each ON camera is fetched exactly once this tick.
            now_mono = time.monotonic()
            with self._lock:
                stale = [n for n, t in self._last_requested.items()
                         if now_mono - t > self.idle_timeout]
                for n in stale:
                    self._active.discard(n)
                    self._last_requested.pop(n, None)
                    self._frames.pop(n, None)   # clear so re-activation never serves a stale frame
                active = set(self._active)

            try:
                ee_state = self._read_left_ee_state()
            except Exception as e:
                print(f"  [collect] get_motion_status failed: {e}")
                ee_state = None

            frames = self._read_frames(active)

            with self._lock:
                self._obs_timestamp = now
                for name, pair in frames.items():
                    self._frames[name] = pair
                if ee_state is not None:
                    if self._last_two_ee_states:
                        self._last_two_ee_states = [self._last_two_ee_states[-1], ee_state]
                    else:
                        self._last_two_ee_states = [ee_state, ee_state]

            next_tick += self.dt
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                self._stop_event.wait(sleep_for)
            else:
                next_tick = time.monotonic()

    def _read_left_ee_state(self):
        """Left EE pose + gripper -> [pos(3), quat xyzw(4), grip(1)].

        Same extraction as RobotDataCollector._get_hand_statuses (left arm only).
        """
        frame = self.robot_controller.get_motion_status()['frames']['arm_left_link7']
        pos = frame['position']
        quat = frame['orientation']['quaternion']
        gripper_states, _ = self.robot.gripper_states()
        return [
            pos['x'], pos['y'], pos['z'],
            quat['x'], quat['y'], quat['z'], quat['w'],
            1.0 if gripper_states[0] > 0.5 else 0.0,
        ]

    def _read_frames(self, active):
        """Latest frame per ACTIVE camera, kept as a rolling [prev, cur] pair.

        camera.get_latest_image returns (image, timestamp); the first frame is
        None (SDK note 9.6), so we skip until a real frame arrives.
        """
        out = {}
        for name in active:
            image, _ = self.camera.get_latest_image(name)
            if image is None:
                continue
            prev = self._frames.get(name)
            out[name] = [prev[-1], image] if prev else [image, image]
        return out

    # ===================== obs snapshot for the caller (RealEnv.get_obs) =====================
    def get_obs(self):
        """Time-aligned snapshot for one inference, or None if not ready yet.

        Returns dict with role-mapped image pairs + the last two EE states:
            agent_imgs: [head_{t-1}, head_t]
            hand_imgs:  [hand_left_{t-1}, hand_left_t]
            state:      [ee_{t-1}, ee_t]   each [pos(3), quat(4), grip(1)]
            timestamp:  float (seconds)
        """
        # Keep the two policy cameras warm while inference polls get_obs().
        self.request(AGENT_CAMERA)
        self.request(HAND_CAMERA)
        with self._lock:
            if self._last_two_ee_states is None:
                return None
            if any(self._frames.get(n) is None for n in (AGENT_CAMERA, HAND_CAMERA)):
                return None
            return {
                'agent_imgs': copy.deepcopy(self._frames[AGENT_CAMERA]),
                'hand_imgs': copy.deepcopy(self._frames[HAND_CAMERA]),
                'state': copy.deepcopy(self._last_two_ee_states),
                'timestamp': self._obs_timestamp,
            }

    # ===================== caller -> env: hand off a prediction =====================
    def submit_actions(self, action_chunk):
        """Replace the pending action queue with the newest predicted chunk.

        action_chunk: list of rows [eef_pos(3), 6D_rot(6), gripper(1)].
        """
        with self._lock:
            self._action_queue = copy.deepcopy(list(action_chunk)) if action_chunk else []

    # ===================== consumer: execution loop (RealEnv.exec_actions) =====================
    def _exec_loop(self):
        while not self._stop_event.is_set():
            with self._lock:
                action = self._action_queue.pop(0) if self._action_queue else None
            if action is None:
                self._stop_event.wait(self.dt)
                continue

            pos, quat, grip = self._decode_ee_action(action)
            pos = np.asarray(pos, dtype=np.float64)
            if self._smoothed_pos is None:
                self._smoothed_pos = pos
            else:
                self._smoothed_pos = self.pos_alpha * pos + (1.0 - self.pos_alpha) * self._smoothed_pos

            self._command_left_ee(self._smoothed_pos, quat, grip)
            self._stop_event.wait(self.dt)

    def _decode_ee_action(self, action):
        """action row [eef_pos(3), 6D_rot(6), gripper(1)] -> (pos(3), quat(4), grip)."""
        return action[:3], rot6d_to_quat(action[3:9]), action[9]

    def _command_left_ee(self, pos, quat, gripper):
        """Command the left arm to an absolute EE pose + set the left gripper.

        pos: (x, y, z) m; quat: (qx, qy, qz, qw); gripper: scalar (>0.5 -> close).
        Uses set_end_effector_pose_control (SDK 8.1) for the arm and move_gripper
        (SDK 7.2) for the gripper. move_gripper sets BOTH grippers, so we read the
        current right value and leave it untouched.
        """
        self.robot_controller.set_end_effector_pose_control(
            lifetime=self.command_lifetime,
            control_group=["left_arm"],
            left_pose={
                'x': float(pos[0]), 'y': float(pos[1]), 'z': float(pos[2]),
                'qx': float(quat[0]), 'qy': float(quat[1]),
                'qz': float(quat[2]), 'qw': float(quat[3]),
            },
            right_pose=None,
        )
        left_cmd = 1.0 if gripper > 0.5 else 0.0
        try:
            cur, _ = self.robot.gripper_states()
            right_cmd = float(cur[1]) if cur is not None and len(cur) > 1 else 0.0
        except Exception:
            right_cmd = 0.0
        self.robot.move_gripper([left_cmd, right_cmd])
