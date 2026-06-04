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
CAMERA_IDLE_TIMEOUT = 5.0

# Camera roles for the left_arm_ee_image policy obs.
AGENT_CAMERA = "head"        # -> agentview_image
HAND_CAMERA = "hand_left"    # -> robot0_eye_in_hand_image
INFERENCE_CAMERAS = [AGENT_CAMERA, HAND_CAMERA]

# Every camera name the SDK knows about. A camera is only SUBSCRIBED (i.e. costs
# DDS bandwidth) while we hold a live CosineCamera object for it — see the
# per-camera dynamic subscription in HumanoidEnv. This list is just the set of
# names a consumer is allowed to request.
KNOWN_CAMERAS = ["head", "head_depth", "hand_left", "hand_right", "head_center_fisheye"] 
# Tho more camera angles exist, limited to these for simplisity


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
                 robot=None, robot_controller=None,
                 cameras=(),
                 allowed_cameras=KNOWN_CAMERAS,
                 frequency=RECORD_HZ,
                 pos_alpha=0.5,
                 command_lifetime=2.0,
                 idle_timeout=CAMERA_IDLE_TIMEOUT):
        # Robot/controller may be injected (the GUI shares them across panels);
        # we only tear down what we created. Cameras are NOT injected — the env is
        # the sole owner of camera subscriptions and manages one CosineCamera per
        # camera dynamically (see _reconcile_cameras), so nothing streams until a
        # consumer request()s it.
        self._owns_robot = robot is None
        self.robot = robot if robot is not None else Robot()
        self.robot_controller = robot_controller if robot_controller is not None else RobotController()
        if self._owns_robot:
            time.sleep(1.0)  # let freshly-created DDS resources come up

        # Names a consumer is allowed to request (validation only — not subscribed).
        self.cameras = list(allowed_cameras)
        # "Pinned" cameras stay ON for the env's whole life (never idle-evicted),
        # e.g. data-collection cameras. Empty for the GUI -> nothing on at launch.
        self._pinned = set(cameras)
        self.dt = 1.0 / frequency
        self.pos_alpha = pos_alpha          # position low-pass (1.0 = no smoothing)
        self.command_lifetime = command_lifetime
        self.idle_timeout = idle_timeout    # auto-off a camera idle longer than this

        # ---- shared state (replaces robot_info.lock + scattered GUI buffers) ----
        self._lock = threading.Lock()
        # Per-camera on/off switch: a camera is SUBSCRIBED (live CosineCamera object
        # in self._cams -> costs bandwidth) only while ACTIVE. It goes active when a
        # consumer request()s it, and is evicted (object closed) after idle_timeout.
        self._active = set()                                   # cameras requested recently
        self._last_requested = {}                              # name -> time.monotonic()
        self._cams = {}                                        # name -> CosineCamera([name]) (live subscription)
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
    def inf_ready(self):
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
        print(f"processing request for camera {name}")

        if name not in self.cameras:
            print(f"[HumanoidEnv] camera name not recognized: {name}")
            return
        with self._lock:
            self._last_requested[name] = time.monotonic()
            if name not in self._active:        # log only on the OFF->ON transition
                print(f"[HumanoidEnv] activated camera: {name}")
                self._active.add(name)
            else:
                print(f"[HumanoidEnv] camera: {name} is already active")
                print(f"current active cameras: {self.active_cameras}")


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

    def active_cameras(self):
        """Names currently SUBSCRIBED (a live CosineCamera object exists -> streaming).

        This is what the GUI's live indicator shows: every camera the humanoid env
        is actually pulling from the robot right now (display ticks + inference + pinned).
        """
        with self._lock:
            return sorted(self._cams.keys())

    def get_intrinsics(self, name):
        """Camera intrinsics, fetched once from the live SDK object then cached.

        Falls back to defaults when the camera isn't currently subscribed (intrinsics
        are static, so a sensible default is fine while the camera is OFF).
        """
        with self._lock:
            cached = self._intrinsics.get(name)
            cam = self._cams.get(name)
        if cached is not None:
            return cached
        info = None
        if cam is not None:                     # only query a live subscription
            try:
                if hasattr(cam, 'get_camera_info'):
                    info = cam.get_camera_info(name)
                elif hasattr(cam, 'get_intrinsics'):
                    info = cam.get_intrinsics(name)
            except Exception as e:
                print(f"[HumanoidEnv] [HumanoidEnv.get_intrinsics] {name}: {e}")
                info = None
        if not info:
            info = _default_camera_intrinsics(name)
            # don't cache the default — let a real value replace it once the camera is on
            return info
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
        print("[HumanoidEnv]: collection + execution threads started.")

    def stop(self):
        self._stop_event.set()
        for t in (self._collect_thread, self._exec_thread):
            if t is not None and t.is_alive():
                t.join(timeout=5.0)
        self._collect_thread = self._exec_thread = None
        # Close every live camera subscription (env owns all of them).
        for name, cam in list(self._cams.items()):
            try:
                cam.close()
            except Exception as e:
                print(f"[HumanoidEnv.stop] camera.close({name}): {e}")
        self._cams.clear()
        if self._owns_robot:
            try:
                self.robot.shutdown()
            except Exception as e:
                print(f"[HumanoidEnv.stop] robot.shutdown: {e}")

    # ===================== per-camera dynamic subscription =====================
    def _reconcile_cameras(self, desired):
        """Make the set of live CosineCamera objects match `desired`.

        Opens a dedicated CosineCamera([name]) for each newly-wanted camera (this
        is the actual DDS subscription that costs bandwidth) and closes the object
        for any camera no longer wanted (stopping its stream). Using one object per
        camera means toggling one camera never disturbs the others' streams.

        Runs only in the collect thread, so self._cams has a single mutator; reads
        of self._cams elsewhere take self._lock.
        """
        current = set(self._cams.keys())
        for name in current - desired:
            cam = self._cams.pop(name)
            try:
                cam.close()
            except Exception as e:
                print(f"[HumanoidEnv] camera.close({name}) failed: {e}")
            with self._lock:
                self._frames.pop(name, None)        # no stale frame on re-activation
                self._intrinsics.pop(name, None)
            print(f"[HumanoidEnv] camera unsubscribed (OFF): {name}")
        for name in desired - current:
            try:
                cam = Camera([name])                # <-- the per-camera DDS subscription
            except Exception as e:
                print(f"[HumanoidEnv] camera open({name}) failed: {e}")
                continue
            with self._lock:
                self._cams[name] = cam
            print(f"[HumanoidEnv] camera subscribed (ON): {name}")

    # ===================== producer: collection loop =====================
    # Pacing + SDK reads copied from RobotDataCollector._stream_loop.
    def _collect_loop(self):
        next_tick = time.monotonic()
        while not self._stop_event.is_set():
            now = time.time()

            # Evict cameras idle longer than the timeout (pinned never evict), then
            # the desired subscription set = recently-requested + pinned.
            now_mono = time.monotonic()
            with self._lock:
                stale = [n for n, t in self._last_requested.items()
                         if now_mono - t > self.idle_timeout and n not in self._pinned]
                for n in stale:
                    self._active.discard(n)
                    self._last_requested.pop(n, None)
                desired = set(self._active) | self._pinned

            # Open/close CosineCamera objects to match desired (controls bandwidth).
            self._reconcile_cameras(desired)

            try:
                ee_state = self._read_left_ee_state()
            except Exception as e:
                print(f"[HumanoidEnv]  [collect] get_motion_status failed: {e}")
                ee_state = None

            frames = self._read_frames(desired)

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
        """Latest frame per live camera, kept as a rolling [prev, cur] pair.

        Each camera has its own CosineCamera object; get_latest_image returns
        (image, timestamp) and the first frame is None (SDK note 9.6), so we skip
        until a real frame arrives.
        """
        out = {}
        for name in active:
            cam = self._cams.get(name)
            if cam is None:                 # just requested; object opens next tick
                continue
            image, _ = cam.get_latest_image(name)
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
        print("[HumanoidEnv] requesting hand and head camera")
        self.request(AGENT_CAMERA)
        self.request(HAND_CAMERA)
        with self._lock:
            if self._last_two_ee_states is None:
                print("[HumanoidEnv] Waiting for EE states")
                return None
            if any(self._frames.get(n) is None for n in (AGENT_CAMERA, HAND_CAMERA)):
                print("[HumanoidEnv] Wait for Camera")
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
                print("[HumanoidEnv] no new action received, skipping...")
                continue #if no information: skip rest and restart loop
            
            print(f"[HumanoidEnv] new action recieved {action}")
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
