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
import json
import pathlib
import threading
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np

from real_world.ik import IKQuery, build_solver, rot6d_to_quat as _ik_rot6d_to_quat

# a2d_sdk only exists on the robot machine. A sim-only machine (pybullet but no SDK) still
# needs to import this module for the IK/exec path, so guard the import. The sim runner never
# constructs these (it injects a _NoRobot stand-in); only the GUI / live robot path does.
try:
    from a2d_sdk.robot import RobotDds as Robot, CosineCamera as Camera, RobotController
except ImportError:
    Robot = Camera = RobotController = None

RECORD_HZ = 30

# ----------------------------------------------------------------------------- #
#  Safety limits for the real-robot release path (see the safety-review fixes).  #
#  Conservative defaults for first hardware bring-up; tune up only after runs.   #
# ----------------------------------------------------------------------------- #
# Max per-tick change of any single arm joint on the hardware path (rad). The
# validation pass subdivides to respect this and the release loop clamps to it,
# so a step-change target becomes a bounded ramp instead of a snap. (C5)
MAX_JOINT_STEP = 0.007 #0.02         # ~1.8 deg per tick @ RECORD_HZ -> ~54 deg/s ceiling
# Orientation EMA factor toward the new target quaternion (0..1; 1 = no smoothing). (H3)
QUAT_ALPHA = 0.5
# Workspace envelope (firmware EE frame, metres) the policy's target EE pos must lie in. A
# target outside is rejected (never sent to IK/robot). Generous box; tighten per workspace. (H4)
WORKSPACE_AABB = ((-0.20, 0.85), (-0.20, 1.10), (0.40, 1.30))   # (x_lo,x_hi),(y..),(z..)
# A live observation older than this (no real change in EE pose / camera) is "stale": the
# auto-inference loop aborts rather than command the arm from frozen sensor data. (H2)
STALE_TIMEOUT = 0.5             # seconds

STEP_TIME = 1/120 #each sub step will be executed over 0.05 seconds

# Gripper is BINARY open/close. The policy emits a noisy raw [0,~85] gripper signal (transient
# spikes exist), so at inference we binarize it to {0,1} (see InferenceController): only a
# (near-)fully-closed reading counts as CLOSED. Everything downstream (sim preview, validation,
# staging, release) then carries {0,1}; the LEFT gripper is then driven via the 'gripper' group of
# trajectory_tracking_control (never move_gripper), so the right gripper is never addressed.
GRIPPER_CLOSE_THRESH = 10.0     # raw gripper value at/above which we call it CLOSED (-> 1)
# Cameras captured to disk during data collection (mirrors the old
# RobotDataCollector.CAMERA_NAMES). build_dataset.py only consumes head +
# hand_left; hand_right is recorded for future use.
RECORD_CAMERAS = ["hand_left", "hand_right", "head"]

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

    Delegates to the pinocchio-based converter in real_world.ik (Gram-Schmidt to a proper
    rotation matrix, then an exact matrix->quaternion). The previous hand-written
    matrix->quaternion here was numerically wrong for many rotations (round-trip rotation error
    up to ~2.4 vs ~1e-8 for the correct one), which corrupted decoded EE orientations and made
    ~20% of recorded targets spuriously IK-unreachable.
    """
    return np.asarray(_ik_rot6d_to_quat(rot6d), dtype=np.float64)


def _slerp(q0, q1, t):
    """Spherical-linear interpolation between two quaternions (any consistent layout; we use
    xyzw). t in [0,1]; t=0 -> q0, t=1 -> q1. Takes the shorter arc."""
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    q0 = q0 / (np.linalg.norm(q0) + 1e-12)
    q1 = q1 / (np.linalg.norm(q1) + 1e-12)
    d = float(np.dot(q0, q1))
    if d < 0.0:                      # shorter arc
        q1 = -q1
        d = -d
    if d > 0.9995:                   # nearly aligned -> linear + renormalize
        out = q0 + t * (q1 - q0)
        return out / (np.linalg.norm(out) + 1e-12)
    th0 = np.arccos(d)
    s0 = np.sin((1.0 - t) * th0) / np.sin(th0)
    s1 = np.sin(t * th0) / np.sin(th0)
    return s0 * q0 + s1 * q1


def _smooth_quat_step(prev, quat, alpha):
    """One step of orientation smoothing (H3): SLERP the previous target toward the new quat
    by `alpha` (1.0 = no smoothing). `prev` None -> pass the new quat through unchanged."""
    q = np.asarray(quat, dtype=np.float64)
    q = q / (np.linalg.norm(q) + 1e-12)
    if prev is None:
        return q
    return _slerp(prev, q, alpha)


class HumanoidEnv:
    """Owns SDK resources + the collection/execution threads for live inference."""

    def __init__(self,
                 robot=None, robot_controller=None,
                 cameras=(),
                 allowed_cameras=KNOWN_CAMERAS,
                 frequency=RECORD_HZ,
                 pos_alpha=0.5,
                 command_lifetime=2.0,
                 idle_timeout=CAMERA_IDLE_TIMEOUT,
                 output_dir="recordings",
                 sim=None,
                 solver=None,
                 real=False,
                 seed_q=None):
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

        # --- IK-based execution: sim backend + a sim-BEFORE-robot release pipeline ---
        # Non-negotiable invariant: an action can reach the real robot ONLY after the sim has
        # stepped through it, self-collision-checked it, and read back the achieved joints.
        # Enforced structurally, not by discipline:
        #   * SimEnv.validate (on the sim thread) is the ONLY producer of self._last_sim_traj and
        #     self._staged_release; it cannot run without a live sim, so "no sim" => nothing is
        #     ever releasable.
        #   * Only the release entry points enqueue onto self._robot_q, and only by copying
        #     freshly sim-validated substeps (E-stop-guarded, C5 ramp-in): release_to_robot()
        #     (one-shot, whole trajectory) and release_n_substeps/ release_remaining_substeps
        #     (drain the accumulating self._staged_release buffer).
        #   * _release_loop is the ONLY caller of run_trajectory_control, draining _robot_q in
        #     batches; only the LEFT arm is ever commanded, so the right arm is never moved.
        #   * _exec_loop drives the sim only (auto/replay preview); it never touches the robot.
        self.sim = sim                               # SimEnv (command/validate), or None
        self.solver = solver if solver is not None else build_solver()
        # Left-arm joint limits (rad), captured once so the execution path can clamp commands
        # without reaching into the IK solver (it only needs the bounds, not the solver).
        self._jlower = np.asarray(self.solver.m.lower, dtype=np.float64).copy()
        self._jupper = np.asarray(self.solver.m.upper, dtype=np.float64).copy()
        self._real = real
        # IK warm-start seed (7,). Seeded from the recording's first arm joints when given,
        # else the limit-clipped zero config.
        self._last_q = (np.asarray(seed_q, dtype=np.float64).copy()
                        if seed_q is not None else self.solver.m.clip(np.zeros(7)))
        self._quat_prev = None                       # previous target quat for SLERP smoothing (H3)
        # Release pipeline state (guarded by self._lock):
        self._last_sim_traj = []                     # [(q7_achieved, grip)] last validated traj
        self._validation_id = 0                      # bumped on each successful validation
        self._released_id = -1                       # validation_id already released (one-shot, H1)
        self._robot_q = deque()                      # released, sim-validated cmds pending on robot
        # Substep-by-substep release buffer: each successful validate_and_stage APPENDS its
        # sim-achieved substeps here ("ready to release"). release_next_substep / release_remaining_substeps
        # move substeps from here onto _robot_q, which _release_loop hands to trajectory_tracking_control.
        self._staged_release = deque()               # [(q7_achieved, grip)] validated, awaiting release
        self._release_thread = None
        self._estop = threading.Event()              # latched stop for the release path (C3)
        self._last_good_arm14 = None                 # last finite 14-joint read (observation anchor)

        # Liveness/staleness tracking (H2): the collect loop bumps _fresh_mono only when the EE
        # pose or a policy camera frame ACTUALLY changes, so a frozen feed is detectable even
        # though every tick is "recent". _firmware_has_error mirrors get_motion_status().
        self._fresh_mono = 0.0
        self._last_ee_sig = None
        self._last_cam_sig = {}
        self._firmware_has_error = False

        # Names a consumer is allowed to request (validation only — not subscribed).
        self.cameras = list(allowed_cameras)
        # "Pinned" cameras stay ON for the env's whole life (never idle-evicted),
        # e.g. data-collection cameras. Empty for the GUI -> nothing on at launch.
        self._pinned = set(cameras)
        self.output_dir = pathlib.Path(output_dir)
        self.record_hz = frequency
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
        # robot0_left_joint obs (policy input): rolling [j_{t-1}, j_t], each = 7 LEFT-arm joint
        # angles (rad) + 1 RAW gripper value (8-dim). Built alongside the EE pair in _collect_loop.
        self._last_two_joint_states = None
        self._obs_timestamp = 0.0

        # Latest both-arm EE poses, refreshed every collect tick (for GUI display).
        self._latest_left_pos = None       # (x, y, z)
        self._latest_left_quat = None      # (qx, qy, qz, qw)
        self._latest_right_pos = None
        self._latest_right_quat = None

        # Recording state — guarded by _rec_lock. _rec holds the in-progress
        # recording session (None when not recording). Ported from RobotDataCollector.
        self._rec_lock = threading.Lock()
        self._rec = None
        self._is_recording = False

        self._action_queue = []            # most recent predicted chunk, FIFO drained
        self._smoothed_pos = None

        self._stop_event = threading.Event()
        self._collect_thread = None
        self._exec_thread = None
        self._run_exec = True

    # ===================== lifecycle (RealEnv.start/stop/__enter__) =====================
    @property
    def inf_ready(self):
        """Ready for inference once the two policy cameras + EE state are populated."""
        with self._lock:
            frames_ready = all(
                self._frames.get(n) is not None for n in (AGENT_CAMERA, HAND_CAMERA))
            return frames_ready and self._last_two_ee_states is not None

    # Latest both-arm EE poses for the data-collection GUI display. None until the
    # collect loop has read at least one motion status.
    @property
    def latest_left_pos(self):
        return self._latest_left_pos

    @property
    def latest_left_quat(self):
        return self._latest_left_quat

    @property
    def latest_right_pos(self):
        return self._latest_right_pos

    @property
    def latest_right_quat(self):
        return self._latest_right_quat

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

    def start(self, run_collect=True, run_exec=True):
        """Start the collect thread (producer) and/or the exec thread (consumer).

        Data collection passes run_exec=False (it only needs the producer loop). The
        sim-only runner passes run_collect=False: obs comes from a recorded source, so the
        SDK camera/collect loop isn't needed.
        """
        self._run_exec = run_exec
        self._stop_event.clear()
        if run_collect:
            self._collect_thread = threading.Thread(target=self._collect_loop, daemon=True)
            self._collect_thread.start()
        if run_exec:
            self._exec_thread = threading.Thread(target=self._exec_loop, daemon=True)
            self._exec_thread.start()
        # The release thread (the ONLY robot-motion driver) runs only when real=True.
        if run_exec and self._real:
            self._release_thread = threading.Thread(target=self._release_loop, daemon=True)
            self._release_thread.start()
        print(f"[HumanoidEnv]: started (collect={'on' if run_collect else 'off'}, "
              f"exec={'on' if run_exec else 'off'}, real={'on' if self._real else 'off'}).")

    def stop(self):
        self._stop_event.set()

        # Grab any in-progress recording before the collect thread can touch it
        # for one more tick, so we can finalize it after the thread exits.
        rec = None
        with self._rec_lock:
            if self._is_recording:
                self._is_recording = False
                rec = self._rec
                self._rec = None

        for t in (self._collect_thread, self._exec_thread, self._release_thread):
            if t is not None and t.is_alive():
                t.join(timeout=5.0)
        self._collect_thread = self._exec_thread = self._release_thread = None

        # Finalize the grabbed recording now that the collect thread is gone.
        if rec is not None:
            self._finalize(rec)

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

            # One get_motion_status + one gripper read per tick; both the
            # inference EE state and the recording rows are derived from them.
            try:
                status = self.robot_controller.get_motion_status()
                grip = self.robot.gripper_states()[0]
                ee_state = self._left_ee_from(status, grip)
                self._update_latest_poses(status)
                arm14 = self._read_arm14()              # refresh last-good arm read (C4)
                # robot0_left_joint obs: 7 LEFT-arm joint angles (rad) + RAW gripper (matches the
                # training zarr layout). grip is array-like (cf. _left_ee_from's grip[0]); take the
                # scalar and .tolist() so the row is pure Python floats (JSON-safe, homogeneous).
                joint_state = (arm14[:7].tolist() + [float(grip[0])]) if arm14 is not None else None
            except Exception as e:
                print(f"[HumanoidEnv]  [collect] get_motion_status failed: {e}")
                status = grip = ee_state = joint_state = None

            frames = self._read_frames(desired)

            # Freshness (H2): did the EE pose or a policy camera frame actually CHANGE? Only then
            # bump _fresh_mono. A frozen feed (stale arm_joint_states / dropped camera) stops
            # advancing _fresh_mono even though `now` keeps moving, so get_obs can flag staleness.
            changed = False
            ee_sig = tuple(np.round(ee_state, 6)) if ee_state is not None else None
            if ee_sig is not None and ee_sig != self._last_ee_sig:
                changed = True
            cam_sigs = {n: self._frame_sig(frames[n][-1]) for n in (AGENT_CAMERA, HAND_CAMERA)
                        if n in frames}
            for n, sig in cam_sigs.items():
                if sig != self._last_cam_sig.get(n):
                    changed = True
            fw_error = bool(status and status.get('error', {}).get('has_error'))

            with self._lock:
                self._obs_timestamp = now
                self._firmware_has_error = fw_error
                if ee_sig is not None:
                    self._last_ee_sig = ee_sig
                self._last_cam_sig.update(cam_sigs)
                if changed or self._fresh_mono == 0.0:
                    self._fresh_mono = now_mono
                for name, pair in frames.items():
                    self._frames[name] = pair
                if ee_state is not None:
                    if self._last_two_ee_states:
                        self._last_two_ee_states = [self._last_two_ee_states[-1], ee_state]
                    else:
                        self._last_two_ee_states = [ee_state, ee_state]
                if joint_state is not None:
                    if self._last_two_joint_states:
                        self._last_two_joint_states = [self._last_two_joint_states[-1], joint_state]
                    else:
                        self._last_two_joint_states = [joint_state, joint_state]

            # Append a recording row while a session is active.
            with self._rec_lock:
                if self._is_recording and self._rec is not None and status is not None:
                    self._record_tick(now, status, grip, frames)

            next_tick += self.dt
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                self._stop_event.wait(sleep_for)
            else:
                next_tick = time.monotonic()

    @staticmethod
    def _extract_pose(frame):
        """One motion-status link frame -> (x, y, z, qx, qy, qz, qw).

        Same extraction as RobotDataCollector._get_hand_statuses._extract.
        """
        pos = frame['position']
        quat = frame['orientation']['quaternion']
        return (
            pos['x'], pos['y'], pos['z'],
            quat['x'], quat['y'], quat['z'], quat['w'],
        )

    def _left_ee_from(self, status, grip):
        """Left EE pose + gripper -> [pos(3), quat xyzw(4), grip(1)] for inference.

        Derived from an already-fetched get_motion_status() result and gripper
        reading so the collect loop does only one SDK read of each per tick.
        """
        pose = self._extract_pose(status['frames']['arm_left_link7'])
        return [
            *pose,
            1.0 if grip[0] > 0.5 else 0.0,
        ]

    @staticmethod
    def _frame_sig(frame):
        """Cheap change signature for a camera frame (H2): a coarse subsample, not the whole
        image, so per-tick freeze detection stays light. None frame -> None."""
        if frame is None:
            return None
        a = np.asarray(frame)
        return a[::64, ::64].tobytes() if a.ndim >= 2 else a.tobytes()

    def _firmware_unsafe(self):
        """Defense-in-depth: True if get_motion_status reports an error or active collisions, or
        can't be read. Polled on the release path so a firmware-detected fault triggers E-stop
        even for hazards the (self-collision-only) sim can't see (e.g. hitting the table)."""
        try:
            st = self.robot_controller.get_motion_status()
        except Exception as e:
            print(f"[HumanoidEnv] get_motion_status failed during release: {e}")
            return True
        if not st:
            return True
        if st.get('error', {}).get('has_error'):
            print(f"[HumanoidEnv] firmware error: {st['error'].get('message')}")
            return True
        if st.get('collisions'):
            print(f"[HumanoidEnv] firmware collision(s): {st['collisions']}")
            return True
        return False

    def _update_latest_poses(self, status):
        """Refresh the cached both-arm EE poses shown by the data-collection GUI."""
        frames = status['frames']
        left = self._extract_pose(frames['arm_left_link7'])
        right = self._extract_pose(frames['arm_right_link7'])
        self._latest_left_pos = left[:3]
        self._latest_left_quat = left[3:]
        self._latest_right_pos = right[:3]
        self._latest_right_quat = right[3:]

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
            age:        seconds since the EE pose / cameras last actually changed (H2)
            stale:      True if age > STALE_TIMEOUT or the firmware reports an error
            firmware_error: bool
        """
        # Keep the two policy cameras warm while inference polls get_obs().
        print("[HumanoidEnv] requesting hand and head camera")
        self.request(AGENT_CAMERA)
        self.request(HAND_CAMERA)
        with self._lock:
            if self._last_two_ee_states is None or self._last_two_joint_states is None:
                print("[HumanoidEnv] Waiting for EE / joint states")
                return None
            if any(self._frames.get(n) is None for n in (AGENT_CAMERA, HAND_CAMERA)):
                print("[HumanoidEnv] Wait for Camera")
                return None
            age = (time.monotonic() - self._fresh_mono) if self._fresh_mono else float('inf')
            stale = age > STALE_TIMEOUT or self._firmware_has_error
            return {
                'agent_imgs': copy.deepcopy(self._frames[AGENT_CAMERA]),
                'hand_imgs': copy.deepcopy(self._frames[HAND_CAMERA]),
                'state': copy.deepcopy(self._last_two_ee_states),
                # robot0_left_joint policy input: [j_{t-1}, j_t], each 7 left-arm joints + raw gripper.
                'joint_state': copy.deepcopy(self._last_two_joint_states),
                'timestamp': self._obs_timestamp,
                'age': age,
                'stale': bool(stale),
                'firmware_error': bool(self._firmware_has_error),
            }

    # ===================== caller -> env: hand off a prediction =====================
    def submit_actions(self, action_chunk):
        """Hand a chunk to the SIM-PREVIEW queue (auto-run / replay). Sim only — this path can
        NEVER reach the robot. The hardware path is validate_and_stage() + release_to_robot()."""
        with self._lock:
            self._action_queue = copy.deepcopy(list(action_chunk)) if action_chunk else []

    def queue_empty(self):
        """True when no actions are pending (used by runners to detect drain)."""
        with self._lock:
            return not self._action_queue

    def _make_solve_fn(self):
        """Build the per-action solve callback the sim validation uses: workspace check (H4) +
        orientation smoothing (H3) + our IK. Returns (q7|None, grip, reason)."""
        qprev_box = [None]    # local quat-smoothing state (don't disturb the preview's)

        def solve_fn(action, seed):
            pos, quat, grip = self._decode_ee_action(action)
            pos = np.asarray(pos, dtype=np.float64)
            if not self._pos_in_workspace(pos):                                  # H4
                return None, grip, "target EE pos outside workspace envelope"
            qprev_box[0] = _smooth_quat_step(qprev_box[0], quat, QUAT_ALPHA)     # H3
            q7 = self.solver.solve(IKQuery(target_pos=pos, target_quat=qprev_box[0],
                                           current_joints=seed))
            if not self.solver.last_reachable:
                return None, grip, f"IK unreachable (pos err {self.solver.last_pos_err*1000:.0f} mm)"
            return q7, grip, None

        return solve_fn

    def calibrate_collisions(self, action_chunk):
        """Learn this coarse URDF's inherent self-collision overlaps from a KNOWN-SAFE action
        chunk: walk it through the sim (same subdivide+settle as validation) and absorb every
        new left-side contact into the ignore baseline. Run over a representative corpus of safe
        recordings before trusting validation, or it will false-positive on normal poses."""
        if self.sim is None:
            return False, "no sim running"
        self.sim.validate(action_chunk, self._make_solve_fn(), self._last_q, MAX_JOINT_STEP,
                          learn=True)
        return True, "collision baseline updated"

    # ===================== shared sim-validation core (manual + auto) =====================
    def _validate_chunk(self, action_chunk, seed_q, fast=False):
        """Run a chunk through self.sim from seed_q (step + self-collision + joint readback) and
        return (traj, ok, reason) where traj is the sim-ACHIEVED [(q7, grip), ...]. The single
        validation primitive shared by the manual validate_and_stage and auto-inference (which
        validates on the SAME preview sim, then auto-splices instead of waiting for a manual
        release). A fresh solve_fn per call gives independent quat-smoothing state.
        fast=True (auto path): skip the per-substep real-time sleep — that sleep is just for the
        operator to watch the manual preview and is ~1s of pure latency per auto inference."""
        if self.sim is None:
            return [], False, "no sim running — press 启动仿真预览 first"
        if not action_chunk:
            return [], False, "empty action chunk"
        return self.sim.validate(action_chunk, self._make_solve_fn(), seed_q, MAX_JOINT_STEP,
                                 fast=fast)

    # ===================== manual: validate a chunk in the sim, then release =====================
    def validate_and_stage(self, action_chunk):
        """Run a predicted chunk through the SIM (step + self-collision + joint readback) and,
        on success, stage the sim-ACHIEVED joint trajectory for release. This is the ONLY
        producer of self._last_sim_traj, and it requires a live sim — so without a sim nothing
        is ever releasable. Returns (ok: bool, reason: str)."""
        if self.sim is None:
            return False, "no sim running — press 启动仿真预览 first"
        if not action_chunk:
            return False, "empty action chunk"

        # Re-anchor the IK seed AND the sim to the robot's LIVE pose before validating, so the
        # self-collision check replicates the arm's ACTUAL geometry rather than the last planned
        # config. Only when the release pipeline is idle (nothing staged/streaming) — while
        # staging ahead we keep planning continuously from the prior segment's end (see below).
        self._resync_to_robot_if_idle()

        traj, ok, reason = self._validate_chunk(action_chunk, self._last_q)
        with self._lock:
            if ok and traj:
                self._last_sim_traj = traj
                self._validation_id += 1
                # Advance the IK warm-start seed to the joints achieved at the end of this
                # validated segment, so a subsequent step-through call (单步/整条) plans
                # continuously from here instead of restarting from the original seed.
                self._last_q = np.asarray(traj[-1][0], dtype=np.float64).copy()
                # APPEND these substeps to the ready-to-release buffer (单步/整条 both accumulate
                # here). release_next_substep / release_remaining_substeps drain it to the robot.
                self._staged_release.extend(traj)
                print(f"[HumanoidEnv] sim-validated {len(traj)} points (id {self._validation_id}); "
                      f"{len(self._staged_release)} substep(s) staged for release.")
            else:
                self._last_sim_traj = []
                print(f"[HumanoidEnv] sim validation FAILED: {reason}")
        return ok, (reason or "")

    def _resync_to_robot_if_idle(self):
        """Re-anchor the IK seed (_last_q) and the sim to a FRESH live robot reading so the next
        validation replicates the arm's REAL pose (both arms + torso + grippers), the same sync
        SimEnv gets at preview launch. No-op when:
          * there's no sim, or
          * substeps are staged or queued (self._staged_release / self._robot_q non-empty) — then
            we're staging/executing AHEAD and must keep planning continuously from the prior
            segment's end (traj[-1][0]) instead of snapping back to a transient mid-motion pose, or
          * the live arm read fails (leave the seed intact).
        reset_full is owning-thread-only, so it's marshalled via submit_job like validate."""
        if self.sim is None:
            return
        with self._lock:
            if self._staged_release or self._robot_q:
                return
        arm14 = self._read_arm14()                       # both arms (rad); None if never readable
        if arm14 is None:
            return
        body_pitch = grip = None                          # best-effort; reset_full takes None fine
        try:
            body_pitch = float(self.robot.waist_joint_states()[0][0])
        except Exception:
            pass
        try:
            grip = self.robot.gripper_states()[0]
        except Exception:
            pass
        self.sim.submit_job(lambda: self.sim.reset_full(
            arm14=arm14, body_pitch=body_pitch, gripper_lr=grip))
        self.set_seed(arm14[:7])                          # IK seed -> real left-arm joints

    # ===================== real-robot release ("validate in sim, then release") =====================
    def release_to_robot(self):
        """Replay the LAST sim-validated trajectory on the REAL robot. The only path to
        hardware. One-shot (a validation can be released once, H1), refused while E-stopped,
        ramped in from the current measured LEFT-arm pose (C5). Only the left arm is ever
        commanded — the right arm is never addressed, so it holds. Returns points queued."""
        if not self._real:
            print("[HumanoidEnv] release ignored: env built with real=False.")
            return 0
        if self._estop.is_set():
            print("[HumanoidEnv] release refused: E-stop latched (press 复位 to reset).")
            return 0
        with self._lock:
            traj = list(self._last_sim_traj)
            vid = self._validation_id
            if not traj:
                print("[HumanoidEnv] release refused: nothing sim-validated (run 执行 first).")
                return 0
            if vid == self._released_id:
                print("[HumanoidEnv] release refused: already released; run 执行 again.")
                return 0
        arm14 = self._read_arm14()                       # current LEFT pose for the C5 ramp-in
        if arm14 is None:
            print("[HumanoidEnv] release refused: cannot read arm_joint_states (ramp-in start).")
            return 0
        ramp = self._ramp(arm14[:7], traj[0][0], MAX_JOINT_STEP, traj[0][1])   # C5 ramp-in
        full = self._subdivide_points(ramp + traj)       # <= MAX_JOINT_STEP for the live per-tick drain
        with self._lock:
            self._robot_q.clear()
            self._robot_q.extend(full)
            self._released_id = vid                      # H1: one-shot
            self._last_sim_traj = []
        print(f"[HumanoidEnv] released {len(traj)} validated pts (+{len(ramp)} ramp-in) to robot.")
        return len(full)

    # ===================== substep-by-substep release (accumulating buffer) =====================
    @property
    def staged_substeps(self):
        """How many validated substeps are staged and not yet released to the robot."""
        with self._lock:
            return len(self._staged_release)

    def staged_preview(self, n=10):
        """The next up-to-n staged substeps as (7-joint array, grip) pairs (copies), in release
        order. grip is the binary {0,1} gripper command staged with the substep. For the live
        substep monitor; cheap snapshot under the lock."""
        with self._lock:
            items = list(self._staged_release)[:n]
        return [(np.asarray(q, dtype=np.float64).copy(), float(grip)) for (q, grip) in items]

    def release_n_substeps(self, n):
        """Release up to N staged substeps to the robot: pops the next n from the ready-to-release
        buffer onto _robot_q (which _release_loop hands to the trajectory controller). Releases
        whatever is left when fewer than n are staged. Refused while E-stopped; _enqueue_for_release
        adds the C5 ramp-in. Returns the number of commands queued (>= n in the synced case; more if
        a ramp-in was needed)."""
        if not self._real:
            print("[HumanoidEnv] release ignored: env built with real=False.")
            return 0
        if self._estop.is_set():
            print("[HumanoidEnv] release refused: E-stop latched (press 重置急停 to reset).")
            return 0
        if n <= 0:
            return 0
        with self._lock:
            if not self._staged_release:
                print("[HumanoidEnv] release refused: no staged substeps (run 仿真验证 first).")
                return 0
            subs = [self._staged_release.popleft()
                    for _ in range(min(n, len(self._staged_release)))]
            print(f"[HumanoidEnv] releasing {len(subs)} substep(s) "
                  f"({len(self._staged_release)} still staged)")
        return self._enqueue_for_release(subs)

    def release_remaining_substeps(self):
        """Release ALL staged substeps to the robot; _release_loop hands them to the trajectory
        controller. Refused while E-stopped; same C5 ramp-in as release_n_substeps. Returns the
        number of commands queued."""
        if not self._real:
            print("[HumanoidEnv] release ignored: env built with real=False.")
            return 0
        if self._estop.is_set():
            print("[HumanoidEnv] release refused: E-stop latched (press 重置急停 to reset).")
            return 0
        with self._lock:
            if not self._staged_release:
                print("[HumanoidEnv] release refused: no staged substeps (run 仿真验证 first).")
                return 0
            subs = list(self._staged_release)
            self._staged_release.clear()
        return self._enqueue_for_release(subs)

    def _enqueue_for_release(self, subs):
        """Append validated substeps [(q7, grip)] onto _robot_q. When the release pipeline is
        idle (queue empty and nothing in flight) prepend a C5 ramp-in from the current measured
        LEFT-arm pose to the first substep (collapses to a single command when already in sync);
        while a batch is already streaming just append, continuous. Only the left arm is ever
        commanded — the right arm is never addressed. Returns commands queued."""
        if not subs:
            return 0
        arm14 = self._read_arm14()                       # current LEFT pose for the C5 ramp-in
        if arm14 is None:
            print("[HumanoidEnv] release refused: cannot read arm_joint_states (ramp-in start).")
            with self._lock:                             # don't lose the popped substeps
                self._staged_release.extendleft(reversed(subs))
            return 0
        with self._lock:
            if self._robot_q:                            # already streaming -> append, continuous
                full = list(subs)
            else:                                        # idle -> C5 ramp from the measured pose
                ramp = self._ramp(arm14[:7], subs[0][0], MAX_JOINT_STEP, subs[0][1])
                full = ramp + list(subs)[1:]             # ramp's last point IS subs[0]
            self._robot_q.extend(self._subdivide_points(full))
            staged_substep_remaining = len(self._staged_release)
        print(f"[HumanoidEnv] queued {len(full)} cmd(s) to robot ({staged_substep_remaining} substep(s) still staged).")
        return len(full)

    # ===================== auto-inference: validate + time-aligned splice (no manual accept) =====================
    def auto_ingest_chunk(self, action_chunk, obs_ts):
        """Auto-inference entry point: validate a predicted chunk on the preview sim (the SAME
        sim-before-robot self-collision check the manual path uses) and, on success, splice the
        sim-achieved substeps into the LIVE _robot_q time-aligned to `obs_ts` — no manual release.
        Refused while E-stopped / without a sim. On validation FAILURE the previous trajectory keeps
        draining (we just skip this inference). Returns (ok: bool, reason: str)."""
        if not self._real:
            return False, "env built with real=False"
        if self.sim is None:
            return False, "no sim running — launch the preview first"
        if self._estop.is_set():
            return False, "E-stop latched"
        # Seed validation from a FRESH real left-arm read so the self-collision check and IK plan
        # from the arm's ACTUAL pose (what the policy observed), not a stale planned config.
        arm14 = self._read_arm14()
        seed = arm14[:7] if arm14 is not None else self._last_q
        traj, ok, reason = self._validate_chunk(action_chunk, seed, fast=True)
        if not ok or not traj:
            return False, reason or "empty validated trajectory"
        with open("traj.jsonl", "a") as f:
            f.write(json.dumps({"obs_ts": float(obs_ts),
                                "traj": [[np.asarray(q7).tolist(), float(grip)]
                                         for q7, grip in traj]}) + "\n")
        self._auto_splice_nearest(traj, obs_ts)
        return True, ""

    def _auto_splice(self, traj, obs_ts):
        """Integrate a freshly validated trajectory into the live _robot_q so consecutive inferences
        join smoothly. `traj` = sim-achieved [(q7, grip), ...], one element per executed substep
        (drained at one substep / STEP_TIME), starting from the pose observed at `obs_ts`. The
        substep that corresponds to "now" is f = round((now - obs_ts)/STEP_TIME) — MEASURED elapsed
        time, so it auto-adapts to the inference gap.

        RUNWAY: we keep a runway of `keep_n ≈ f` committed substeps (one inference-latency's worth)
        rather than just 2, so a STALE chunk only rewrites the FUTURE tail, not motion the arm is
        already committed to. We then splice the new chunk at the TIME-ALIGNED index g = f + keep_n
        — the substep whose wall-clock matches the END of the runway — and _ramp (bounded <=
        MAX_JOINT_STEP) from the runway's last point to traj[g], then follow traj[g+1:]. Keeping only
        2 substeps + splicing at traj[f] was the back-and-forth bug: keep[-1] and traj[f] are the
        SAME instant but from different (old vs new) plans, so the bridge could run backward. With
        the shift, keep[-1] and traj[g] refer to the same FUTURE instant -> forward, short bridge,
        and the chunk is joined where it is still fresh (needs horizon_substeps > ~2f; otherwise g
        saturates at the end -> clean lurch, no oscillation). When idle/first inference we ramp from
        the real pose to traj[0] (the whole trajectory plays)."""
        print(f"world time: {time.time()} | obs time: {obs_ts} | time elapsed:{time.time() - obs_ts}")
        f = max(0, min(int(round((time.time() - obs_ts) / STEP_TIME)), len(traj) - 1))
        arm14 = self._read_arm14()                        # read outside the lock (DDS I/O)
        start = arm14[:7] if arm14 is not None else np.asarray(traj[0][0], dtype=np.float64)
        keep_n, g = 0, f                                  # for logging; set in the merge branch
        with self._lock:
            if not self._robot_q:                         # idle / first inference -> fresh ramp-in
                ramp = self._ramp(start, traj[0][0], MAX_JOINT_STEP, traj[0][1])
                new_q = ramp + list(traj[1:])             # ramp's last point IS traj[0]
            else:
                # Keep ~f substeps of committed runway (capped by what's queued), then bridge from
                # the runway's last point to the TIME-ALIGNED substep g = f + keep_n and replace the
                # tail. keep[-1] is robust when fewer than keep_n remain.
                keep_n = max(1, min(f, len(self._robot_q)))
                keep = [self._robot_q[i] for i in range(keep_n)]
                g = min(f + keep_n, len(traj) - 1)
                ramp = self._ramp(keep[-1][0], traj[g][0], MAX_JOINT_STEP, traj[g][1])  # bridge -> g
                new_q = keep + ramp + list(traj[g + 1:])   # ramp ends at traj[g]; replace the tail
            self._robot_q.clear()
            self._robot_q.extend(self._subdivide_points(new_q))
            qlen = len(self._robot_q)
            # Advance the IK warm-start seed to the end of this trajectory for continuity.
            self._last_q = np.asarray(traj[-1][0], dtype=np.float64).copy()
        print(f"[HumanoidEnv] auto-splice: f={f}/{len(traj)-1}, keep={keep_n}, g={g}, "
              f"+{len(ramp)} ramp -> queue {qlen}")

    def _auto_splice_nearest(self, traj, obs_ts=None,
                             match_win=16, blend_win=10, search_radius=30):
        """Alternative to the time-based _auto_splice: align the new chunk to the in-flight queue by
        WINDOWED SHAPE-MATCHING, then CROSSFADE old->new (instead of a hard ramp).

        Why windowed: matching a SINGLE pose is ambiguous — many substeps share a joint config, so
        argmin can latch onto the wrong one (the back-and-forth). Instead we slide a window of the
        committed queue's front against the new traj and pick the offset d minimizing windowed L1
        over the curve's SHAPE:  cost(d) = Σ_{i<W} |old_q[i] - new_q[d+i]|₁.
        Why bounded: when obs_ts is known we restrict d to [f-R, f+R] around the time estimate
        f=round((now-obs_ts)/STEP_TIME) — time gets the neighbourhood, shape locks on, so a
        featureless (slow/straight) stretch can't mis-match far away.
        Merge: crossfade old->new over blend_win substeps (αᵢ: 0->1), then follow traj[d+B:].
        CAVEAT: the blended segment interpolates two validated paths but is not itself collision-
        checked (same as the ramp it replaces); _subdivide_points still bounds each step to
        MAX_JOINT_STEP. Idle/first inference -> plain ramp-in from the real pose."""
        arm14 = self._read_arm14()                        # DDS I/O, outside the lock
        live = arm14[:7] if arm14 is not None else np.asarray(traj[0][0], dtype=np.float64)
        new_q = np.array([t[0] for t in traj], dtype=np.float64)        # (N, 7)
        N = new_q.shape[0]
        with self._lock:
            if not self._robot_q:                         # idle / first inference -> fresh ramp-in
                ramp = self._ramp(live, traj[0][0], MAX_JOINT_STEP, traj[0][1])
                self._robot_q.extend(self._subdivide_points(ramp + list(traj[1:])))
                qlen = len(self._robot_q)
                self._last_q = np.asarray(traj[-1][0], dtype=np.float64).copy()
                print(f"[HumanoidEnv] auto-splice(blend): idle ramp-in -> queue {qlen}")
                return
            # Snapshot the committed queue's joints (front = next to execute = "now").
            old_q = np.array([self._robot_q[i][0] for i in range(len(self._robot_q))],
                             dtype=np.float64)             # (M, 7)
            M = old_q.shape[0]
            W = max(1, min(match_win, M, N))
            d_max = N - W                                  # last valid window start (>=0 since W<=N)
            # Candidate offsets: bound near the time estimate f if obs_ts known, else the whole traj.
            # Clamp BOTH ends to [0, d_max] so new_q[d:d+W] is always a full window — when latency
            # >> horizon, f-search_radius can exceed d_max and an unclamped lo crashes the L1 below.
            if obs_ts is not None:
                f = int(round((time.time() - obs_ts) / STEP_TIME))
                lo = max(0, min(f - search_radius, d_max))
                hi = max(lo, min(d_max, f + search_radius))
            else:
                lo, hi = 0, d_max
            # Windowed L1 over the (small) candidate range -> best shape alignment.
            best_d, best_cost = lo, np.inf
            for d in range(lo, hi + 1):
                cost = np.abs(old_q[:W] - new_q[d:d + W]).sum()
                if cost < best_cost:
                    best_cost, best_d = cost, d
            # Crossfade old -> new over B substeps from the matched point, then follow the new tail.
            B = max(1, min(blend_win, M, N - best_d))
            blended = [((1.0 - (i + 1) / B) * old_q[i] + ((i + 1) / B) * new_q[best_d + i],
                        traj[best_d + i][1])              # new chunk's gripper
                       for i in range(B)]
            new_seq = blended + list(traj[best_d + B:])
            self._robot_q.clear()
            self._robot_q.extend(self._subdivide_points(new_seq))
            qlen = len(self._robot_q)
            self._last_q = np.asarray(traj[-1][0], dtype=np.float64).copy()
        print(f"[HumanoidEnv] auto-splice(blend): d={best_d}/{N-1}, W={W}, B={B}, "
              f"cost={best_cost:.3f} -> queue {qlen}")

    def lock_robot(self):
        """E-STOP (latched). Drop pending robot commands and ACTIVELY hold the arm at its
        current measured pose — the SDK has no brake, so re-commanding 'here' is the stop.
        Stays latched until reset_estop(). (The physical E-stop remains the primary safety.)"""
        self._estop.set()
        with self._lock:
            dropped = len(self._robot_q) + len(self._staged_release)
            self._robot_q.clear()
            self._staged_release.clear()             # drop un-released substeps too (re-validate after reset)
        print(f"[HumanoidEnv] E-STOP: latched; dropped {dropped} pending/staged cmds; holding pose.")
        if self._real:
            arm14 = self._read_arm14()
            if arm14 is None:
                print("[HumanoidEnv] E-stop WARNING: no joint read to hold — use physical E-stop.")
                return
            for _ in range(3):                           # re-assert a few times to be sure
                # Hold the LEFT arm at its current pose; don't touch the gripper. NOTE: the SDK
                # has no trajectory abort, so this can only PREEMPT (not cancel) an in-flight
                # trajectory — clearing _robot_q above stops further batches; the physical E-stop
                # remains primary for an already-dispatched batch.
                self.run_trajectory_control([(arm14[:7], None)],  # None grip -> gripper untouched
                                            ignore_estop=True)    # hold must send while estopped

    def reset_estop(self):
        """Clear the latched E-stop (operator confirms the arm is safe). Release stays refused
        until a fresh 执行→释放 (the previous trajectory was consumed)."""
        self._estop.clear()
        print("[HumanoidEnv] E-stop reset; release re-enabled (run 执行 then 释放).")

    @property
    def estopped(self):
        return self._estop.is_set()

    @property
    def robot_pending(self):
        with self._lock:
            return len(self._robot_q)

    def set_seed(self, q7):
        """Set the IK warm-start seed (e.g. to the robot's current left-arm joints)."""
        self._last_q = np.asarray(q7, dtype=np.float64).copy()

    @staticmethod
    def _ramp(q_from, q_to, cap, grip):
        """Joint-space ramp from q_from to q_to with every step <= cap (rad). Includes q_to,
        excludes q_from. Used so the first released command starts at the current pose (C5)."""
        q_from = np.asarray(q_from, dtype=np.float64)
        q_to = np.asarray(q_to, dtype=np.float64)
        n = int(np.ceil(np.max(np.abs(q_to - q_from)) / cap))
        n = max(n, 1)
        return [(q_from + (q_to - q_from) * (i / n), float(grip)) for i in range(1, n+1)]
        # this gives the points that provides a smooth transition BETWEEN q_from and q_to (incl. q_to)

    def _pos_in_workspace(self, pos):
        """True if the target EE position is inside the configured workspace AABB (H4)."""
        (xl, xh), (yl, yh), (zl, zh) = WORKSPACE_AABB
        x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
        return xl <= x <= xh and yl <= y <= yh and zl <= z <= zh

    def _read_arm14(self):
        """Return a finite 14-vector of arm joints, caching the last good read. Falls back to
        the last good value on a failed/garbage read; None only if never read once (C4)."""
        try:
            cur, _ = self.robot.arm_joint_states()
            arr = np.asarray(cur, dtype=np.float64)
            if arr.shape == (14,) and np.all(np.isfinite(arr)):
                self._last_good_arm14 = arr.copy()
                return arr.copy()
            print(f"[HumanoidEnv] arm_joint_states bad read (shape {arr.shape}); using last good.")
        except Exception as e:
            print(f"[HumanoidEnv] arm_joint_states read failed: {e}; using last good.")
        return self._last_good_arm14.copy() if self._last_good_arm14 is not None else None

    # ===================== consumer: sim-preview execution loop =====================
    def _exec_loop(self):
        """Drain the sim-preview queue (auto-run / replay) -> IK -> SIM only. NEVER touches the
        robot (that is _release_loop). Applies the same workspace + smoothing guards as
        validation so the preview matches what would be validated."""
        while not self._stop_event.is_set():
            with self._lock:
                action = self._action_queue.pop(0) if self._action_queue else None
            if action is None:
                self._stop_event.wait(self.dt)
                continue

            pos, quat, grip = self._decode_ee_action(action)
            pos = np.asarray(pos, dtype=np.float64)
            if not self._pos_in_workspace(pos):                        # H4
                print("[HumanoidEnv] preview: target outside workspace; skipping")
                self._stop_event.wait(self.dt)
                continue

            #check if there exists a smoothed path, if no smooth it
            if self._smoothed_pos is None:
                self._smoothed_pos = pos
            else:
                self._smoothed_pos = self.pos_alpha * pos + (1.0 - self.pos_alpha) * self._smoothed_pos
            self._quat_prev = _smooth_quat_step(self._quat_prev, quat, QUAT_ALPHA)   # H3

            q7 = self._solve_ik(self._smoothed_pos, self._quat_prev)
            if q7 is None:                     # unreachable target -> skip (seed not advanced)
                self._stop_event.wait(self.dt)
                continue
            if self.sim is not None:
                self.sim.command(q7, grip)
            self._stop_event.wait(self.dt)

    def _release_loop(self):
        """The ONLY driver of real-robot motion. Drains self._robot_q ONE substep per STEP_TIME,
        LIVE (popleft under the lock each tick), so an auto-inference splice into the queue takes
        effect on the very next tick. Substeps are pre-subdivided to <= MAX_JOINT_STEP at
        enqueue/splice time, so per-tick streaming stays velocity-bounded (C5). Each substep is its
        own single-waypoint trajectory, so an E-stop halts the arm within ~STEP_TIME (C3)."""
        while not self._stop_event.is_set():
            if self._estop.is_set():
                self._stop_event.wait(STEP_TIME)
                continue
            with self._lock:
                sub = self._robot_q.popleft() if self._robot_q else None
            if sub is None:
                self._stop_event.wait(STEP_TIME)
                continue
            if self._firmware_unsafe():                        # defense-in-depth: firmware fault
                self.lock_robot()
                continue
            # run_trajectory_control([sub]) sends the one waypoint AND paces STEP_TIME internally,
            # so there is no extra wait here (that would double the period).
            if not self.run_trajectory_control([sub]):
                self.lock_robot()                              # couldn't dispatch -> estop

    def _decode_ee_action(self, action):
        """action row [eef_pos(3), 6D_rot(6), gripper(1)] -> (pos(3), quat(4), grip)."""
        return action[:3], rot6d_to_quat(action[3:9]), action[9]

    def _solve_ik(self, pos, quat):
        """Target EE pose -> 7 left-arm joint angles via our URDF IK (warm-started from the
        last solution). Returns None when the target is unreachable, leaving the seed intact
        so the next target plans from the last good config."""
        q7 = self.solver.solve(IKQuery(
            target_pos=np.asarray(pos, dtype=np.float64),
            target_quat=np.asarray(quat, dtype=np.float64),
            current_joints=self._last_q))
        if not self.solver.last_reachable:
            print(f"[HumanoidEnv] IK unreachable (pos err "
                  f"{self.solver.last_pos_err * 1000:.1f} mm); skipping action")
            return None
        self._last_q = q7
        return q7

    def run_trajectory_control(self, points, ignore_estop=False):
        """Stream a batch of sim-validated LEFT-arm waypoints to trajectory_tracking_control ONE at
        a time (ABS_JOINT, SDK doc 8.2.4), pacing at STEP_TIME and bailing on E-stop between points
        so the arm can be halted mid-batch (the SDK has no trajectory abort). robot_states is read
        once up front as the observation anchor (a read, never a command).

        points: [(q7, grip), ...] sim-achieved left-arm joints (rad) + binary {0,1} gripper, or
        grip=None to leave the gripper untouched (the E-stop hold). BLOCKS for ~len(points) *
        STEP_TIME while streaming. Each waypoint addresses only the LEFT arm and (when grip is not
        None) the 'gripper' group — the right arm and right gripper are never addressed, so they
        hold. Returns True on normal completion or an E-stop/shutdown halt; False on a dispatch error."""
        if not points:
            return True
        # points are already <= MAX_JOINT_STEP apart (C5): subdivision now happens UPSTREAM at
        # enqueue/splice time, so the live _robot_q can be drained one substep per tick (and spliced
        # into mid-stream by auto-inference) without an over-cap velocity.
        # robot_states: the observation that anchors the trajectory (8.2.4). Reads only, optional.
        robot_states = {}


        try:
            robot_states["arm"] = list(self.robot.arm_joint_states()[0])
        except Exception:
            pass
        try:
            robot_states["waist"] = list(self.robot.waist_joint_states()[0])
        except Exception:
            pass
        try:
            robot_states["head"] = list(self.robot.head_joint_states()[0])
        except Exception:
            pass
        # Stream the batch ONE waypoint at a time (not one big trajectory). The SDK has no
        # trajectory abort, so this is what makes an E-stop effective: latched mid-batch, we simply
        # stop sending the next waypoint and the arm halts within ~STEP_TIME at the last point.
        # Each point is its own single-waypoint ABS_JOINT trajectory (8.2.4 schema, LEFT arm only ->
        # right arm never moves) with reference_time = STEP_TIME; we pace on self._stop_event so a
        # shutdown breaks promptly too.
        for q7, grip in points:
            # Stop streaming a RELEASE batch the instant an E-stop latches (or on shutdown). The
            # E-stop HOLD itself passes ignore_estop=True so it can re-assert the pose while latched.
            if not ignore_estop and (self._estop.is_set() or self._stop_event.is_set()):
                return True                                  # halt: send no further waypoints
            infer_timestamp = int(time.time() * 1e9)     
            left7 = np.clip(np.asarray(q7, dtype=np.float64),
                            self._jlower, self._jupper)      # joint-limit clamp (no IK solver needed)
            # ONE waypoint driving BOTH the left arm and (optionally) the left gripper, via the
            # 'left_arm' and 'gripper' GROUP_IDS (8.2.1). No move_gripper call, and neither the
            # right arm nor the right gripper is ever addressed, so both hold.
            robot_action = [{"left_arm": {"action_data": left7.tolist(),
                                         "control_type": "ABS_JOINT"}}]
            if grip is not None:                         # None -> leave the gripper alone (E-stop hold)
                # grip is already the binary {0,1} normalized at inference -> open/close command.
                grip_cmd = 1 if grip >= 0.5 else 0
                self.robot.move_gripper([grip_cmd, 0])
            try:
                self.robot_controller.trajectory_tracking_control(
                    infer_timestamp, robot_states, robot_action, "base_link", STEP_TIME
                    )
            except Exception as e:
                print(f"[HumanoidEnv] trajectory_tracking_control failed: {e}")
                return False
            self._stop_event.wait(STEP_TIME)                 # pace; interruptible by stop_event
        return True

    @staticmethod
    def _subdivide_points(points):
        """Insert linearly-interpolated LEFT-arm waypoints so every consecutive gap <= MAX_JOINT_STEP
        (C5 velocity bound under trajectory control). grip carries from each segment's target point.
        A single point (or already-fine spacing) passes through unchanged."""
        pts = list(points)
        if len(pts) < 2:
            return pts
        out = [pts[0]]
        prev = np.asarray(pts[0][0], dtype=np.float64)
        for q, grip in pts[1:]:
            q = np.asarray(q, dtype=np.float64)
            d = q - prev
            n = max(int(np.ceil(np.max(np.abs(d)) / MAX_JOINT_STEP)), 1) if MAX_JOINT_STEP > 0 else 1
            for i in range(1, n + 1):
                out.append((prev + d * (i / n), grip))
            prev = q
        return out

    # ===================== data-collection recording (ported from RobotDataCollector) =====================
    def start_recording(self, episode_name=None):
        """Begin a recording session that writes mp4 per RECORD_CAMERA + an npz.

        Pins the record cameras so a tick never misses one to idle-eviction.
        """
        with self._rec_lock:
            if self._is_recording:
                print("[HumanoidEnv] Already recording.")
                return

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
                'cam_ready':     {name: False for name in RECORD_CAMERAS},
                'start_time':    datetime.now().isoformat(),
            }
            self._is_recording = True

        self._pinned |= set(RECORD_CAMERAS)
        print(f"[HumanoidEnv] Recording started -> {episode_dir}")

    def stop_recording(self):
        """End the active session and flush video/npz/metadata to disk."""
        with self._rec_lock:
            if not self._is_recording:
                return
            print("[HumanoidEnv] Stopping recording...")
            self._is_recording = False
            rec = self._rec
            self._rec = None

        self._pinned -= set(RECORD_CAMERAS)
        # Finalize outside the lock — the collect loop sees _rec is None and won't
        # touch this session, so writing files can't block streaming.
        self._finalize(rec)
        print("[HumanoidEnv] Recording stopped.")

    def _record_tick(self, t, status, grip, frames):
        """Append one row to the active session. Caller holds _rec_lock.

        `frames` is the collect loop's per-camera rolling [prev, cur] pairs; the
        current frame is frames[name][-1] (absent for a camera that dropped this tick).
        """
        rec = self._rec

        # Wait until every record camera has produced at least one frame before
        # writing anything, so row counts stay in sync.
        for name in RECORD_CAMERAS:
            if name in frames:
                rec['cam_ready'][name] = True
        if not all(rec['cam_ready'].values()):
            missing = [n for n, r in rec['cam_ready'].items() if not r]
            print(f"[HumanoidEnv]   waiting for cameras: {missing}")
            return

        # --- Record robot state ---
        rec['left'].append(self._extract_pose(status['frames']['arm_left_link7']))
        rec['right'].append(self._extract_pose(status['frames']['arm_right_link7']))
        # arm_joints sourced from arm_joint_states() (status quo). NOTE: this freezes
        # under concurrent VR teleop — arm_joints_ts records its sample timestamp so a
        # freeze is detectable (a constant ts across the episode == stale joints).
        arm_vals, arm_ts = self.robot.arm_joint_states()
        rec['arm_joints'].append(arm_vals)
        rec['arm_joints_ts'].append(arm_ts)
        rec['gripper'].append(grip)
        rec['timestamps'].append(t)

        # --- Write camera frames (skip cameras that dropped this tick) ---
        for name in RECORD_CAMERAS:
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
                print(f"[HumanoidEnv]   opened cameras/{name}.mp4  ({w}x{h})")

            rec['writers'][name].write(
                cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if rec['needs_bgr'][name] else frame
            )

    def _finalize(self, rec):
        """Flush video files and write npz/metadata for a finished session."""
        for w in rec['writers'].values():
            w.release()

        timestamps = rec['timestamps']
        n = len(timestamps)
        if n == 0:
            print(f"[HumanoidEnv]   no frames recorded -> {rec['episode_dir']}")
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
                'camera_names': RECORD_CAMERAS,
            }, f, indent=2)

        print(f"[HumanoidEnv]   saved {n} frames -> {rec['episode_dir']}")
