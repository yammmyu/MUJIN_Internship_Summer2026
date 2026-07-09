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

import json
import pathlib
import threading
import time
from collections import deque

import numpy as np

from real_world.ik import build_solver, rot6d_to_quat as _ik_rot6d_to_quat
# Episode recording is owned by a dedicated Recorder (its own lock + disk I/O); the env just
# drives it from the collect loop and lifecycle. RECORD_CAMERAS is re-exported here for
# back-compat (data_collection_gui imports it from this module).
from real_world.recording import Recorder, RECORD_CAMERAS
# Dynamic camera subscriptions live in a dedicated CameraHub (its own lock; one SDK camera
# object per camera, opened/closed on demand). KNOWN_CAMERAS / CAMERA_IDLE_TIMEOUT are the
# hub's defaults, imported here for the env constructor's signature.
from real_world.camera import CameraHub, KNOWN_CAMERAS, CAMERA_IDLE_TIMEOUT
# The sim-only preview loop (#7) is its own collaborator. Chunk IK-solve + sim-validation now live in
# the post-inference pipeline (#8, real_world.postprocess) alongside the merge; the env delegates
# _solve_chunk_ik / _validate_chunk to self.pipeline. Workspace envelopes + QUAT_ALPHA live there too.
from real_world.sim_preview import SimPreview
# Post-inference action pipeline (#8): raw policy chunk -> robot-ready substeps (gripper binarize +
# temporal-ensemble merge; later stages absorb IK, sim validation, and the queue splice). The env
# owns one PostProcessor (self.pipeline). GRIPPER_CLOSE_THRESH is re-exported here for back-compat
# (scripts import it from this module).
from real_world.postprocess import PostProcessor, GRIPPER_CLOSE_THRESH, APPEND_AHEAD_ROWS
# Producer-side observation buffers + freshness + get_obs/inf_ready (#3). extract_pose is the
# shared status-frame parser (obs right-EE + recorder both use it).
from real_world.observer import ObsCollector, extract_pose
# Timing/rate constants live in ONE place (real_world/timing.py) so RECORD_HZ, the substep rate
# (CONTROL_HZ/STEP_TIME) and the velocity cap (MAX_JOINT_VEL/MAX_JOINT_STEP) can't drift apart.
# Re-exported below for back-compat — scripts and tests import these names from humanoid_env.
from real_world.timing import (
    RECORD_HZ, ROW_DT, CONTROL_HZ, STEP_TIME, SUBSTEPS_PER_ROW, MAX_JOINT_VEL, MAX_JOINT_STEP,
    RAMP_JOINT_STEP, SPEED_SCALE, TRACE_DIR,
)

# a2d_sdk only exists on the robot machine. A sim-only machine (pybullet but no SDK) still
# needs to import this module for the IK/exec path, so guard the import. The sim runner never
# constructs these (it injects a _NoRobot stand-in); only the GUI / live robot path does.
try:
    from a2d_sdk.robot import RobotDds as Robot, CosineCamera as Camera, RobotController, Slam
except ImportError:
    Robot = Camera = RobotController = Slam = None

# ----------------------------------------------------------------------------- #
#  Safety limits for the real-robot release path (see the safety-review fixes).  #
# ----------------------------------------------------------------------------- #
# MAX_JOINT_STEP (= MAX_JOINT_VEL / CONTROL_HZ, from timing.py) is the max per-substep change of
# any single arm joint on the hardware path (rad) — a pure SAFETY velocity ceiling. Validation
# rejects a policy row whose joint velocity exceeds MAX_JOINT_VEL; the release loop and the
# ramp/bridge subdivision clamp to MAX_JOINT_STEP so a step-change target becomes a bounded ramp
# instead of a snap (C5). Motion smoothness is set by CONTROL_HZ, NOT by this cap.
# QUAT_ALPHA (orientation smoothing, H3) and the WORKSPACE_AABB / WORKSPACE_AABB_RIGHT envelopes
# (H4) now live in real_world.postprocess, alongside the solve_chunk_ik code that applies them.
# A live observation older than this (no real change in EE pose / camera) is "stale": the
# auto-inference loop aborts rather than command the arm from frozen sensor data. (H2)
STALE_TIMEOUT = 0.5             # seconds

# Streaming append depth (append_actions): keep at most this many policy ROWS queued ahead of the
# master clock. Each inference tops the tail up to here with NEW master ids only (never re-queues an
# id), so the arm executes one row at a time in strict order. Must be large enough that n rows of
# execution (n * SUBSTEPS_PER_ROW * STEP_TIME seconds) outlasts one inference round-trip, or the
# queue drains and the arm stalls between appends. Larger -> safer against latency, but the queued
# rows are older predictions (less reactive). 4 rows is the default starting point — tune to latency.
# TUNE [~2..12 rows]: latency-robustness knob (not directly smoothness). Constraint:
# n * SUBSTEPS_PER_ROW * STEP_TIME  must exceed one inference round-trip, else the queue starves.
# APPEND_AHEAD_ROWS is defined in real_world.postprocess (the pipeline owns append_actions) and
# re-exported via the import above for back-compat.

# STEP_TIME = 1/CONTROL_HZ (imported from timing.py): the release loop streams one sim-validated
# substep per STEP_TIME, so the arm advances on the same uniform time grid the auto-splice f index
# (f = round(elapsed / STEP_TIME)) assumes. Each policy row spans SUBSTEPS_PER_ROW such substeps.

# Gripper is BINARY open/close. The policy emits a noisy raw [0,~85] gripper signal (transient
# spikes exist), so the pipeline binarizes it to {0,1} (PostProcessor.binarize_grippers): only a
# (near-)fully-closed reading counts as CLOSED. Everything downstream (sim preview, validation,
# staging, release) then carries {0,1}; the LEFT gripper is then driven via the 'gripper' group of
# trajectory_tracking_control (never move_gripper), so the right gripper is never addressed.
# GRIPPER_CLOSE_THRESH itself is defined in real_world.postprocess and re-exported via the import above.
# RECORD_CAMERAS (the cameras captured to disk during data collection) now lives in
# real_world.recording and is imported/re-exported above.

# KNOWN_CAMERAS (allowed names), CAMERA_IDLE_TIMEOUT, and the fallback intrinsics now live in
# real_world.camera (CameraHub). KNOWN_CAMERAS / CAMERA_IDLE_TIMEOUT are imported above for the
# constructor defaults. The camera ROLES below stay here — they are policy-obs mapping, not hub state.

# Camera roles for the dual_arm_ee_image policy obs.
AGENT_CAMERA = "head"               # -> agentview_image
HAND_CAMERA_LEFT = "hand_left"      # -> robotl_eye_in_hand_image
HAND_CAMERA_RIGHT = "hand_right"    # -> robotr_eye_in_hand_image
HAND_CAMERA = HAND_CAMERA_LEFT      # back-compat alias (left wrist)
INFERENCE_CAMERAS = [AGENT_CAMERA, HAND_CAMERA_LEFT, HAND_CAMERA_RIGHT]


def rot6d_to_quat(rot6d):
    """6D rotation (first two rotation-matrix columns) -> quaternion [x, y, z, w].

    Delegates to the pinocchio-based converter in real_world.ik (Gram-Schmidt to a proper
    rotation matrix, then an exact matrix->quaternion). The previous hand-written
    matrix->quaternion here was numerically wrong for many rotations (round-trip rotation error
    up to ~2.4 vs ~1e-8 for the correct one), which corrupted decoded EE orientations and made
    ~20% of recorded targets spuriously IK-unreachable.
    """
    return np.asarray(_ik_rot6d_to_quat(rot6d), dtype=np.float64)


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
        # we only tear down what we created. Cameras are NOT injected — the CameraHub
        # (self.camhub, built below) is the sole owner of camera subscriptions and manages
        # one SDK camera object per camera dynamically, so nothing streams until a
        # consumer request()s it.
        self._owns_robot = robot is None
        self.robot = robot if robot is not None else Robot()
        self.robot_controller = robot_controller if robot_controller is not None else RobotController()
        # RobotDds.{arm,head,waist}_joint_states() only stream live data if a Slam()
        # instance exists in THIS process — Slam() brings up the robot-state pipeline
        # those topics ride on. Without it they return a frozen initial sample (while
        # gripper + EE poses, on a different channel, stay live: the signature of the
        # frozen-joints bug). Construct it (side-effect only) when we own the robot.
        self._slam = Slam() if (self._owns_robot and Slam is not None) else None
        if self._owns_robot:
            time.sleep(1.0)  # let freshly-created DDS resources come up

        # --- IK-based execution: sim backend + a sim-BEFORE-robot release pipeline ---
        # Non-negotiable invariant: an action can reach the real robot ONLY after the sim has
        # stepped through it, self-collision-checked it, and read back the achieved joints.
        # Enforced structurally, not by discipline:
        #   * SimEnv.validate (on the sim thread) is the ONLY producer of self._last_sim_traj and
        #     self._staged_release; it cannot run without a live sim, so "no sim" => nothing is
        #     ever releasable.self.robot
        #   * Only the release entry points enqueue onto self._robot_q, and only by copying
        #     freshly sim-validated substeps (E-stop-guarded, C5 ramp-in): release_to_robot()
        #     (one-shot, whole trajectory) and release_n_substeps/ release_remaining_substeps
        #     (drain the accumulating self._staged_release buffer).
        #   * _release_loop is the ONLY caller of run_trajectory_control, draining _robot_q in
        #     batches; only the LEFT arm is ever commanded, so the right arm is never moved.
        #   * _exec_loop drives the sim only (auto/replay preview); it never touches the robot.
        self.sim = sim                               # SimEnv (command/validate), or None
        self.solver = solver if solver is not None else build_solver()          # LEFT arm IK
        # RIGHT-arm IK solver (dual-arm execution): its own URDF joints + FK calibration
        # (fk_calibration_right.json). Both arms are solved per row and streamed in lockstep.
        self.solver_r = build_solver(side="right")
        # Shared release-pipeline lock + latched E-stop, created BEFORE the pipeline so the SAME lock
        # guards the robot queue whether it is touched by the pipeline (producer: append/auto-ingest)
        # or the env (consumer: release loop; manual release; lock_robot). threading.Lock is NOT
        # reentrant, so env code holding self._lock must never call a pipeline method that re-locks.
        self._lock = threading.Lock()
        self._estop = threading.Event()              # latched stop for the release path (C3)
        # Post-inference action pipeline (#8): raw policy chunk -> robot-ready substeps, ALL in one
        # place (real_world.postprocess). The inference controller hands each server chunk to
        # self.pipeline.merge (gripper binarize + temporal-ensemble smoothing); the release path calls
        # solve_chunk_ik / validate_chunk (dual-arm IK + sim validation); and the streaming splice
        # (append_actions / auto_ingest_chunk) + the robot QUEUE live here too. The env injects the
        # shared lock + callbacks (live pose read, E-stop predicate, current sim handle) and owns the
        # release-loop CONSUMER that drains pipeline.pop_next_substep. Standalone-constructible.
        self.pipeline = PostProcessor(
            self.solver, self.solver_r,
            lock=self._lock, read_arm14=self._read_arm14,
            estopped=self._estop.is_set, get_sim=lambda: self.sim, real=real)
        # All release-loop / inference traces land in one folder (see timing.TRACE_DIR). Create it
        # once here so the per-tick recorders in _release_loop can just open files inside it.
        TRACE_DIR.mkdir(parents=True, exist_ok=True)
        # Left-arm joint limits (rad), captured once so the execution path can clamp commands
        # without reaching into the IK solver (it only needs the bounds, not the solver).
        self._jlower = np.asarray(self.solver.m.lower, dtype=np.float64).copy()
        self._jupper = np.asarray(self.solver.m.upper, dtype=np.float64).copy()
        # 14-joint limits [left7, right7] for clamping dual-arm release commands.
        self._jlower14 = np.concatenate([self._jlower, np.asarray(self.solver_r.m.lower, dtype=np.float64)])
        self._jupper14 = np.concatenate([self._jupper, np.asarray(self.solver_r.m.upper, dtype=np.float64)])
        self._real = real
        # Sim-only action preview (#7). Owns the action queue + smoothing state + the LEFT IK
        # warm-start seed (its last_q / set_seed); the env's exec loop drives it via preview.tick.
        self.preview = SimPreview(self.solver, seed_q=seed_q, pos_alpha=pos_alpha)
        # DUAL IK warm-start seed (14,) [left7, right7], used by the validate/append/release path.
        # Re-anchored to the live robot on each idle validate (_resync) / auto ingest. Its LEFT half
        # starts from the preview's seed so both paths agree on the initial left-arm pose.
        self._last_q14 = np.concatenate([self.preview.last_q, self.solver_r.m.clip(np.zeros(7))])
        # Last 14-joint config actually COMMANDED to the robot (dispatch-time C5 velocity guard in
        # run_trajectory_control). None until the first command / after an E-stop; re-seeded from the
        # live pose so the guard bounds the very first waypoint too.
        self._last_cmd_q14 = None
        # Release pipeline state. The robot QUEUE (self._robot_q), the master clock
        # (self._current_row_id) and the streaming cursor (self._queued_through) now live in
        # self.pipeline (guarded by the shared self._lock); the env reaches them via delegating
        # properties (below). The MANUAL staging buffer + one-shot bookkeeping stay here:
        self._last_sim_traj = []                     # [(q7_achieved, grip)] last validated traj
        self._validation_id = 0                      # bumped on each successful validation
        self._released_id = -1                       # validation_id already released (one-shot, H1)
        # Substep-by-substep release buffer: each successful validate_and_stage APPENDS its
        # sim-achieved substeps here ("ready to release"). release_n_substeps / release_remaining_substeps
        # move substeps from here onto the pipeline queue, which _release_loop hands to the controller.
        self._staged_release = deque()               # [(q7_achieved, grip)] validated, awaiting release
        self._release_thread = None
        self._last_good_arm14 = None                 # last finite 14-joint read (observation anchor)

        # Gripper CLOSE LATCH (anti-regrab): once a channel has been COMMANDED closed continuously for
        # _grip_latch_sec, fix it at closed so a policy that keeps toggling can't re-open/re-grab. Per
        # channel [left, right]; "closed" = the binary dispatch command (== the upstream GRIPPER_CLOSE_
        # THRESH binarize). Reset by reset_grip_latch (E-stop / reset / auto start). None -> disabled.
        self._grip_latch_sec = 5.0
        self._grip_closed_since = [None, None]       # monotonic time each channel's closed streak began
        self._grip_latched = [False, False]          # channel currently fixed at closed

        # Camera subscriptions are owned by a CameraHub (its own lock; one SDK camera object per
        # camera, opened/closed on demand). `cameras=` (constructor) are the PINNED cameras — kept
        # ON for the env's life, e.g. data collection; `allowed_cameras` is the requestable set.
        self.camhub = CameraHub(Camera, allowed=allowed_cameras, pinned=cameras,
                                idle_timeout=idle_timeout)
        self.output_dir = pathlib.Path(output_dir)
        self.record_hz = frequency
        self.dt = 1.0 / frequency
        # pos_alpha (position low-pass EMA for the sim preview) is owned by self.preview. It applies
        # ONLY to the preview and does NOT touch the real robot path; robot-path position smoothing
        # comes from the temporal ensemble (TE_* in inference_controller).
        self.command_lifetime = command_lifetime

        # ---- shared state (self._lock is created above, before the pipeline, so both share it) ----
        self._log_ts = {}                                      # key -> last monotonic print time (throttling)
        # Producer-side observation (#3): rolling dual-arm EE-pose / gripper buffers + freshness +
        # get_obs/inf_ready live in the ObsCollector. The collect thread feeds it one ingest() per
        # tick; it FKs BOTH EE rows from the live joints (left + right solvers) so neither freezes on
        # the firmware's parked EE frame while its arm executes an ABS_JOINT trajectory.
        self.obs = ObsCollector(
            self.solver, self.solver_r, INFERENCE_CAMERAS, AGENT_CAMERA, HAND_CAMERA_LEFT,
            HAND_CAMERA_RIGHT, STALE_TIMEOUT)
        # Master row clock (self.pipeline._current_row_id) + streaming cursor (._queued_through) + the
        # robot queue live in the pipeline; _release_loop advances the clock as substeps dispatch and
        # _collect_loop snapshots it (via the self._current_row_id property) into the obs so each
        # prediction is anchored to the row the arm was actually on. Alignment is keyed on this, not time.

        # LIVE-TUNABLE execution knobs. speed_scale is env-owned (set_speed_scale recomputes the
        # pipeline's substeps_per_row (= K) + ramp_joint_step from it); substeps_per_row / ramp_joint_step
        # / append_ahead_rows are owned by the pipeline and reached via the delegating properties below.
        self.speed_scale = SPEED_SCALE

        # Episode recording is delegated to a Recorder (owns its own lock + session state +
        # disk I/O). It reads arm joints via the robot and reuses this env's status-frame pose
        # parser, so recorded rows match the obs source exactly. The env only drives it (tick
        # from the collect loop; start/stop/finalize from lifecycle) and pins the record cameras.
        self.recorder = Recorder(
            output_dir=self.output_dir,
            record_hz=self.record_hz,
            read_arm_joints=self.robot.arm_joint_states,
            extract_pose=extract_pose,
        )

        self._stop_event = threading.Event()
        self._collect_thread = None
        self._exec_thread = None
        self._run_exec = True

    # ===================== lifecycle (RealEnv.start/stop/__enter__) =====================
    @property
    def cameras(self):
        """Names a consumer is allowed to request (back-compat: the GUI reads env.cameras)."""
        return self.camhub.allowed

    @property
    def inf_ready(self):
        """Ready for inference once the three policy cameras + both EE states are populated."""
        return self.obs.inf_ready(self.camhub)

    # ===================== consumer-facing camera switch =====================
    def _throttled(self, key, interval=1.0):
        """True at most once per `interval` seconds for `key` (rate-limits hot-loop prints)."""
        now = time.monotonic()
        if now - self._log_ts.get(key, 0.0) >= interval:
            self._log_ts[key] = now
            return True
        return False

    # Camera switch/query is delegated to the CameraHub (kept as methods for the GUI's API).
    def request(self, name):
        """Mark a camera as wanted this cycle and switch it ON (idempotent; call every tick)."""
        self.camhub.request(name)

    def get_frame(self, name):
        """request(name) + return the latest single frame (copy), or None while warming up."""
        return self.camhub.get_frame(name)

    def active_cameras(self):
        """Names currently SUBSCRIBED (streaming) — what the GUI's live indicator shows."""
        return self.camhub.active_cameras()

    def get_intrinsics(self, name):
        """Camera intrinsics (live SDK value cached; falls back to defaults while OFF)."""
        return self.camhub.get_intrinsics(name)

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
            self._exec_thread = threading.Thread(target=self._sim_loop, daemon=True)
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
        rec = self.recorder.stop()

        for t in (self._collect_thread, self._exec_thread, self._release_thread):
            if t is not None and t.is_alive():
                t.join(timeout=5.0)
        self._collect_thread = self._exec_thread = self._release_thread = None

        # Finalize the grabbed recording now that the collect thread is gone.
        if rec is not None:
            self.recorder.finalize(rec)

        # Close every live camera subscription (the hub owns all of them).
        self.camhub.close_all()
        if self._owns_robot:
            try:
                self.robot.shutdown()
            except Exception as e:
                print(f"[HumanoidEnv.stop] robot.shutdown: {e}")

    # ===================== producer: collection loop =====================
    # Pacing + SDK reads copied from RobotDataCollector._stream_loop.
    def _collect_loop(self):
        """Producer: the SINGLE source of SDK reads for the INFERENCE OBSERVATION (+ recording
        and freshness/staleness). Once per tick it reads cameras, EE, joints and gripper, stamps
        them with one timestamp, and publishes the latest-wins buffers get_obs() snapshots.

        Scope is deliberately observation-only. The control path (ramp-in / release / splice)
        does NOT read from these buffers — it takes its own fresh synchronous reads
        (_read_arm14 etc.) because it needs the arm's pose AT command time, not a value up to one
        tick (~33ms) old. So concurrent direct SDK reads from those threads are intentional, not
        a bypass to consolidate here.
        """
        next_tick = time.monotonic()
        while not self._stop_event.is_set():
            now = time.time()
            now_mono = time.monotonic()

            # Camera producer step: evict idle, reconcile the live subscription set, read the
            # latest frame per camera. Returns this tick's fresh {name: [prev, cur]} pairs.
            frames = self.camhub.capture_tick()

            # Raw per-tick SDK reads (the env owns the robot/controller): motion status (firmware
            # error flag), gripper, and both-arm joints. The ObsCollector turns these into the
            # policy obs (left-EE FK, right-EE from status, grippers, freshness).
            try:
                status = self.robot_controller.get_motion_status()
                grip = self.robot.gripper_states()[0]     # array-like [left, right] RAW gripper
                arm14 = self._read_arm14()                # refresh last-good arm read (C4)
            except Exception as e:
                print(f"[HumanoidEnv]  [collect] SDK read failed: {e}")
                status = grip = arm14 = None

            with self._lock:
                cur = self._current_row_id                # master-ID this obs is anchored to
            self.obs.ingest(now, now_mono, status, grip, arm14, frames, cur)

            # Append a recording row while a session is active (no-op otherwise).
            self.recorder.tick(now, status, grip, frames)

            next_tick += self.dt
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                self._stop_event.wait(sleep_for)
            else:
                next_tick = time.monotonic()

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

    # ===================== obs snapshot for the caller (RealEnv.get_obs) =====================
    def get_obs(self):
        """Time-aligned dual_arm_ee_image obs for one inference, or None if not ready yet.
        Delegated to the ObsCollector (which pulls camera frames from the hub)."""
        return self.obs.get_obs(self.camhub)

    # ===================== caller -> env: hand off a prediction =====================
    def submit_actions(self, action_chunk):
        """Hand a chunk to the SIM-PREVIEW queue (auto-run / replay). Sim only — this path can
        NEVER reach the robot. The hardware path is validate_and_stage() + release_to_robot()."""
        self.preview.submit(action_chunk)

    def queue_empty(self):
        """True when no preview actions are pending (used by runners to detect drain)."""
        return self.preview.queue_empty()

    # ===================== chunk IK-solve + sim-validation (delegated to the pipeline) =====================
    def _solve_chunk_ik(self, action_chunk, seed_q, skip_unreachable=False):
        """Solve IK for an ENTIRE chunk (workspace check H4 + orientation smoothing H3 + dual-arm IK
        with nominal-posture fallback) on the caller's thread. Delegated to the pipeline; returns
        (configs, ok, reason) with configs = [(q14, [gl,gr]), ...] for the sim to validate."""
        return self.pipeline.solve_chunk_ik(action_chunk, seed_q, skip_unreachable=skip_unreachable)

    def _validate_chunk(self, configs, seed_q, fast=False, substeps_per_row=None):
        """Run PRE-SOLVED configs through self.sim from seed_q (substep + self-collision + joint
        readback) -> (traj, ok, reason, rows): traj = sim-ACHIEVED [(q14, grip), ...], rows[i] = the
        config row traj[i] realizes. The single validation primitive shared by manual
        validate_and_stage and auto-inference. Owns the no-sim / empty-chunk guards (env owns the sim);
        the kinematic check is the pipeline's.
        substeps_per_row: the SAME K the caller uses for master-id tagging; None -> current value."""
        if self.sim is None:
            return [], False, "no sim running — press 启动仿真预览 first", []
        if not configs:
            return [], False, "empty action chunk", []
        K = self.substeps_per_row if substeps_per_row is None else substeps_per_row
        return self.pipeline.validate_chunk(self.sim, configs, seed_q, K, fast=fast)

    def calibrate_collisions(self, action_chunk):
        """Learn this coarse URDF's inherent self-collision overlaps from a KNOWN-SAFE action
        chunk: walk it through the sim (same subdivide+settle as validation) and absorb every
        new left-side contact into the ignore baseline. Run over a representative corpus of safe
        recordings before trusting validation, or it will false-positive on normal poses."""
        if self.sim is None:
            return False, "no sim running"
        configs, ok, reason = self._solve_chunk_ik(action_chunk, self._last_q14,
                                                   skip_unreachable=True)
        if not ok:
            return False, reason
        self.sim.validate(configs, self._last_q14, MAX_JOINT_STEP, learn=True,
                          substeps_per_row=self.substeps_per_row)
        return True, "collision baseline updated"

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

        # Solve IK for the whole chunk first (outside the sim job), then validate the resulting
        # joint configs kinematically. On an IK/envelope failure there's nothing to validate.
        configs, ok, reason = self._solve_chunk_ik(action_chunk, self._last_q14)
        traj, ok, reason, _rows = (self._validate_chunk(configs, self._last_q14)
                                   if ok else ([], False, reason, []))
        with self._lock:
            if ok and traj:
                self._last_sim_traj = traj
                self._validation_id += 1
                # Advance the dual IK warm-start seed to the 14-joints achieved at the end of this
                # validated segment, so a subsequent step-through call (单步/整条) plans
                # continuously from here instead of restarting from the original seed.
                self._last_q14 = np.asarray(traj[-1][0], dtype=np.float64).copy()
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
        self.set_seed(arm14[:7])                          # LEFT preview IK seed -> real left joints
        self._last_q14 = np.asarray(arm14, dtype=np.float64).copy()   # dual seed -> real both arms

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
        arm14 = self._read_arm14()                       # current BOTH-arm pose for the C5 ramp-in
        if arm14 is None:
            print("[HumanoidEnv] release refused: cannot read arm_joint_states (ramp-in start).")
            return 0
        ramp = self.pipeline._ramp(arm14, traj[0][0], self.ramp_joint_step, traj[0][1])  # C5 ramp-in
        full = self.pipeline._subdivide_points(ramp + traj)   # <= MAX_JOINT_STEP for the per-tick drain
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
        """The next up-to-n staged substeps as (14-joint array, [gl, gr]) pairs (copies), in release
        order. grip is the binary {0,1} gripper command per arm. For the live substep monitor;
        cheap snapshot under the lock."""
        with self._lock:
            items = list(self._staged_release)[:n]
        return [(np.asarray(q, dtype=np.float64).copy(), [float(g[0]), float(g[1])])
                for (q, g) in items]

    def robot_q_preview(self, n=10):
        """The next up-to-n commands queued for the robot as (14-joint array, grip) pairs (copies), in
        execution order. Reflects upcoming real motion in BOTH manual-release and auto-inference modes.
        Delegated to the pipeline (which owns the queue)."""
        return self.pipeline.robot_q_preview(n)

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
        arm14 = self._read_arm14()                       # current BOTH-arm pose for the C5 ramp-in
        if arm14 is None:
            print("[HumanoidEnv] release refused: cannot read arm_joint_states (ramp-in start).")
            with self._lock:                             # don't lose the popped substeps
                self._staged_release.extendleft(reversed(subs))
            return 0
        with self._lock:
            # Bridge from the queue TAIL when already streaming, else from the live measured pose.
            # ALWAYS ramp anchor -> subs[0] then subdivide the whole thing, so every gap — including
            # the queue seam — is <= MAX_JOINT_STEP. Previously the streaming branch appended subs
            # directly, leaving the tail->subs[0] seam unbounded (a large IK jump there could leak an
            # over-cap substep). The dispatch guard also catches this, but bounding it here keeps the
            # queue itself clean.
            anchor = (np.asarray(self._robot_q[-1][0], dtype=np.float64)
                      if self._robot_q else arm14)
            ramp = self.pipeline._ramp(anchor, subs[0][0], self.ramp_joint_step, subs[0][1])
            full = ramp + list(subs)[1:]                 # ramp's last point IS subs[0]
            self._robot_q.extend(self.pipeline._subdivide_points(full))
            staged_substep_remaining = len(self._staged_release)
        print(f"[HumanoidEnv] queued {len(full)} cmd(s) to robot ({staged_substep_remaining} substep(s) still staged).")
        return len(full)

    # ===================== auto-inference: validate + master-ID splice (delegated to the pipeline) =====================
    def auto_ingest_chunk(self, action_chunk, obs_step_id):
        """Auto-inference entry point (queue-REPLACE variant): validate a predicted chunk on the
        preview sim and, on success, replace the live robot queue with a fresh ramp-in from the arm's
        current pose to the chunk's still-future rows, aligned by master row id. Delegated to the
        pipeline (which owns the queue + the splice); returns (ok, reason)."""
        return self.pipeline.auto_ingest_chunk(action_chunk, obs_step_id)

    def append_actions(self, action_chunk, obs_step_id, n_rows=None):
        """Streaming auto-inference entry point: keep n_rows policy rows queued ahead of the master
        clock, appending only the not-yet-queued ids with a velocity-matched seam ramp. Delegated to
        the pipeline (which owns the queue + the splice); returns (ok, reason)."""
        return self.pipeline.append_actions(action_chunk, obs_step_id, n_rows=n_rows)

    def lock_robot(self):
        """E-STOP (latched). Drop pending robot commands and ACTIVELY hold the arm at its
        current measured pose — the SDK has no brake, so re-commanding 'here' is the stop.
        Stays latched until reset_estop(). (The physical E-stop remains the primary safety.)"""
        self._estop.set()
        with self._lock:
            dropped = len(self._robot_q) + len(self._staged_release)
            self._robot_q.clear()
            self._staged_release.clear()             # drop un-released substeps too (re-validate after reset)
            self._queued_through = -1                # queue emptied -> next append re-anchors to clock
        print(f"[HumanoidEnv] E-STOP: latched; dropped {dropped} pending/staged cmds; holding pose.")
        if self._real:
            arm14 = self._read_arm14()
            if arm14 is None:
                print("[HumanoidEnv] E-stop WARNING: no joint read to hold — use physical E-stop.")
                return
            for _ in range(3):                           # re-assert a few times to be sure
                # Hold BOTH arms at their current pose; don't touch the grippers. NOTE: the SDK
                # has no trajectory abort, so this can only PREEMPT (not cancel) an in-flight
                # trajectory — clearing _robot_q above stops further batches; the physical E-stop
                # remains primary for an already-dispatched batch.
                self.run_trajectory_control([(arm14, None)],  # 14-vec both arms; None grip -> untouched
                                            ignore_estop=True)    # hold must send while estopped

    def reset_estop(self):
        """Clear the latched E-stop (operator confirms the arm is safe). Release stays refused
        until a fresh 执行→释放 (the previous trajectory was consumed)."""
        self._estop.clear()
        self._last_cmd_q14 = None       # arm may have moved while latched -> re-seed guard from live
        self.reset_grip_latch()         # fresh start -> a new grasp can re-latch from scratch
        print("[HumanoidEnv] E-stop reset; release re-enabled (run 执行 then 释放).")

    @property
    def estopped(self):
        return self._estop.is_set()

    @property
    def robot_pending(self):
        with self._lock:
            return len(self._robot_q)

    def set_speed_scale(self, speed_scale):
        """LIVE-tunable execution speed (= fraction of demo speed). Recomputes substeps_per_row (= K,
        the time-uniform substeps each policy row expands to) and ramp_joint_step together, so the sim
        expansion, the master-id tagging, and the seam-ramp cruise speed stay consistent. Only affects
        chunks validated AFTER the change (already-queued substeps keep their tags). Returns the
        applied (speed_scale, substeps_per_row)."""
        s = max(1e-3, float(speed_scale))
        self.speed_scale = s
        self.substeps_per_row = max(1, round((CONTROL_HZ / RECORD_HZ) / s))
        # Clamp to the C5 safety cap: s>1 must NOT push ramps above MAX_JOINT_STEP (auto-path ramps
        # reach the queue without a _subdivide_points re-clamp).
        self.ramp_joint_step = min(MAX_JOINT_STEP, MAX_JOINT_STEP * s)
        print(f"[HumanoidEnv] speed_scale={s:.3f} -> substeps_per_row={self.substeps_per_row}, "
              f"ramp_joint_step={self.ramp_joint_step:.4f}")
        return self.speed_scale, self.substeps_per_row

    def queue_status(self):
        """(current_row_id, queued_through) — the live master clock + streaming cursor. Delegated to
        the pipeline (which owns them). The inference controller uses this to split its smoothed buffer
        into the FROZEN (id <= queued_through) and MUTABLE (id > queued_through) regions."""
        return self.pipeline.queue_status()

    def set_seed(self, q7):
        """Set the sim-preview IK warm-start seed (e.g. to the robot's current left-arm joints).
        Delegated to the SimPreview, which owns that seed."""
        self.preview.set_seed(q7)

    # ------------------------------------------------------------------ #
    #  Delegating accessors: the robot queue + master clock + streaming    #
    #  cursor + dual IK seed + execution knobs live in self.pipeline. These #
    #  keep the env's (and tests'/GUI's) existing attribute API working.    #
    # ------------------------------------------------------------------ #
    @property
    def _robot_q(self):
        return self.pipeline._robot_q            # the deque itself (mutated in place under self._lock)

    @property
    def _current_row_id(self):
        return self.pipeline._current_row_id

    @_current_row_id.setter
    def _current_row_id(self, v):
        self.pipeline._current_row_id = v

    @property
    def _queued_through(self):
        return self.pipeline._queued_through

    @_queued_through.setter
    def _queued_through(self, v):
        self.pipeline._queued_through = v

    @property
    def _last_q14(self):
        return self.pipeline._last_q14

    @_last_q14.setter
    def _last_q14(self, v):
        self.pipeline._last_q14 = v

    @property
    def substeps_per_row(self):
        return self.pipeline.substeps_per_row

    @substeps_per_row.setter
    def substeps_per_row(self, v):
        self.pipeline.substeps_per_row = v

    @property
    def ramp_joint_step(self):
        return self.pipeline.ramp_joint_step

    @ramp_joint_step.setter
    def ramp_joint_step(self, v):
        self.pipeline.ramp_joint_step = v

    @property
    def append_ahead_rows(self):
        return self.pipeline.append_ahead_rows

    @append_ahead_rows.setter
    def append_ahead_rows(self, v):
        self.pipeline.append_ahead_rows = v

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

    def reset_grip_latch(self):
        """Clear the gripper close-latch state so a channel can re-open after a fresh start
        (E-stop / reset / new auto run). Both channels drop their closed streak + latch."""
        self._grip_closed_since = [None, None]
        self._grip_latched = [False, False]

    def move_to_joints(self, q14_target, joint_step=None):
        """Slowly move BOTH arms to an ABSOLUTE 14-joint target (rad) by streaming a velocity-bounded
        linear ramp from the current measured pose, one waypoint per STEP_TIME (drained by
        run_trajectory_control, which clamps + paces + honours the E-stop). joint_step = per-substep
        joint delta (rad); smaller = slower (default 0.3*MAX_JOINT_STEP, a gentle home). Grippers are
        left untouched. BLOCKS until the arm arrives. Intended for the auto-run start pose while the
        robot queue is idle. Returns True on completion (or an E-stop/shutdown halt), False on a bad
        read / dispatch error / E-stop refusal."""
        if self._estop.is_set():
            print("[HumanoidEnv] move_to_joints refused: E-stop latched.")
            return False
        start = self._read_arm14()
        if start is None:
            print("[HumanoidEnv] move_to_joints refused: cannot read arm_joint_states.")
            return False
        target = np.clip(np.asarray(q14_target, dtype=np.float64), self._jlower14, self._jupper14)
        step = float(joint_step) if joint_step else MAX_JOINT_STEP * 0.3
        span = float(np.max(np.abs(target - start)))
        n = max(1, int(np.ceil(span / max(step, 1e-6))))
        print(f"[HumanoidEnv] move_to_joints: |Δq|={span:.3f} rad over {n} substeps "
              f"(~{n * STEP_TIME:.1f}s) to the start pose.")
        pts = [(start + (target - start) * (i / n), None) for i in range(1, n + 1)]
        return self.run_trajectory_control(pts)

    def _latched_grip(self, grip_bin):
        """Anti-regrab latch applied at dispatch: given the binary [gl, gr] this substep would send,
        return the effective pair. A channel COMMANDED closed continuously for _grip_latch_sec is
        latched to closed (1) and stays closed until reset_grip_latch; any open command before the
        streak completes just resets that channel's timer. _grip_latch_sec None -> pass through."""
        if self._grip_latch_sec is None:
            return grip_bin
        now = time.monotonic()
        out = [grip_bin[0], grip_bin[1]]
        for i in (0, 1):
            if self._grip_latched[i]:
                out[i] = 1                                    # fixed closed
                continue
            if grip_bin[i] >= 1:                              # commanded closed -> grow the streak
                if self._grip_closed_since[i] is None:
                    self._grip_closed_since[i] = now
                elif now - self._grip_closed_since[i] >= self._grip_latch_sec:
                    self._grip_latched[i] = True
                    print(f"[HumanoidEnv] gripper channel {i} closed >= {self._grip_latch_sec:.1f}s "
                          f"-> LATCHED closed (anti-regrab; reset on E-stop/restart).")
                out[i] = 1
            else:                                             # commanded open -> streak broken
                self._grip_closed_since[i] = None
        return out

    # ===================== consumer: sim-preview execution loop =====================
    def _sim_loop(self):
        """Own the sim-preview thread + pacing; the per-tick work (decode -> workspace -> smooth ->
        IK -> sim.command) is the SimPreview's. Sim only — NEVER touches the robot (that is
        _release_loop). Passes the current self.sim in, since the GUI attaches/detaches it live."""
        while not self._stop_event.is_set():
            self.preview.tick(self.sim)
            self._stop_event.wait(self.dt)

    def _release_loop(self):
        """The ONLY driver of real-robot motion. Drains the pipeline's robot queue ONE substep per
        STEP_TIME via pipeline.pop_next_substep (which pops under the shared lock and advances the
        master clock), so an auto-inference splice into the queue takes effect on the very next tick.
        Substeps are pre-subdivided to <= MAX_JOINT_STEP at enqueue/splice time, so per-tick streaming
        stays velocity-bounded (C5). Each substep is its own single-waypoint trajectory, so an E-stop
        halts the arm within ~STEP_TIME (C3). The env is the CONSUMER; the pipeline is the producer."""
        # DIAGNOSTIC: per-substep timing split (recorder I/O vs firmware check vs dispatch), averaged
        # and printed ~1/s. If total >> STEP_TIME the release loop — not inference — caps arm speed.
        _t = {"rec": 0.0, "fw": 0.0, "disp": 0.0, "tot": 0.0, "n": 0, "log": time.monotonic()}
        # The per-substep trace recorders (below) do 2 file writes + an EXTRA DDS joint read every
        # tick. At idle that costs ~0.3ms, but under the live auto pipeline (3 cameras + inference +
        # IK/sim all contending for the GIL and the DDS bus) it balloons to ~20ms — over half the
        # 8.3ms tick budget — starving this 120Hz loop down to ~26Hz (arm ~5x slow). They are a debug
        # aid ONLY, so keep them OFF unless HUMANOID_SUBSTEP_TRACE is explicitly set.
        import os
        trace_substeps = os.environ.get("HUMANOID_SUBSTEP_TRACE", "") not in ("", "0", "false", "False")
        while not self._stop_event.is_set():
            if self._estop.is_set():
                self._stop_event.wait(STEP_TIME)
                continue
            sub = self.pipeline.pop_next_substep()       # pops + advances the master clock under the lock
            if sub is None:
                self._stop_event.wait(STEP_TIME)
                continue
            _t0 = time.monotonic()
            # The master row this substep realized (auto path tags a row id as the 3rd element; the
            # manual release path is a 2-tuple). pop_next_substep already advanced the clock; we read
            # the id here only for the diagnostic recorders below.
            row_id = int(sub[2]) if len(sub) > 2 and sub[2] is not None else None

            # Diagnostic recorders (debug aid; NEVER allowed to kill the release loop — this is the
            # ONLY robot-motion driver). Recorder 1: the exact substep popped/dispatched this tick —
            # the ABSOLUTE left-arm joint command (q7, rad) + binary gripper, keyed by master row id
            # (post-IK target actually sent, vs chunks.jsonl's EE-space chunk). Recorder 2: the LIVE
            # measured joints at the same tick, same id, for tracking-error / lag analysis. The whole
            # block is wrapped so a disk/IO error logs once and motion continues. Gated OFF by default
            # (see trace_substeps above) — the file I/O + extra DDS read here is what caps arm speed
            # under the live auto pipeline.
            if trace_substeps:
                try:
                    q14_cmd, grip_cmd = sub[0], sub[1]
                    with open(TRACE_DIR / "released_substeps.jsonl", "a") as f:
                        f.write(json.dumps({
                            "step_id": row_id,
                            "q14": np.asarray(q14_cmd, dtype=np.float64).tolist(),   # [left7, right7]
                            "grip": (None if grip_cmd is None else
                                     [float(grip_cmd[0]), float(grip_cmd[1])]),      # [gl, gr]
                        }) + "\n")
                    try:
                        live_vals, _ = self.robot.arm_joint_states()
                        live_joints = np.asarray(live_vals, dtype=np.float64).tolist()
                    except Exception:
                        live_joints = None
                    with open(TRACE_DIR / "live_joints.jsonl", "a") as f:
                        f.write(json.dumps({"step_id": row_id, "joints": live_joints}) + "\n")
                except Exception as e:
                    print(f"[HumanoidEnv] substep recorder failed (continuing): {e}")

            _t1 = time.monotonic()
            if self._firmware_unsafe():                        # defense-in-depth: firmware fault
                self.lock_robot()
                continue
            _t2 = time.monotonic()
            # run_trajectory_control([sub]) sends the one waypoint AND paces STEP_TIME internally,
            # so there is no extra wait here (that would double the period).
            if not self.run_trajectory_control([sub]):
                self.lock_robot()                              # couldn't dispatch -> estop
            _t3 = time.monotonic()
            _t["rec"] += _t1 - _t0; _t["fw"] += _t2 - _t1; _t["disp"] += _t3 - _t2
            _t["tot"] += _t3 - _t0; _t["n"] += 1
            if _t3 - _t["log"] >= 1.0 and _t["n"]:
                n = _t["n"]
                print(f"[release-timing] {n} substeps/s | per-substep avg: "
                      f"total {_t['tot']/n*1e3:.1f}ms = recorder {_t['rec']/n*1e3:.1f} + "
                      f"firmware {_t['fw']/n*1e3:.1f} + dispatch {_t['disp']/n*1e3:.1f} "
                      f"(STEP_TIME={STEP_TIME*1e3:.1f}ms)")
                _t = {"rec": 0.0, "fw": 0.0, "disp": 0.0, "tot": 0.0, "n": 0, "log": _t3}

    def run_trajectory_control(self, points, ignore_estop=False):
        """Stream a batch of sim-validated DUAL-arm waypoints to trajectory_tracking_control ONE at
        a time (ABS_JOINT, SDK doc 8.2.4), pacing at STEP_TIME and bailing on E-stop between points
        so the arm can be halted mid-batch (the SDK has no trajectory abort). robot_states is read
        once up front as the observation anchor (a read, never a command).

        points: [(q14, grip), ...] sim-achieved joints (rad) for BOTH arms — q14 = [left7, right7] —
        plus a binary {0,1} [gl, gr] gripper pair, or grip=None to leave the grippers untouched (the
        E-stop hold). BLOCKS for ~len(points) * STEP_TIME while streaming. Each waypoint drives both
        arms (left_arm=q14[:7], right_arm=q14[7:]) and, when grip is not None, both gripper channels.
        Returns True on normal completion or an E-stop/shutdown halt; False on a dispatch error."""
        if not points:
            return True
        # points are already <= MAX_JOINT_STEP apart (C5): subdivision now happens UPSTREAM at
        # enqueue/splice time, so the live _robot_q can be drained one substep per tick (and spliced
        # into mid-stream by auto-inference) without an over-cap velocity.
        # robot_states: the observation that anchors the trajectory (8.2.4). Reads only, optional.
        # The release loop calls this ONCE PER SUBSTEP (120Hz), so re-reading arm+waist+head here
        # meant 3 DDS round-trips every tick — under the live auto pipeline that DDS/GIL contention
        # inflated dispatch from ~0.3ms to ~10ms. These states barely change tick-to-tick and are only
        # an anchor, so cache them and refresh at ~RECORD_HZ (10Hz) instead of CONTROL_HZ (120Hz).
        now = time.monotonic()
        if getattr(self, "_rs_cache", None) is None or (now - self._rs_cache_t) >= ROW_DT:
            rs = {}
            try:
                rs["arm"] = list(self.robot.arm_joint_states()[0])
            except Exception:
                pass
            try:
                rs["waist"] = list(self.robot.waist_joint_states()[0])
            except Exception:
                pass
            try:
                rs["head"] = list(self.robot.head_joint_states()[0])
            except Exception:
                pass
            self._rs_cache = rs
            self._rs_cache_t = now
        robot_states = self._rs_cache
        # Seed the dispatch guard from the live pose on the first command so even the very first
        # waypoint is velocity-bounded from where the arm actually is.
        if self._last_cmd_q14 is None and not ignore_estop:
            live = self._read_arm14()
            if live is not None:
                self._last_cmd_q14 = np.asarray(live, dtype=np.float64).copy()

        # Stream the batch ONE waypoint at a time (not one big trajectory). The SDK has no
        # trajectory abort, so this is what makes an E-stop effective: latched mid-batch, we simply
        # stop sending the next waypoint and the arm halts within ~STEP_TIME at the last point.
        # Each waypoint is its own single-waypoint ABS_JOINT trajectory (8.2.4 schema) driving BOTH
        # arms; reference_time = STEP_TIME; we pace on self._stop_event so a shutdown breaks promptly.
        for item in points:
            q14, grip = item[0], item[1]                 # tolerate (q14, grip) or (q14, grip, row_id)
            if not ignore_estop and (self._estop.is_set() or self._stop_event.is_set()):
                return True                                  # halt: send no further waypoints
            q14 = np.clip(np.asarray(q14, dtype=np.float64),
                          self._jlower14, self._jupper14)    # 14-joint limit clamp (no IK solver needed)
            # C5 DISPATCH GUARD (defense-in-depth, BOTH arms): never send a waypoint more than
            # MAX_JOINT_STEP from the last commanded config. If the target jumps further — a large IK
            # step, a queue seam, a redundancy branch switch, a jumpy policy row — RAMP it: subdivide
            # into ceil(|Δq|/cap) linear waypoints and stream each, paced. This makes an over-cap joint
            # velocity physically impossible regardless of what produced `points`. Upstream ramps keep
            # this a no-op in the common case; when it fires it means an upstream gap leaked through.
            prev = self._last_cmd_q14
            if prev is not None and not ignore_estop and MAX_JOINT_STEP > 0:
                span = float(np.max(np.abs(q14 - prev)))
                nseg = max(1, int(np.ceil(span / MAX_JOINT_STEP)))
                if nseg > 1:
                    print(f"[HumanoidEnv] dispatch-ramp: |Δq|={span:.3f} rad exceeds cap "
                          f"{MAX_JOINT_STEP:.3f} -> streaming {nseg} bounded substeps")
                waypoints = [prev + (q14 - prev) * (i / nseg) for i in range(1, nseg + 1)]
            else:
                waypoints = [q14]
            # Grippers (binary) accompany the motion; send once for the whole (possibly ramped) step.
            # Anti-regrab close-latch applied here so a sustained grasp can't be re-opened by a toggling
            # policy (see _latched_grip); the hold path (ignore_estop) never touches the grippers.
            if grip is not None:
                gl, gr = grip
                self.robot.move_gripper(self._latched_grip([1 if gl >= 0.5 else 0,
                                                            1 if gr >= 0.5 else 0]))
            for wp in waypoints:
                if not ignore_estop and (self._estop.is_set() or self._stop_event.is_set()):
                    return True
                robot_action = [{"left_arm":  {"action_data": wp[:7].tolist(),  "control_type": "ABS_JOINT"},
                                 "right_arm": {"action_data": wp[7:].tolist(),  "control_type": "ABS_JOINT"}}]
                try:
                    self.robot_controller.trajectory_tracking_control(
                        int(time.time() * 1e9), robot_states, robot_action, "base_link", STEP_TIME)
                except Exception as e:
                    print(f"[HumanoidEnv] trajectory_tracking_control failed: {e}")
                    return False
                self._last_cmd_q14 = wp                       # advance the guard's reference
                self._stop_event.wait(STEP_TIME)             # pace; interruptible by stop_event
        return True

    # _subdivide_points, _ramp, _hermite_ramp now live in the pipeline (real_world.postprocess); the
    # manual-release path calls self.pipeline._subdivide_points / self.pipeline._ramp.

    # ===================== data-collection recording (delegated to Recorder) =====================
    def start_recording(self, episode_name=None):
        """Begin a recording session (writes mp4 per RECORD_CAMERA + an npz). Pins the record
        cameras so a tick never misses one to idle-eviction. Delegated to self.recorder; the env
        only owns the camera-pin side effect."""
        if self.recorder.start(episode_name):
            self.camhub.pin(RECORD_CAMERAS)

    def stop_recording(self):
        """End the active session and flush video/npz/metadata to disk. Unpins the record
        cameras and finalizes outside the recorder lock (the collect loop already sees the
        session gone, so writing files can't block streaming)."""
        rec = self.recorder.stop()
        if rec is None:
            return
        self.camhub.unpin(RECORD_CAMERAS)
        self.recorder.finalize(rec)
        print("[HumanoidEnv] Recording stopped.")
