"""In-process PyBullet sim backend for the G1 left arm (SDK-free).

Owns the PyBullet world + the URDF IK model, and offers two ways to drive the arm:

* `replay(recording, ...)`  — the offline recording replay used by scripts/sim_replay_eval
  (recorded EE poses -> IK -> step until settle).
* `command(q7, grip)` / `step()` — a live drive path: a producer (e.g. HumanoidEnv's exec
  thread) calls `command()` to hand over the latest 7 joint targets; the **owning thread**
  calls `step()` to apply them and advance the sim. The handoff is a lock-guarded "latest
  target" variable, NOT a socket — IK is an in-process method and PyBullet is in-process, so
  no IPC is needed between the solver and the sim.

PyBullet constraint: every `p.*` call must come from the thread that called `p.connect`
(the "owning thread"). `command()` makes no `p.*` call (safe from any thread); `step()`,
`replay()`, `reset_arm()`, and `run_until()` must run on the owning thread.

SDK-free on purpose: imports only numpy + pybullet + real_world.ik, so it runs in the
sim/CI path without `a2d_sdk`.
"""

import os
import queue
import tempfile
import threading
import time

import numpy as np
import pybullet as p

from real_world.ik import (
    PinocchioArmModel, PinocchioDLSIKSolver, IKQuery,
    quat_xyzw_to_se3, LEFT_ARM_JOINTS, DEFAULT_URDF, DEFAULT_CALIBRATION,
    load_calibration,
)
from real_world.timing import CONTROL_HZ, SUBSTEPS_PER_ROW

RIGHT_ARM_JOINTS = [f"Joint{i}_r" for i in range(1, 8)]
LEFT_GRIPPER_JOINTS = ([f"left_narrow{i}_joint" for i in (1, 2, 3, 4)] +
                       [f"left_wide{i}_joint" for i in (1, 2, 3, 4)])
RIGHT_GRIPPER_JOINTS = ([f"right_narrow{i}_joint" for i in (1, 2, 3, 4)] +
                        [f"right_wide{i}_joint" for i in (1, 2, 3, 4)])
WAIST_PITCH_JOINT = "joint_body_pitch"   # the URDF's only torso DOF (no head/lift joints)
# Finger joint angle for a fully OPEN gripper. Measured from the URDF: at angle 0 the jaws are
# together (CLOSED, ~27mm tip gap) and at ~0.6 rad they are apart (OPEN, ~70mm) — larger angle =
# more open. (The old name GRIPPER_CLOSE_RAD was backwards and rendered open/close swapped.)
GRIPPER_OPEN_RAD = 0.6


def _grip_to_finger_angle(grip):
    """Normalized grip (0 = open, 1 = closed) -> URDF finger joint angle. Inverted because angle 0
    is CLOSED and GRIPPER_OPEN_RAD is OPEN: grip 1 -> 0 rad (jaws together), grip 0 -> 0.6 (apart)."""
    return (1.0 - float(np.clip(grip, 0.0, 1.0))) * GRIPPER_OPEN_RAD
# Self-collision is flagged only when two links INTERPENETRATE deeper than this (metres). The
# URDF's coarse collision meshes (esp. the torso) over-approximate the real body by several cm
# and graze at normal poses; requiring real penetration filters that noise while still catching
# genuine collisions. Tune per URDF; pair with calibrate_collisions() over safe data if needed.
SELF_COLLISION_PENETRATION = 0.02   # 2 cm


def patched_urdf(urdf_path):
    """PyBullet can't resolve package://; rewrite to relative and load from the URDF dir."""
    with open(urdf_path) as f:
        text = f.read().replace("package://", "")
    d = os.path.dirname(urdf_path)
    fd, tmp = tempfile.mkstemp(suffix=".urdf", dir=d)
    with os.fdopen(fd, "w") as f:
        f.write(text)
    return tmp, d


def load_trajectory(recordings_dir, recording, max_frames=0):
    """Read a recording's left-EE poses, grip, and live arm joints from robot_states.npz."""
    path = os.path.join(recordings_dir, recording, "robot_states.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"no recording at {path}")
    npz = np.load(path)
    ee_pos = npz["left_pos"].astype(np.float64)
    ee_quat = npz["left_quat"].astype(np.float64)
    grip = npz["gripper"].astype(np.float64)[:, 0]
    arm_joints = npz["arm_joints"].astype(np.float64)   # (T, 14)
    n = len(ee_pos) if not max_frames else min(len(ee_pos), max_frames)
    return ee_pos, ee_quat, grip, arm_joints, n


class SimEnv:
    """PyBullet world + URDF IK for the G1 left arm. Drives the arm by recording replay or
    by a live `command()`/`step()` handoff. All `p.*` calls stay on the owning thread."""

    def __init__(self,
                 urdf=DEFAULT_URDF,
                 ee_frame=None,
                 calibration=DEFAULT_CALIBRATION,
                 sim_hz=240.0,
                 direct=False,
                 anchor_first=False,
                 fixed_rate=False,
                 settle_tol_deg=1.0,
                 settle_timeout_s=1.5,
                 playback_hz=20.0):
        self.direct = direct
        self.anchor_first = anchor_first
        self.fixed_rate = fixed_rate

        # --- IK model + solver (calibrated, constant base_offset) ---
        self.cal = load_calibration(calibration)
        ee_frame = ee_frame or self.cal.ee_frame
        self.model = PinocchioArmModel(urdf_path=urdf, ee_frame=ee_frame)
        self.left_slice = slice(0, 7) if self.cal.arm_joint_slice == "left_first" else slice(7, 14)
        if not self.anchor_first:
            self.model.base_offset = self.cal.base_offset_se3()
        self.solver = PinocchioDLSIKSolver(self.model, damping=1e-2, max_iters=100, tol=1e-4)

        # --- PyBullet ---
        p.connect(p.DIRECT if self.direct else p.GUI)
        tmp_urdf, search_dir = patched_urdf(urdf)
        p.setAdditionalSearchPath(search_dir)
        p.setGravity(0, 0, 0)                          # kinematic view: arm holds commanded pose
        # NOTE: we deliberately do NOT pass URDF_USE_SELF_COLLISION. Enabling it makes the
        # model's inherent mesh overlaps (gripper fingers, arm-base) generate contact forces
        # that fling the unactuated right arm/body/grippers around. Self-collision is instead
        # detected force-free via getClosestPoints (see _penetrating_pairs), which needs no flag
        # and doesn't perturb the kinematic preview.
        try:
            self.body = p.loadURDF(tmp_urdf, useFixedBase=True)
        finally:
            os.remove(tmp_urdf)
        self.jmap = {}
        self.jlim = {}
        self.link_name = {-1: "base"}
        for i in range(p.getNumJoints(self.body)):
            ji = p.getJointInfo(self.body, i)
            name = ji[1].decode()
            self.jmap[name] = i
            self.jlim[name] = (ji[8], ji[9])     # (lower, upper); lower>upper => no limit
            self.link_name[i] = ji[12].decode()  # child link name of this joint
        self.arm_idx = [self.jmap[j] for j in LEFT_ARM_JOINTS]
        self.grip_idx = [self.jmap[j] for j in LEFT_GRIPPER_JOINTS if j in self.jmap]
        # Self-collision trigger set = LEFT ARM links only (Joint*_l children). The right
        # arm/gripper's inherent overlaps are irrelevant to a left-arm trajectory, and the LEFT
        # gripper fingers' coarse, grip-driven meshes brush the arm base constantly -> excluded
        # to avoid false positives. The real hazard we gate on is an ARM linkage hitting the
        # torso / the other arm. (Gripper/environment hazards: firmware collisions + operator +
        # physical E-stop.)
        self._left_links = set(self.arm_idx)
        # Parent/child link pairs always meet at their shared joint -> exclude from the
        # force-free distance check (no EXCLUDE_PARENT flag applies to getClosestPoints).
        self._adjacent = set()
        for i in range(p.getNumJoints(self.body)):
            parent = p.getJointInfo(self.body, i)[16]   # parent link index (-1 = base)
            self._adjacent.add(tuple(sorted((i, parent))))

        # --- stepping / settling ---
        self.sim_hz = sim_hz
        self.sim_dt = 1.0 / sim_hz
        p.setTimeStep(self.sim_dt)
        self.settle_tol = np.radians(settle_tol_deg)
        self.settle_tol_deg = settle_tol_deg
        self.settle_max_steps = max(1, int(settle_timeout_s * sim_hz))
        self.fixed_steps = max(1, int(sim_hz / playback_hz))

        # --- self-collision baseline: left-arm link pairs already interpenetrating at the
        # loaded/rest pose are inherent geometry overlaps; ignore them. Validation flags only
        # NEW penetrations. getClosestPoints is force-free and needs no step priming.
        self._ignore_pairs = self._penetrating_pairs()

        # --- live-drive handoff: latest (q7, grip) target, written by any thread ---
        self._target_lock = threading.Lock()
        self._target = None        # (np.ndarray(7,), float grip) or None

        # --- job queue: validation must run on the owning thread (PyBullet single-thread). Any
        # other thread submits a job; the owning thread runs it inside step()/run_until. ---
        self._owner_tid = threading.get_ident()
        self._job_q = queue.Queue()

    def connected(self):
        return p.isConnected()

    def disconnect(self):
        if p.isConnected():
            p.disconnect()

    # ===================== live drive (producer + owning thread) =====================
    def command(self, q7, gripper):
        """Hand over the latest 7 joint targets + gripper. Thread-safe; no `p.*` call.

        Called by the producer (HumanoidEnv exec thread). `step()` (owning thread) applies it.
        """
        q = np.clip(np.asarray(q7, dtype=np.float64), self.model.lower, self.model.upper)
        with self._target_lock:
            self._target = (q, float(gripper))

    def reset_arm(self, q7):
        """Snap the left arm to a seed config (owning thread; start of a run)."""
        q = np.clip(np.asarray(q7, dtype=np.float64), self.model.lower, self.model.upper)
        for k, qi in zip(self.arm_idx, q):
            p.resetJointState(self.body, k, float(qi))

    def _set_joints(self, name_to_pos):
        """resetJointState a set of named joints, clamped to URDF limits. Owning thread."""
        for name, pos in name_to_pos.items():
            idx = self.jmap.get(name)
            if idx is None:
                continue
            lo, hi = self.jlim.get(name, (1.0, -1.0))
            if lo < hi:
                pos = min(max(float(pos), lo), hi)
            p.resetJointState(self.body, idx, float(pos))

    def reset_full(self, arm14=None, body_pitch=None, gripper_lr=None):
        """Snap the WHOLE model to a physical-robot reading so the sim starts matched.

        arm14:       14 joint angles (rad), [left 0..6, right 0..6] (SDK arm_joint_states).
        body_pitch:  waist pitch (rad) -> joint_body_pitch (the only torso DOF in the URDF;
                     the model has no head or waist-lift joints, so those can't be mirrored).
        gripper_lr:  [left, right] normalized [0,1] -> each side's 8 finger joints.
        Owning thread only. Leaves the live-drive target untouched, so step() holds this pose
        until the exec loop commands the left arm.
        """
        cmds = {}
        if arm14 is not None:
            arm14 = np.asarray(arm14, dtype=np.float64)
            for i, n in enumerate(LEFT_ARM_JOINTS):
                cmds[n] = arm14[i]
            for i, n in enumerate(RIGHT_ARM_JOINTS):
                cmds[n] = arm14[7 + i]
        if body_pitch is not None:
            cmds[WAIST_PITCH_JOINT] = float(body_pitch)
        if gripper_lr is not None:
            gl = _grip_to_finger_angle(gripper_lr[0])
            gr = _grip_to_finger_angle(gripper_lr[1])
            for n in LEFT_GRIPPER_JOINTS:
                cmds[n] = gl
            for n in RIGHT_GRIPPER_JOINTS:
                cmds[n] = gr
        self._set_joints(cmds)

    # ===================== jobs that must run on the owning thread =====================
    def _drain_jobs(self):
        """Run any queued owning-thread jobs (e.g. validation). Owning thread only."""
        while True:
            try:
                fn, box, ev = self._job_q.get_nowait()
            except queue.Empty:
                return
            try:
                box["result"] = fn()
            except Exception as e:                       # surface to the submitting thread
                box["err"] = e
            finally:
                ev.set()

    def submit_job(self, fn):
        """Run `fn` on the owning thread and return its result. Inline if already on the owning
        thread; otherwise enqueue and block until step()/run_until executes it."""
        if threading.get_ident() == self._owner_tid:
            return fn()
        box, ev = {}, threading.Event()
        self._job_q.put((fn, box, ev))
        ev.wait()
        if "err" in box:
            raise box["err"]
        return box.get("result")

    def step(self):
        """Run pending owning-thread jobs, apply the latest commanded target (if any), and
        advance one sim step. Owning thread only. GUI mode sleeps one dt for near real time."""
        self._drain_jobs()
        with self._target_lock:
            target = self._target
        if target is not None:
            q7, grip = target
            p.setJointMotorControlArray(self.body, self.arm_idx, p.POSITION_CONTROL,
                                        targetPositions=q7.tolist(),
                                        forces=[200.0] * len(self.arm_idx))
            if self.grip_idx:
                g = _grip_to_finger_angle(grip)
                p.setJointMotorControlArray(self.body, self.grip_idx, p.POSITION_CONTROL,
                                            targetPositions=[g] * len(self.grip_idx),
                                            forces=[20.0] * len(self.grip_idx))
        p.stepSimulation()
        if not self.direct:
            time.sleep(self.sim_dt)

    def run_until(self, stop_fn):
        """Step the sim on the owning thread until `stop_fn()` is true or the window closes."""
        try:
            while self.connected() and not stop_fn():
                self.step()
        finally:
            self.disconnect()

    def idle_step(self):
        """Step once with no command applied (keeps the GUI responsive while idle)."""
        self._drain_jobs()
        if self.direct:
            time.sleep(0.02)
        else:
            p.stepSimulation()
            time.sleep(self.sim_dt)

    def _cur_arm_q(self):
        return np.array([p.getJointState(self.body, k)[0] for k in self.arm_idx])

    # ===================== validation (run on the owning thread) =====================
    def validate(self, configs, seed_q, max_joint_step, learn=False, fast=False, substeps_per_row=None):
        """Run PRE-SOLVED joint configs through the sim; return (validated_traj, ok, reason).
        Thread-safe entry (marshals to the owning thread via submit_job).

        configs: [(q7, grip), ...] — the raw IK joints + gripper for each chunk row, ALREADY
        solved by the caller (IK + workspace/orientation checks live OUTSIDE the sim job now,
        on the caller's thread; see HumanoidEnv._solve_chunk_ik). The sim job is purely the
        kinematic check: expand each row gap into SUBSTEPS_PER_ROW time-uniform configs (so
        inter-row wall-clock = ROW_DT, independent of motion size), reject a row whose per-
        substep delta exceeds max_joint_step (the velocity ceiling), apply, settle, self-
        collision check, and record the sim-ACHIEVED joints (not raw IK).
        learn=False (validate): abort (ok=False) on any NEW left-side self-collision.
        learn=True (calibrate): never abort on collision — instead ABSORB every new left-side
        pair into the ignore baseline (used to learn this coarse URDF's inherent overlaps from
        known-SAFE motions). Leaves the arm where it ended.
        fast=True: skip the per-substep real-time sleep (auto-inference latency path).
        substeps_per_row: LIVE override of the per-row substep count (= SPEED_SCALE knob); None ->
        the module default. The caller (HumanoidEnv) passes its own value so the sim expansion and the
        env's master-id tagging (K) always use the SAME count, even when SPEED_SCALE changes live."""
        return self.submit_job(
            lambda: self._validate_impl(configs, seed_q, max_joint_step, learn, fast, substeps_per_row))

    def _validate_impl(self, configs, seed_q, max_joint_step, learn=False, fast=False,
                       substeps_per_row=None):
        K = int(substeps_per_row) if substeps_per_row else SUBSTEPS_PER_ROW
        seed = np.asarray(seed_q, dtype=np.float64).copy()
        # Deterministic start (so learn & validate see the same path, and repeat runs agree):
        # reset the left arm to the seed AND the gripper to open (GRIPPER_OPEN_RAD, since angle 0
        # is CLOSED — see _grip_to_finger_angle).
        self.reset_arm(seed)
        for k in self.grip_idx:
            p.resetJointState(self.body, k, GRIPPER_OPEN_RAD)
        # Clip every row config up front: the C1 inter-row interpolation (centripetal Catmull-Rom)
        # sets each waypoint's tangent from its NEIGHBOURING rows, so it needs the whole sequence and
        # can't expand one gap in isolation like a plain lerp. P[k] = row k's joint target.
        P = [np.clip(np.asarray(q7, dtype=np.float64), self.model.lower, self.model.upper)
             for (q7, _g) in configs]
        grips = [float(g) for (_q, g) in configs]
        out = []
        for k in range(len(P)):
            q7, grip = P[k], grips[k]
            # Time-uniform expansion: each row gap becomes a FIXED SUBSTEPS_PER_ROW substeps (uniform
            # in time), so inter-row wall-clock is always ROW_DT regardless of motion size and the
            # splice's f=elapsed/STEP_TIME stays exact. Smoothness is set by CONTROL_HZ (=>
            # SUBSTEPS_PER_ROW), NOT by joint displacement. Row 0 emits a single config: the
            # seed->row0 transient is the ramp-in, added (velocity-bounded) by the release/splice layer.
            if k == 0:
                subs = [q7]
            else:
                # Centripetal Catmull-Rom through row k-1 -> k, with the rows on either side as
                # tangent context (duplicated at the chunk ends). Curving through the waypoints with
                # MATCHED endpoint velocities makes joint velocity continuous across row boundaries
                # (C1) instead of the lerp's per-row staircase -> the jerk spike at each row is gone.
                before = P[k - 2] if k >= 2 else P[k - 1]
                after = P[k + 1] if k + 1 < len(P) else P[k]
                subs = self._catmull_rom_segment(before, P[k - 1], P[k], after, K)
                # Safety velocity ceiling, checked on the ACTUAL curve: a cubic peaks faster than the
                # straight-line average, so cap the largest per-substep joint delta ALONG the spline
                # (incl. the step from the previous row) rather than the |dq_row|/K linear estimate.
                if not learn:
                    per_sub = float(np.max(np.abs(np.diff(np.vstack([P[k - 1][None], subs]), axis=0))))
                    if per_sub > max_joint_step + 1e-9:
                        return [], False, (
                            f"action {k}: joint-velocity cap exceeded "
                            f"({np.degrees(per_sub * CONTROL_HZ):.0f} deg/s > "
                            f"{np.degrees(max_joint_step * CONTROL_HZ):.0f} deg/s)")
            for sub in subs:
                if fast:
                    # Collision-only fast path: KINEMATIC TELEPORT, no physics settle. The preview is
                    # gravity-free position-hold, so resetJointState to `sub` IS the achieved pose,
                    # and getClosestPoints (below) is a geometric query computed from the current
                    # config — it needs no stepSimulation. This drops the per-substep settle loop
                    # (up to settle_max_steps steps), which was the dominant validation cost.
                    self.reset_arm(sub)
                    if self.grip_idx:
                        g = _grip_to_finger_angle(grip)
                        for gk in self.grip_idx:
                            p.resetJointState(self.body, gk, g)
                else:
                    self._apply_arm(sub, grip)         # position control + physics settle (GUI watch)
                    self._settle(sub)
                new = self._new_left_pairs()
                if new:
                    if learn:
                        self._ignore_pairs |= new          # absorb inherent overlap
                    else:
                        a, b = sorted(new)[0]
                        return [], False, (f"action {k}: self-collision "
                                           f"{self.link_name.get(a, a)} <-> {self.link_name.get(b, b)}")
                out.append((self._cur_arm_q().copy(), float(grip)))   # sim-ACHIEVED, not raw IK
        return out, True, None

    @staticmethod
    def _resample_uniform(q_from, q_to, k):
        """k TIME-uniform configs from q_from->q_to (excl. start, incl. end), straight line. Fixed
        count = k, so one policy-row gap always occupies the same wall-clock (ROW_DT) no matter how
        far the joints move. Superseded on the row path by _catmull_rom_segment (C1); kept as the
        degenerate/fallback interpolator."""
        q_from = np.asarray(q_from, dtype=np.float64)
        q_to = np.asarray(q_to, dtype=np.float64)
        k = max(int(k), 1)
        return [q_from + (q_to - q_from) * (i / k) for i in range(1, k + 1)]

    @staticmethod
    def _catmull_rom_segment(p0, p1, p2, p3, k, alpha=0.5):
        """k configs along the centripetal Catmull-Rom spline on the p1->p2 segment (excl. p1, incl.
        p2 — same count/endpoints as _resample_uniform, so the row's wall-clock and the f-index are
        unchanged). p0 and p3 are the neighbouring rows; they only set the endpoint tangents so that
        adjacent segments share the same velocity at p1/p2 -> C1 across row boundaries.

        Barry-Goldman pyramidal evaluation (no explicit tangent scaling, robust to non-uniform knot
        spacing). TUNE alpha [0..1] = inter-row spline curvature on the robot path: 0 = uniform (C1
        but OVERSHOOTS the waypoints / can loop), 0.5 = centripetal (no cusps/self-intersections,
        bounded overshoot — recommended), 1 = chordal. A duplicated neighbour at a chunk end (p0==p1
        or p3==p2) is REFLECTED to a one-sided tangent; a zero-length p1->p2 segment returns a hold."""
        p0 = np.asarray(p0, dtype=np.float64); p1 = np.asarray(p1, dtype=np.float64)
        p2 = np.asarray(p2, dtype=np.float64); p3 = np.asarray(p3, dtype=np.float64)
        k = max(int(k), 1)
        if np.linalg.norm(p2 - p1) < 1e-9:                 # stationary row -> hold (avoids div-by-0)
            return [p2.copy() for _ in range(k)]
        if np.linalg.norm(p1 - p0) < 1e-9:                 # chunk start / repeated row: mirror p2
            p0 = 2.0 * p1 - p2
        if np.linalg.norm(p3 - p2) < 1e-9:                 # chunk end / repeated row: mirror p1
            p3 = 2.0 * p2 - p1
        t0 = 0.0
        t1 = t0 + float(np.linalg.norm(p1 - p0)) ** alpha
        t2 = t1 + float(np.linalg.norm(p2 - p1)) ** alpha
        t3 = t2 + float(np.linalg.norm(p3 - p2)) ** alpha
        out = []
        for i in range(1, k + 1):
            t = t1 + (t2 - t1) * (i / k)
            a1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
            a2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
            a3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
            b1 = (t2 - t) / (t2 - t0) * a1 + (t - t0) / (t2 - t0) * a2
            b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3
            out.append((t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2)
        return out

    def _apply_arm(self, q7, grip):
        p.setJointMotorControlArray(self.body, self.arm_idx, p.POSITION_CONTROL,
                                    targetPositions=np.asarray(q7, dtype=np.float64).tolist(),
                                    forces=[200.0] * len(self.arm_idx))
        if self.grip_idx:
            g = _grip_to_finger_angle(grip)
            p.setJointMotorControlArray(self.body, self.grip_idx, p.POSITION_CONTROL,
                                        targetPositions=[g] * len(self.grip_idx),
                                        forces=[20.0] * len(self.grip_idx))

    def _settle(self, q_des, fast=False):
        """Step until the arm reaches q_des (or settle timeout). Sleeps in GUI mode so the user
        sees the validated motion play out — UNLESS fast=True (auto-inference), where the per-step
        real-time sleep is pure latency (nobody is watching the substeps): the GUI still renders
        the achieved poses, just faster. This sleep is the dominant cost of one inference cycle."""
        q_des = np.asarray(q_des, dtype=np.float64)
        for _ in range(self.settle_max_steps):
            p.stepSimulation()
            if not self.direct and not fast:
                time.sleep(self.sim_dt)
            if np.max(np.abs(self._cur_arm_q() - q_des)) < self.settle_tol:
                break

    def calibrate_collision(self, arm_configs):
        """Union the self-collision pairs seen across KNOWN-SAFE left-arm configs into the ignore
        baseline, then return its size. This URDF's torso/base collision meshes are coarse and
        overlap the arm at normal operating poses; those overlaps are inherent (not real
        collisions), so feed a representative corpus of safe configs (e.g. from recordings)
        before validating. Genuinely novel/dangerous folds still trip NEW pairs. Owning thread."""
        def job():
            keep = self._cur_arm_q()
            for q in arm_configs:
                self.reset_arm(q)
                self._ignore_pairs |= self._penetrating_pairs()
            self.reset_arm(keep)
            return len(self._ignore_pairs)
        return self.submit_job(job)

    def _penetrating_pairs(self):
        """LEFT-ARM link vs any-other-link pairs interpenetrating beyond
        SELF_COLLISION_PENETRATION, via getClosestPoints (geometry only -> no contact forces, no
        URDF self-collision flag needed). Adjacent (parent/child) pairs are skipped."""
        n = p.getNumJoints(self.body)
        others = [-1] + list(range(n))
        hits = set()
        for la in self.arm_idx:
            for lb in others:
                if lb == la or tuple(sorted((la, lb))) in self._adjacent:
                    continue
                for cp in p.getClosestPoints(self.body, self.body, 0.0,
                                             linkIndexA=la, linkIndexB=lb):
                    if cp[8] < -SELF_COLLISION_PENETRATION:     # cp[8] = signed distance (<0 = overlap)
                        hits.add(tuple(sorted((la, lb))))
                        break
        return hits

    def _new_left_pairs(self):
        """Set of NEW penetrating left-arm pairs (beyond the rest-pose baseline)."""
        return self._penetrating_pairs() - self._ignore_pairs

    # ===================== recording replay (owning thread) =====================
    def replay(self, recording, recordings_dir, max_frames, emit):
        """Replay one recording. `emit(str)` receives each log line (stdout + client)."""
        ee_pos, ee_quat, grip, arm_joints, n = load_trajectory(
            recordings_dir, recording, max_frames)
        emit(f"Recording {recording}: {n} EE poses")

        # Seed from the recording's first live left-arm joints, clamped to URDF limits.
        q_seed = np.clip(arm_joints[0, self.left_slice], self.model.lower, self.model.upper)
        if self.anchor_first:
            # Re-anchor base_offset so the EE starts exactly at the seed's FK.
            E0 = self.model.fk_local(q_seed)
            S0 = quat_xyzw_to_se3(ee_pos[0], ee_quat[0])
            self.model.base_offset = S0 * E0.inverse()
            emit("base_offset: anchored to this recording's first EE pose (--anchor-first)")
        else:
            self.model.base_offset = self.cal.base_offset_se3()
            emit(f"base_offset: calibrated; residual "
                 f"{self.cal.pos_residual_m * 1000:.1f} mm / {self.cal.rot_residual_deg:.2f} deg")

        # Reset the arm to the seed so each recording starts clean.
        self.reset_arm(q_seed)

        idx = np.linspace(0, n - 1, n).astype(int)
        q = q_seed.copy()
        ik_err, steps, n_timeout, n_skipped = [], [], 0, 0
        mode = "fixed-rate" if self.fixed_rate else f"settle(tol={self.settle_tol_deg}deg)"
        emit(f"mode: {mode}")
        emit(f"{'frame':>6} | {'IK pos err(mm)':>14} {'|dq| step(deg)':>14} "
             f"{'settle steps':>12} {'near-lim':>8}")

        for i in idx:
            # warm-start: last config is current
            q_des = self.solver.solve(IKQuery(target_pos=ee_pos[i], target_quat=ee_quat[i],
                                              current_joints=q))

            # Unreachable target: skip it entirely. Don't command the arm and DON'T advance
            # the seed — the next target is planned from the last reachable config, so a skip
            # can't poison the warm-start.
            if not self.solver.last_reachable:
                n_skipped += 1
                emit(f"{i:>6} | {self.solver.last_pos_err * 1000:>14.2f} "
                     f"{'SKIP (unreachable)':>14} {'-':>12} {'-':>8}")
                continue

            p.setJointMotorControlArray(self.body, self.arm_idx, p.POSITION_CONTROL,
                                        targetPositions=q_des.tolist(),
                                        forces=[200.0] * len(self.arm_idx))
            if self.grip_idx:
                g = _grip_to_finger_angle(grip[i])
                p.setJointMotorControlArray(self.body, self.grip_idx, p.POSITION_CONTROL,
                                            targetPositions=[g] * len(self.grip_idx),
                                            forces=[20.0] * len(self.grip_idx))

            # Default: step until the arm settles at q_des (or times out). --fixed-rate: just
            # give a fixed budget per pose.
            n_steps = 0
            timed_out = False
            while True:
                p.stepSimulation()
                if not self.direct:
                    time.sleep(self.sim_dt)
                n_steps += 1
                if self.fixed_rate:
                    if n_steps >= self.fixed_steps:
                        break
                else:
                    if np.max(np.abs(self._cur_arm_q() - q_des)) < self.settle_tol:
                        break
                    if n_steps >= self.settle_max_steps:
                        timed_out = True
                        break

            res = np.linalg.norm(self.model.fk(q_des)[0] - ee_pos[i])    # IK's own reach error
            step = np.degrees(np.max(np.abs(q_des - q)))
            near = bool(np.any((q_des - self.model.lower < 0.05) |
                               (self.model.upper - q_des < 0.05)))
            ik_err.append(res)
            steps.append(step)
            if timed_out:
                n_timeout += 1
            emit(f"{i:>6} | {res * 1000:>14.2f} {step:>14.2f} {n_steps:>12}"
                 f"{' TO' if timed_out else '':>0} {str(near):>8}")
            q = q_des                       # the last solution becomes the current seed

        ik_err, steps = np.array(ik_err), np.array(steps)
        if len(ik_err):
            emit(f"\nSummary: IK pos-err median {np.median(ik_err) * 1000:.1f} mm "
                 f"(max {ik_err.max() * 1000:.1f}) | step median {np.median(steps):.1f} deg "
                 f"(max {steps.max():.1f}) | reached {len(ik_err)}/{len(idx)} | "
                 f"skipped(unreachable) {n_skipped}/{len(idx)} | "
                 f"settle-timeouts {n_timeout}/{len(idx)}")
        else:
            emit(f"\nSummary: every target was unreachable — skipped {n_skipped}/{len(idx)} "
                 f"(check base_offset anchoring / workspace).")
