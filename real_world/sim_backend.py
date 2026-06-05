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

LEFT_GRIPPER_JOINTS = ([f"left_narrow{i}_joint" for i in (1, 2, 3, 4)] +
                       [f"left_wide{i}_joint" for i in (1, 2, 3, 4)])
GRIPPER_CLOSE_RAD = 0.6   # rough scalar grip[0,1] -> joint angle (uncalibrated)


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
        try:
            self.body = p.loadURDF(tmp_urdf, useFixedBase=True)
        finally:
            os.remove(tmp_urdf)
        jmap = {p.getJointInfo(self.body, i)[1].decode(): i
                for i in range(p.getNumJoints(self.body))}
        self.arm_idx = [jmap[j] for j in LEFT_ARM_JOINTS]
        self.grip_idx = [jmap[j] for j in LEFT_GRIPPER_JOINTS if j in jmap]

        # --- stepping / settling ---
        self.sim_hz = sim_hz
        self.sim_dt = 1.0 / sim_hz
        p.setTimeStep(self.sim_dt)
        self.settle_tol = np.radians(settle_tol_deg)
        self.settle_tol_deg = settle_tol_deg
        self.settle_max_steps = max(1, int(settle_timeout_s * sim_hz))
        self.fixed_steps = max(1, int(sim_hz / playback_hz))

        # --- live-drive handoff: latest (q7, grip) target, written by any thread ---
        self._target_lock = threading.Lock()
        self._target = None        # (np.ndarray(7,), float grip) or None

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
        """Snap the arm to a seed config (owning thread; start of a run)."""
        q = np.clip(np.asarray(q7, dtype=np.float64), self.model.lower, self.model.upper)
        for k, qi in zip(self.arm_idx, q):
            p.resetJointState(self.body, k, float(qi))

    def step(self):
        """Apply the latest commanded target (if any) and advance one sim step.

        Owning thread only. GUI mode sleeps one dt so playback runs near real time.
        """
        with self._target_lock:
            target = self._target
        if target is not None:
            q7, grip = target
            p.setJointMotorControlArray(self.body, self.arm_idx, p.POSITION_CONTROL,
                                        targetPositions=q7.tolist(),
                                        forces=[200.0] * len(self.arm_idx))
            if self.grip_idx:
                g = float(np.clip(grip, 0, 1)) * GRIPPER_CLOSE_RAD
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
        if self.direct:
            time.sleep(0.02)
        else:
            p.stepSimulation()
            time.sleep(self.sim_dt)

    def _cur_arm_q(self):
        return np.array([p.getJointState(self.body, k)[0] for k in self.arm_idx])

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
                g = float(np.clip(grip[i], 0, 1)) * GRIPPER_CLOSE_RAD
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
