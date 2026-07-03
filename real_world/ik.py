"""Swappable, transparent inverse kinematics for the G1 left arm.

This replaces the SDK's firmware blackbox (`set_end_effector_pose_control`) with our
own URDF-based IK: given a target EE pose + the current joint angles, compute the 7
left-arm joint targets we then command directly via `move_arm`. The same solver runs
in the PyBullet sim check and (later) on the real robot.

SDK-free on purpose: depends only on numpy + pinocchio + the URDF, so it imports and
runs without `a2d_sdk` (needed by the offline FK probe and the sim runner).

Design
------
- `IKQuery`     : the per-call inputs (target pose, current joints, current EE pose).
- `IKSolver`    : abstract base — the swap point for alternative IK methods.
- `PinocchioArmModel` : builds a *reduced* pinocchio model with only the 7 left-arm
  joints free (everything else locked), so FK/Jacobian are a clean 7-DOF function of
  the arm joints expressed in the URDF `base_link` frame. A constant `base_offset` SE3
  maps `base_link` -> the firmware reference frame (calibrated offline; absorbs the
  fixed waist/lift pose and any frame mismatch).
- `PinocchioDLSIKSolver` : damped-least-squares solver (the canonical pinocchio loop),
  seeded from the current joints, clamped to URDF limits.

All quaternions at this module's API boundary are **xyzw** (matching the SDK/policy).
"""

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pinocchio as pin

# Default robot model (the URDF the SDK ships; FK calibrated against firmware offline).
# Vendored under real_world/assets/ so IK is self-contained and works from a fresh clone —
# the URDF uses package://meshes/ refs and ships its own meshes/ alongside (sim_backend's
# patched_urdf strips package:// and loads them relative to this file's dir).
DEFAULT_URDF = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "assets", "A2D_Omnipicker", "A2D.urdf",
)
# FK calibration + nominal-posture config live under real_world/config/.
_CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config")
DEFAULT_CALIBRATION = os.path.join(_CONFIG_DIR, "fk_calibration.json")
# Right-arm FK calibration (mirror of the left, produced by scripts/fk_consistency_check.py --side
# right). Separate file so each arm carries its own base_offset / ee_frame / joint-slice.
DEFAULT_CALIBRATION_RIGHT = os.path.join(_CONFIG_DIR, "fk_calibration_right.json")

LEFT_ARM_JOINTS = ["Joint1_l", "Joint2_l", "Joint3_l", "Joint4_l",
                   "Joint5_l", "Joint6_l", "Joint7_l"]
RIGHT_ARM_JOINTS = ["Joint1_r", "Joint2_r", "Joint3_r", "Joint4_r",
                    "Joint5_r", "Joint6_r", "Joint7_r"]

# Per-side defaults: URDF joint names, EE frame, and calibration file. build_solver(side=...) and
# the sim/env read from here so "which arm" is defined in exactly one place.
ARM_SIDES = {
    "left":  {"joints": LEFT_ARM_JOINTS,  "ee_frame": "Link7_l", "calibration": DEFAULT_CALIBRATION},
    "right": {"joints": RIGHT_ARM_JOINTS, "ee_frame": "Link7_r", "calibration": DEFAULT_CALIBRATION_RIGHT},
}


# --------------------------------------------------------------------------- #
#  quaternion / SE3 helpers (xyzw convention at the boundary)                  #
# --------------------------------------------------------------------------- #
def quat_xyzw_to_se3(pos, quat_xyzw) -> pin.SE3:
    q = np.asarray(quat_xyzw, dtype=np.float64)
    q = q / (np.linalg.norm(q) + 1e-12)
    rot = pin.Quaternion(q).matrix()   # pin.Quaternion takes coeffs (x, y, z, w)
    return pin.SE3(rot, np.asarray(pos, dtype=np.float64))


def se3_to_pos_quat_xyzw(M: pin.SE3):
    pos = np.asarray(M.translation, dtype=np.float64).copy()
    quat = pin.Quaternion(M.rotation)
    quat.normalize()
    return pos, np.asarray(quat.coeffs(), dtype=np.float64)   # coeffs() is (x, y, z, w)


def rot6d_to_quat(rot6d):
    """6D rotation (first two rotation-matrix columns) -> quaternion [x, y, z, w].

    Identical math to build_left_arm_ee_replay_buffer / humanoid_env.rot6d_to_quat, kept
    here so the sim runner can decode policy actions without importing the SDK-coupled
    humanoid_env module.
    """
    rot6d = np.asarray(rot6d, dtype=np.float64)
    c1, c2 = rot6d[:3], rot6d[3:6]
    c1 = c1 / (np.linalg.norm(c1) + 1e-8)
    c2 = c2 - np.dot(c2, c1) * c1
    c2 = c2 / (np.linalg.norm(c2) + 1e-8)
    c3 = np.cross(c1, c2)
    m = np.stack([c1, c2, c3], axis=-1)
    return np.asarray(se3_to_pos_quat_xyzw(pin.SE3(m, np.zeros(3)))[1], dtype=np.float64)


def decode_action_row(row):
    """LEFT half of a policy action row [eef_pos(3), 6D_rot(6), gripper(1)] ->
    (pos(3), quat_xyzw(4), grip). Reads cols 0:10, so it works on both a 10-col left row and a
    20-col dual row (the sim-preview path decodes only the left arm)."""
    row = np.asarray(row, dtype=np.float64)
    return row[:3], rot6d_to_quat(row[3:9]), float(row[9])


def decode_action_row_dual(row):
    """Dual 20-col action row L[pos3,rot6d6,grip1] ++ R[pos3,rot6d6,grip1] ->
    (Lpos, Lquat, Lgrip, Rpos, Rquat, Rgrip). Left = cols 0:10, right = cols 10:20."""
    a = np.asarray(row, dtype=np.float64)
    return (a[0:3],   rot6d_to_quat(a[3:9]),   float(a[9]),
            a[10:13], rot6d_to_quat(a[13:19]), float(a[19]))


def slerp(q0, q1, t):
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


def smooth_quat_step(prev, quat, alpha):
    """One step of orientation smoothing (H3): SLERP the previous target toward the new quat
    by `alpha` (1.0 = no smoothing). `prev` None -> pass the new quat through unchanged."""
    q = np.asarray(quat, dtype=np.float64)
    q = q / (np.linalg.norm(q) + 1e-12)
    if prev is None:
        return q
    return slerp(prev, q, alpha)


# --------------------------------------------------------------------------- #
#  IK interface                                                               #
# --------------------------------------------------------------------------- #
@dataclass
class IKQuery:
    """Per-call IK inputs. Quaternions are xyzw. current_ee_* are optional (for
    diagnostics / delta methods); the Pinocchio solver only needs target + seed."""
    target_pos: np.ndarray          # (3,)
    target_quat: np.ndarray         # (4,) xyzw
    current_joints: np.ndarray      # (7,) seed, left-arm order Joint1_l..Joint7_l
    current_ee_pos: Optional[np.ndarray] = None
    current_ee_quat: Optional[np.ndarray] = None


class IKSolver(ABC):
    """Swap point: turn an IKQuery into 7 left-arm joint targets."""

    @abstractmethod
    def solve(self, q: IKQuery) -> np.ndarray:  # -> (7,)
        ...


# --------------------------------------------------------------------------- #
#  Pinocchio reduced-model FK                                                 #
# --------------------------------------------------------------------------- #
class PinocchioArmModel:
    """Reduced 7-DOF model of the left arm built from the URDF.

    FK is expressed in the URDF `base_link` frame; `base_offset` (SE3) maps that to
    the firmware reference frame in which the policy/SDK poses live.
    """

    def __init__(self,
                 urdf_path: str = DEFAULT_URDF,
                 ee_frame: str = "Link7_l",
                 arm_joints: Sequence[str] = LEFT_ARM_JOINTS,
                 base_offset: Optional[pin.SE3] = None):
        self.urdf_path = urdf_path
        self.ee_frame = ee_frame
        self.arm_joints = list(arm_joints)

        full = pin.buildModelFromUrdf(urdf_path)
        # Lock every joint except the left-arm joints, at the neutral (0) config.
        keep_ids = {full.getJointId(n) for n in self.arm_joints}
        lock_ids = [jid for jid in range(1, full.njoints) if jid not in keep_ids]
        self.model = pin.buildReducedModel(full, lock_ids, pin.neutral(full))
        self.data = self.model.createData()

        if not self.model.existFrame(ee_frame):
            raise ValueError(f"EE frame {ee_frame!r} not in model")
        self.ee_fid = self.model.getFrameId(ee_frame)

        # Map our canonical [Joint1_l..Joint7_l] order onto the reduced config indices.
        self._qpos_index = []
        for n in self.arm_joints:
            jid = self.model.getJointId(n)
            self._qpos_index.append(self.model.joints[jid].idx_q)
        self._qpos_index = np.asarray(self._qpos_index, dtype=int)

        self.lower = self.model.lowerPositionLimit[self._qpos_index].copy()
        self.upper = self.model.upperPositionLimit[self._qpos_index].copy()
        self.base_offset = base_offset if base_offset is not None else pin.SE3.Identity()

    # -- configuration plumbing -------------------------------------------- #
    def _to_model_q(self, q7) -> np.ndarray:
        q = pin.neutral(self.model)
        q[self._qpos_index] = np.asarray(q7, dtype=np.float64)
        return q

    def fk_local(self, q7) -> pin.SE3:
        """EE pose in the URDF base_link frame (function of the 7 arm joints only)."""
        q = self._to_model_q(q7)
        pin.framesForwardKinematics(self.model, self.data, q)
        return self.data.oMf[self.ee_fid].copy()

    def fk(self, q7):
        """EE pose in the firmware frame: base_offset * fk_local. Returns (pos, quat_xyzw)."""
        return se3_to_pos_quat_xyzw(self.base_offset * self.fk_local(q7))

    def clip(self, q7) -> np.ndarray:
        return np.clip(np.asarray(q7, dtype=np.float64), self.lower, self.upper)


# --------------------------------------------------------------------------- #
#  Damped-least-squares IK (canonical pinocchio loop)                         #
# --------------------------------------------------------------------------- #
class PinocchioDLSIKSolver(IKSolver):
    def __init__(self,
                 model: PinocchioArmModel,
                 damping: float = 1e-2,
                 max_iters: int = 100,
                 tol: float = 1e-4,
                 step_scale: float = 1.0,
                 max_step: float = 0.2,
                 reach_pos_tol: float = 0.02,
                 single_step: bool = False,
                 clamp_to_limits: bool = True):
        self.m = model
        self.damping = damping
        self.max_iters = max_iters
        self.tol = tol
        self.step_scale = step_scale
        # Cap on the per-iteration joint step (rad, L2 over the 7 joints). Near a
        # singular/ill-conditioned Jacobian the raw DLS step can be several radians,
        # which overshoots, throws the config across joint space, and makes the loop
        # diverge (we saw EE solutions land >1 m / ~360deg off target). Clamping the
        # step turns Gauss-Newton into a safe damped descent. <=0 disables the cap.
        self.max_step = max_step
        # Position residual (m) above which the returned config is deemed to NOT reach
        # the target (target outside the workspace / a local minimum). Callers should
        # gate on `last_reachable` and skip such targets rather than command them.
        self.reach_pos_tol = reach_pos_tol
        self.single_step = single_step
        self.clamp_to_limits = clamp_to_limits

        # Diagnostics for the most recent solve() (set on every call):
        self.last_pos_err = float("inf")   # position error (m) of the returned config
        self.last_reachable = False        # last_pos_err <= reach_pos_tol

    def solve(self, q: IKQuery) -> np.ndarray:
        m = self.m
        # Target expressed in the base_link frame the reduced model lives in.
        target_world = quat_xyzw_to_se3(q.target_pos, q.target_quat)
        oMdes = m.base_offset.actInv(target_world)

        qj = np.asarray(q.current_joints, dtype=np.float64).copy()
        I6 = np.eye(6)
        damp2 = self.damping ** 2
        n_iters = 1 if self.single_step else self.max_iters

        # Track the lowest-error config seen: the iterate can wobble near the optimum
        # or, for an unreachable target, settle on the workspace boundary, so the last
        # iterate is not necessarily the best one to return.
        best_q = qj.copy()
        best_err = np.inf

        for _ in range(n_iters):
            cfg = m._to_model_q(qj)
            pin.framesForwardKinematics(m.model, m.data, cfg)
            Mf = m.data.oMf[m.ee_fid]
            iMd = Mf.actInv(oMdes)
            err = pin.log6(iMd).vector
            err_norm = np.linalg.norm(err)
            if err_norm < best_err:
                best_err, best_q = err_norm, qj.copy()
            if err_norm < self.tol:
                break
            J = pin.computeFrameJacobian(m.model, m.data, cfg, m.ee_fid)  # LOCAL, 6x7
            J = -pin.Jlog6(iMd.inverse()) @ J
            dq = self.step_scale * (-J.T @ np.linalg.solve(J @ J.T + damp2 * I6, err))
            dq_norm = np.linalg.norm(dq)
            if self.max_step > 0 and dq_norm > self.max_step:
                dq *= self.max_step / dq_norm
            qj = pin.integrate(m.model, cfg, dq)[m._qpos_index]
            if self.clamp_to_limits:
                qj = np.clip(qj, m.lower, m.upper)
        # single_step is delta/velocity control: return the one stepped config. The
        # iterative solver returns the best config found (robust to wobble/divergence).
        result = qj if self.single_step else best_q
        # Report reachability so callers can skip out-of-workspace / unconverged targets.
        self.last_pos_err = float(np.linalg.norm(
            m.fk(result)[0] - np.asarray(q.target_pos, dtype=np.float64)))
        self.last_reachable = self.last_pos_err <= self.reach_pos_tol
        return result


# --------------------------------------------------------------------------- #
#  Calibration (produced by scripts/fk_consistency_check.py)                  #
# --------------------------------------------------------------------------- #
@dataclass
class FKCalibration:
    ee_frame: str = "Link7_l"
    quat_convention: str = "xyzw"          # how recorded SDK quats are interpreted
    arm_joint_slice: str = "left_first"     # 'left_first' -> arm_joints[:7], else [7:]
    base_offset_pos: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    base_offset_quat_xyzw: list = field(default_factory=lambda: [0.0, 0.0, 0.0, 1.0])
    pos_residual_m: Optional[float] = None
    rot_residual_deg: Optional[float] = None

    def base_offset_se3(self) -> pin.SE3:
        return quat_xyzw_to_se3(self.base_offset_pos, self.base_offset_quat_xyzw)


def load_calibration(path: str = DEFAULT_CALIBRATION) -> FKCalibration:
    with open(path) as f:
        d = json.load(f)
    known = {k: d[k] for k in FKCalibration().__dict__ if k in d}
    return FKCalibration(**known)


def build_solver(calibration_path: str = None,
                 urdf_path: str = DEFAULT_URDF,
                 side: str = "left",
                 **solver_kwargs) -> "PinocchioDLSIKSolver":
    """Convenience: load the calibrated config and return a ready DLS solver for one arm.

    side ("left"/"right") selects the URDF arm joints, EE frame, and default calibration file
    (ARM_SIDES). calibration_path overrides the per-side default (e.g. a custom right-arm fit).
    The right arm needs its own fk_calibration_right.json (scripts/fk_consistency_check.py
    --side right); until it exists, build_solver(side='right') raises on the missing file."""
    spec = ARM_SIDES[side]
    cal = load_calibration(calibration_path or spec["calibration"])
    model = PinocchioArmModel(urdf_path=urdf_path, ee_frame=cal.ee_frame,
                              arm_joints=spec["joints"], base_offset=cal.base_offset_se3())
    return PinocchioDLSIKSolver(model, **solver_kwargs)
