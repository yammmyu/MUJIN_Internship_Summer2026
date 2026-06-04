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
DEFAULT_URDF = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "G1_SDK_ENV", "a2d_sdk", "A2D_Omnipicker", "A2D.urdf",
)
DEFAULT_CALIBRATION = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fk_calibration.json")

LEFT_ARM_JOINTS = ["Joint1_l", "Joint2_l", "Joint3_l", "Joint4_l",
                   "Joint5_l", "Joint6_l", "Joint7_l"]


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
                 single_step: bool = False,
                 clamp_to_limits: bool = True):
        self.m = model
        self.damping = damping
        self.max_iters = max_iters
        self.tol = tol
        self.step_scale = step_scale
        self.single_step = single_step
        self.clamp_to_limits = clamp_to_limits

    def solve(self, q: IKQuery) -> np.ndarray:
        m = self.m
        # Target expressed in the base_link frame the reduced model lives in.
        target_world = quat_xyzw_to_se3(q.target_pos, q.target_quat)
        oMdes = m.base_offset.actInv(target_world)

        qj = np.asarray(q.current_joints, dtype=np.float64).copy()
        I6 = np.eye(6)
        damp2 = self.damping ** 2
        n_iters = 1 if self.single_step else self.max_iters

        for _ in range(n_iters):
            cfg = m._to_model_q(qj)
            pin.framesForwardKinematics(m.model, m.data, cfg)
            Mf = m.data.oMf[m.ee_fid]
            iMd = Mf.actInv(oMdes)
            err = pin.log6(iMd).vector
            if np.linalg.norm(err) < self.tol:
                break
            J = pin.computeFrameJacobian(m.model, m.data, cfg, m.ee_fid)  # LOCAL, 6x7
            J = -pin.Jlog6(iMd.inverse()) @ J
            dq = -J.T @ np.linalg.solve(J @ J.T + damp2 * I6, err)
            qj = pin.integrate(m.model, cfg, self.step_scale * dq)[m._qpos_index]
            if self.clamp_to_limits:
                qj = np.clip(qj, m.lower, m.upper)
        return qj


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


def build_solver(calibration_path: str = DEFAULT_CALIBRATION,
                 urdf_path: str = DEFAULT_URDF,
                 **solver_kwargs) -> "PinocchioDLSIKSolver":
    """Convenience: load the calibrated config and return a ready DLS solver."""
    cal = load_calibration(calibration_path)
    model = PinocchioArmModel(urdf_path=urdf_path, ee_frame=cal.ee_frame,
                              base_offset=cal.base_offset_se3())
    return PinocchioDLSIKSolver(model, **solver_kwargs)
