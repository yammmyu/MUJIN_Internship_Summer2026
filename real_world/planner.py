"""Chunk IK-solve + sim-validation for HumanoidEnv (the "plan a chunk" step).

Extracted from HumanoidEnv. A ChunkPlanner turns a policy action chunk into a sim-validated
joint trajectory in two stages:

  1. ``solve_chunk_ik`` — pure numeric IK on the CALLER's thread: per-row workspace check (H4),
     orientation smoothing (H3), dual-arm IK with a nominal-posture fallback for contorted/parked
     arms, chaining each row's warm-start from the previous solution. Produces raw
     ``[(q14, [gl, gr]), ...]`` (no sim).
  2. ``validate_chunk`` — run those configs through the sim (substep + self-collision + joint
     readback) and return the sim-ACHIEVED ``[(q14, grip), ...]``.

The env keeps ownership of the solvers (both arms), the sim, and the live-tunable K; the planner
holds only the FALLBACK nominal posture and references the solvers it is handed. Workspace
envelopes + QUAT_ALPHA live here as module globals; the safety suite widens WORKSPACE_AABB to
isolate the release-pipeline invariants from the H4 envelope check.
"""

import json
import pathlib

import numpy as np

from real_world.ik import IKQuery, smooth_quat_step, decode_action_row_dual
from real_world.timing import MAX_JOINT_STEP


# Orientation EMA factor toward the new target quaternion (H3; 0..1, 1 = no smoothing). The ONLY
# orientation smoother on the robot path (applied row-to-row in solve_chunk_ik). 1.0 -> no
# smoothing (snappiest, roughest); LOWER -> smoother orientation but more lag.
QUAT_ALPHA = 0.5
# Workspace envelope (firmware EE frame, metres) the LEFT policy target EE pos must lie in. A
# target outside is rejected (never sent to IK/robot). Generous box; tighten per workspace. (H4)
WORKSPACE_AABB = ((-0.20, 0.85), (-0.20, 1.10), (0.40, 1.30))   # (x_lo,x_hi),(y..),(z..) LEFT arm
# RIGHT-arm envelope (H4): the right arm lives at NEGATIVE y (recorded right EE y ~[-0.38, -0.17]),
# so its AABB is the LEFT box mirrored across y=0 (same x, z). Per-arm envelopes keep the bound
# tight for each side instead of one loose union box.
WORKSPACE_AABB_RIGHT = ((-0.20, 0.85), (-1.10, 0.20), (0.40, 1.30))


def pos_in_workspace(pos, side="left"):
    """True if the target EE position is inside that arm's workspace AABB (H4). side="right" uses
    the mirrored right-arm envelope. Reads the module globals at call time so the safety suite can
    widen them to isolate the release-pipeline invariants."""
    (xl, xh), (yl, yh), (zl, zh) = WORKSPACE_AABB_RIGHT if side == "right" else WORKSPACE_AABB
    x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
    return xl <= x <= xh and yl <= y <= yh and zl <= z <= zh


def load_nominal_config():
    """Nominal per-arm training posture (real_world/nominal_arm_config.json) as a 14-vec
    [left7, right7] — the IK FALLBACK seed. Zeros if the file is absent (fallback disabled-ish)."""
    path = pathlib.Path(__file__).parent / "nominal_arm_config.json"
    try:
        d = json.loads(path.read_text())
        return np.asarray(list(d["left"]) + list(d["right"]), dtype=np.float64)
    except Exception as e:
        print(f"[ChunkPlanner] nominal_arm_config.json unavailable ({e}); IK fallback seed = zeros.")
        return np.zeros(14, dtype=np.float64)


def _limit_pinned(solver, q, margin=0.10):
    """True if any joint of q sits within `margin` rad of its limit — the signature of a
    CONTORTED IK solution (the redundant DOF shoved into a limit to reach the target)."""
    lo = np.asarray(solver.m.lower, dtype=np.float64)
    hi = np.asarray(solver.m.upper, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    return float(np.min(np.minimum(q - lo, hi - q))) < margin


class ChunkPlanner:
    """Dual-arm chunk IK + sim validation. References the env's LEFT/RIGHT solvers; owns only the
    nominal-posture IK fallback."""

    def __init__(self, solver, solver_r, nominal_q14=None):
        self.solver = solver                       # LEFT-arm IK solver
        self.solver_r = solver_r                   # RIGHT-arm IK solver
        # Nominal per-arm joint posture (median of the training recordings): the IK FALLBACK
        # warm-start when the live/chained seed resolves to a contorted or unreachable config.
        self.nominal_q14 = (load_nominal_config() if nominal_q14 is None
                            else np.asarray(nominal_q14, dtype=np.float64))

    def _ik_robust(self, solver, pos, quat, seed, nominal):
        """Solve IK from the live/chained `seed`; if that lands UNREACHABLE or LIMIT-PINNED
        (contorted), retry from `nominal` (the training posture) and prefer a clean solve. Returns
        (q7, reachable). Same rule for both arms: a warm seed keeps its natural branch (no retry, no
        regression); a parked / out-of-distribution seed escapes contortion. This is what makes the
        robot's PARKED right arm resolve to a sane pose instead of 'going crazy' — the sim eval never
        saw it because it seeds from the recorded (warm) joints."""
        q1 = solver.solve(IKQuery(target_pos=pos, target_quat=quat, current_joints=seed))
        if solver.last_reachable and not _limit_pinned(solver, q1):
            return np.asarray(q1, dtype=np.float64), True          # live seed is clean -> keep it
        r1 = solver.last_reachable
        q2 = solver.solve(IKQuery(target_pos=pos, target_quat=quat, current_joints=nominal))
        if solver.last_reachable and not _limit_pinned(solver, q2):
            return np.asarray(q2, dtype=np.float64), True          # nominal fallback is clean
        if solver.last_reachable:                                  # neither clean; prefer reachable
            return np.asarray(q2, dtype=np.float64), True
        if r1:
            return np.asarray(q1, dtype=np.float64), True
        return np.asarray(q2, dtype=np.float64), False             # genuinely unreachable

    def solve_chunk_ik(self, action_chunk, seed_q, skip_unreachable=False):
        """Solve IK for an ENTIRE chunk up front, OUTSIDE the sim validation job: workspace
        check (H4) + orientation smoothing (H3) + our IK, chaining each row's warm-start from
        the previous raw IK solution starting at seed_q (fresh quat-smoothing state per call).
        Returns (configs, ok, reason) where configs = [(q14, [gl,gr]), ...] are the raw IK joints
        the sim then validates kinematically. Runs pure-numerical Pinocchio IK on the caller's
        thread while the sim job only does the substep + self-collision check on solved configs.
        skip_unreachable=True (calibrate path): drop unreachable/out-of-envelope rows and keep
        going (mirrors the old learn=True behaviour) instead of aborting."""
        configs = []
        seed = np.asarray(seed_q, dtype=np.float64).copy()   # 14-vec [left7, right7]
        seedL, seedR = seed[:7].copy(), seed[7:14].copy()
        qprevL = qprevR = None    # per-arm quat-smoothing state (don't disturb the preview's)
        for k, action in enumerate(action_chunk):
            Lpos, Lquat, Lgrip, Rpos, Rquat, Rgrip = decode_action_row_dual(action)
            Lpos = np.asarray(Lpos, dtype=np.float64); Rpos = np.asarray(Rpos, dtype=np.float64)
            if not pos_in_workspace(Lpos, "left") or not pos_in_workspace(Rpos, "right"):
                if skip_unreachable:
                    continue
                return [], False, f"action {k}: target EE pos outside workspace envelope"
            qprevL = smooth_quat_step(qprevL, Lquat, QUAT_ALPHA)                  # H3 (per arm)
            qprevR = smooth_quat_step(qprevR, Rquat, QUAT_ALPHA)
            # live/chained seed first, nominal-training-posture fallback on a contorted/unreachable
            # solve (per arm) — keeps a warm arm's natural branch, un-contorts a parked one.
            qL, okL = self._ik_robust(self.solver, Lpos, qprevL, seedL, self.nominal_q14[:7])
            if not okL:
                if skip_unreachable:
                    continue
                return [], False, (f"action {k}: LEFT IK unreachable "
                                   f"(pos err {self.solver.last_pos_err*1000:.0f} mm)")
            qR, okR = self._ik_robust(self.solver_r, Rpos, qprevR, seedR, self.nominal_q14[7:])
            if not okR:
                if skip_unreachable:
                    continue
                return [], False, (f"action {k}: RIGHT IK unreachable "
                                   f"(pos err {self.solver_r.last_pos_err*1000:.0f} mm)")
            configs.append((np.concatenate([qL, qR]), [Lgrip, Rgrip]))
            seedL, seedR = qL, qR
        return configs, True, None

    def validate_chunk(self, sim, configs, seed_q, substeps_per_row, fast=False):
        """Run PRE-SOLVED joint configs through `sim` from seed_q (substep + self-collision +
        joint readback) and return (traj, ok, reason) where traj is the sim-ACHIEVED
        [(q14, grip), ...]. The single validation primitive shared by the manual
        validate_and_stage and auto-inference paths. Assumes `sim` is not None and `configs`
        non-empty (the env wrapper checks both).
        fast=True (auto path): skip the per-substep real-time sleep (operator-preview only).
        substeps_per_row: the SAME K the caller uses for master-id tagging, so a live speed_scale
        change between expansion and tagging can't desync them."""
        return sim.validate(configs, seed_q, MAX_JOINT_STEP, fast=fast,
                            substeps_per_row=substeps_per_row)
