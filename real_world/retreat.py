"""Torch-free 'retreat to home' primitive shared by grasp-failure recovery (grasp_recovery.py) and
the unreachable-target handler (InferenceController). Kept OUT of grasp_recovery.py so a caller that
doesn't use the torch-dependent grasp detector can still trigger a retreat without importing torch.
"""

import logging

import numpy as np

from real_world.timing import CONTROL_HZ, MAX_JOINT_VEL

log = logging.getLogger(__name__)

# Retreat cruise speed as a fraction of the MAX_JOINT_VEL safety ceiling. The per-substep joint delta
# handed to move_to_joints is (frac * MAX_JOINT_VEL / CONTROL_HZ), so the resulting cruise is exactly
# frac * MAX_JOINT_VEL rad/s -- independent of move_to_joints' own default derate.
# Kept deliberately SLOW (0.15 * 4.0 = 0.6 rad/s): a recovery/unreachable retreat should be an
# unhurried, legible back-off, not a fast snap. Lower this to slow it further; it only sets the
# retreat cruise, never the safety cap (MAX_JOINT_STEP still bounds every dispatched substep).
RETREAT_JOINT_VEL_FRAC = 0.15
RETREAT_JOINT_STEP = RETREAT_JOINT_VEL_FRAC * MAX_JOINT_VEL / CONTROL_HZ


def nearest_waypoint_not_ahead(q14, waypoints):
    """Index of the retreat waypoint to fall back to: the joint-space NEAREST one that is NOT AHEAD
    of the arm's current approach phase, so a retreat never advances the arm toward the grasp.

    `waypoints` is an (n, 14) array ordered from the average start (row 0) to the average pre-grasp
    pose (last row) — i.e. row index == approach phase. We locate `q14` along that ordered polyline
    by projecting it onto each segment (the closest projection gives a continuous phase in
    [0, n-1]), then pick the closest waypoint whose index does not exceed that phase. An arm at or
    beyond the last waypoint is eligible for all of them and retreats the least; an early arm is
    limited to the first few and retreats more. Robust to an off-trajectory pose (projection still
    yields the nearest segment). Returns 0 when a single waypoint is given."""
    q = np.asarray(q14, dtype=np.float64)
    W = np.asarray(waypoints, dtype=np.float64)
    n = len(W)
    if n <= 1:
        return 0
    # 1) continuous approach phase of q = i + t at the closest point on segment (W[i] -> W[i+1]).
    best_d, phase = np.inf, 0.0
    for i in range(n - 1):
        ab = W[i + 1] - W[i]
        denom = float(ab @ ab)
        t = 0.0 if denom == 0.0 else float(np.clip((q - W[i]) @ ab / denom, 0.0, 1.0))
        d = float(np.linalg.norm(q - (W[i] + t * ab)))
        if d < best_d:
            best_d, phase = d, i + t
    # 2) eligible = waypoints not ahead of that phase (small epsilon so one the arm sits AT counts).
    eligible = [i for i in range(n) if i <= phase + 1e-6] or [0]
    # 3) of those, the joint-space closest to q.
    return min(eligible, key=lambda i: float(np.linalg.norm(q - W[i])))


def retreat_to_nearest(env, waypoints, *, open_grip=None, joint_step=RETREAT_JOINT_STEP):
    """Retreat to the nearest approach waypoint that isn't ahead of the arm's current phase, instead
    of always homing to the fixed start. `waypoints` is an (n, 14) array [left7, right7] per row,
    ordered start -> pre-grasp (see real_world.postprocess.load_retreat_waypoints). Reads the live
    arm pose to choose (nearest_waypoint_not_ahead), then defers to retreat_to_home for the actual
    clear-queue + open-grip + velocity-bounded move. Falls back to the first waypoint if the pose
    can't be read. Returns retreat_to_home's result (False if no move could be issued)."""
    W = np.asarray(waypoints, dtype=np.float64).reshape(-1, 14)
    q = env._read_arm14() if hasattr(env, "_read_arm14") else None
    if q is None:
        idx = 0
        log.warning("[retreat] no live arm read -> retreating to waypoint 1/%d", len(W))
    else:
        idx = nearest_waypoint_not_ahead(q, W)
        log.info("[retreat] nearest-not-ahead -> waypoint %d/%d", idx + 1, len(W))
    return retreat_to_home(env, W[idx], open_grip=open_grip, joint_step=joint_step)


def retreat_to_home(env, home_q14, *, open_grip=None, joint_step=RETREAT_JOINT_STEP):
    """Shared 'let go + reset to home' retreat, with NO detector coupling: clear the robot queue,
    (optionally) open the right gripper, and move BOTH arms to the fixed HOME joint pose by absolute
    joint angles (velocity-bounded, E-stop-aware; no EE-space lift / IK / streaming). Blocks until the
    arm arrives; the policy re-approaches from home afterwards. Used by BOTH grasp-failure recovery
    (open_grip set) and the unreachable-target handler in InferenceController.
    Returns True if the home move was issued, False if no home pose / move_to_joints was available."""
    with env._lock:                                    # clear queue (non-estop half of lock_robot)
        env._robot_q.clear()
        env._staged_release.clear()
        env._queued_through = -1                       # next append re-anchors to the clock
    # Drop any anti-regrab close-latch so a latched-closed grasp can't override an intended open, and
    # so the post-retreat re-approach starts from a clean latch.
    if hasattr(env, "reset_grip_latch"):
        env.reset_grip_latch()
    # The home move bypasses pipeline.merge(), so clear the smoothed buffer -> the FIRST post-retreat
    # merge re-anchors from the live clock instead of materialising a stale pre-retreat run.
    env.pipeline.reset_merge()
    # Open the right gripper (left untouched) when asked; command_gripper also updates the obs source
    # so the policy sees "open" promptly.
    if open_grip is not None and hasattr(env, "command_gripper"):
        env.command_gripper(gr=float(open_grip))
    if home_q14 is not None and hasattr(env, "move_to_joints"):
        env.move_to_joints(np.asarray(home_q14, dtype=float), joint_step=joint_step)
        return True
    log.warning("[retreat] no home pose / move_to_joints unavailable -> gripper only")
    return False
