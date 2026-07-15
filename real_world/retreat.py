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
RETREAT_JOINT_VEL_FRAC = 0.5
RETREAT_JOINT_STEP = RETREAT_JOINT_VEL_FRAC * MAX_JOINT_VEL / CONTROL_HZ


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
