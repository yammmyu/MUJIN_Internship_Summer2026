#!/usr/bin/env python3
"""Safety-invariant pre-flight for the diffusion-policy -> G1 release pipeline.

Runs fully offline with fakes (no robot, no policy server, headless PyBullet) and asserts the
non-negotiable properties that gate hardware motion. A failure means a safety regression and
MUST block deployment:

  C1  no sim running       -> nothing can ever be released to the robot
  C2  validation           -> steps the sim, self-collision-checks, records sim-ACHIEVED joints
  C5  rate limit / ramp    -> released steps are <= MAX_JOINT_STEP, ramped from the current pose
  H1  one-shot release     -> a validation can be released once (no re-release snap-back)
  C3  E-stop               -> latched, actively holds, refuses release until reset
  C4  right-arm hold       -> a failed joint read REFUSES move_arm (never zeros the right arm)

Self-contained: targets are the FK of a small joint sweep near a safe seed, so they are
reachable and self-consistent without needing a recording.

Run standalone:
    .venv/bin/python scripts/test_safety_invariants.py
It also runs automatically as a launch pre-flight in robot_control_gui.py; a failure blocks
the GUI from starting. Set HUMANOID_SKIP_SAFETY_PREFLIGHT=1 to bypass (logs a loud warning).
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# A safe, clearly-away-from-torso left-arm seed (same as the sim replay runner's seed).
SAFE_SEED = np.array([0.0, -0.3, 0.0, -0.6, 0.0, 0.3, 0.0])


def _synthetic_actions(env, seed, n=24):
    """Reachable action rows [pos(3), 6D_rot(6), grip(1)] from FK of a small sweep near seed."""
    import pinocchio as pin
    m = env.solver.m
    lo, hi = m.lower, m.upper
    rows = []
    for t in range(n):
        q = np.clip(seed + 0.12 * np.sin(2 * np.pi * t / n) * np.ones(7), lo, hi)
        pos, quat = m.fk(q)                       # EE pose in the firmware frame
        R = pin.Quaternion(np.asarray(quat, dtype=float)).matrix()
        rows.append(list(pos) + list(R[:, 0]) + list(R[:, 1]) + [0.0])
    return rows


class _FakeRobot:
    def __init__(self):
        self.moves = []
        self.fail = False

    def arm_joint_states(self):
        if self.fail:
            raise RuntimeError("simulated frozen joint read")
        return ([0.0] * 14, 0)

    def gripper_states(self):
        return ([0.0, 0.0], 0)

    def move_arm(self, p):
        self.moves.append(np.asarray(p, dtype=float))

    def move_gripper(self, p):
        pass


class _FakeCtl:
    def __init__(self):
        self.moves = []          # recorded 14-joint waypoints (left7 + right7)

    def get_motion_status(self):
        return {'error': {'has_error': False}, 'collisions': []}

    def trajectory_tracking_control(self, infer_timestamp, robot_states, robot_actions,
                                    robot_link="base_link", trajectory_reference_time=1.0):
        for a in robot_actions:
            left = np.asarray(a["left_arm"]["action_data"], dtype=float)
            right = np.asarray(a["right_arm"]["action_data"], dtype=float)
            self.moves.append(np.concatenate([left, right]))


def run(verbose=True):
    """Run all invariant checks. Returns True on success, raises AssertionError on a violation."""
    import real_world.humanoid_env as he
    from real_world.humanoid_env import HumanoidEnv, MAX_JOINT_STEP
    from real_world.sim_backend import SimEnv

    # Isolate the RELEASE-pipeline invariants from the workspace-envelope check (H4 is config,
    # tested separately): widen the envelope so synthetic FK targets aren't rejected by it.
    he.WORKSPACE_AABB = ((-9, 9), (-9, 9), (-9, 9))

    sim = SimEnv(direct=True)
    seed = np.clip(SAFE_SEED, sim.model.lower, sim.model.upper)
    env = HumanoidEnv(robot=_FakeRobot(), robot_controller=_FakeCtl(), sim=sim,
                      real=True, seed_q=seed)
    robot = env.robot
    ctl = env.robot_controller          # records the commanded waypoints (trajectory_tracking_control)
    actions = _synthetic_actions(env, seed)
    env.start(run_collect=False, run_exec=True)
    try:
        # C1 — no sim => nothing releasable
        e0 = HumanoidEnv(robot=_FakeRobot(), robot_controller=_FakeCtl(), sim=None,
                         real=True, seed_q=seed)
        assert e0.validate_and_stage(actions)[0] is False, "C1: validate without a sim must fail"
        assert e0.release_to_robot() == 0, "C1: release without a sim must be a no-op"

        # C2 — validation stages a sim-achieved trajectory
        ok, reason = env.validate_and_stage(actions)
        assert ok and len(env._last_sim_traj) > 0, f"C2: validation of safe motion failed: {reason}"

        # release reaches the robot; C5 — every released step <= MAX_JOINT_STEP
        nr = env.release_to_robot()
        time.sleep(nr / 30.0 + 1.0)
        assert nr > 0 and len(ctl.moves) >= nr - 2, "release did not reach the robot"
        dmax = float(np.max(np.abs(np.diff(np.array(ctl.moves)[:, :7], axis=0))))
        assert dmax <= MAX_JOINT_STEP + 1e-6, f"C5: step {dmax:.4f} exceeds cap {MAX_JOINT_STEP}"

        # H1 — one-shot
        assert env.release_to_robot() == 0, "H1: re-release without new validation must refuse"

        # C3 — E-stop latch + active hold + refuse + reset
        env.validate_and_stage(actions)
        env.release_to_robot()
        time.sleep(0.2)
        before = len(ctl.moves)
        env.lock_robot()
        time.sleep(0.3)
        assert env.estopped, "C3: E-stop not latched"
        assert env.robot_pending == 0, "C3: pending commands not dropped"
        assert len(ctl.moves) - before >= 1, "C3: no active-hold command issued"
        assert env.release_to_robot() == 0, "C3: release allowed while E-stopped"
        env.reset_estop()
        assert not env.estopped, "C3: reset_estop did not clear the latch"

        # C4 — a failed joint read REFUSES (never zeros the right arm)
        robot.fail = True
        env._last_good_arm14 = None
        assert env._command_left_joints(np.zeros(7), 0.0) is False, \
            "C4: must refuse arm command when the right-arm hold can't be read"
    finally:
        env.stop()
        sim.disconnect()

    if verbose:
        print("[safety] ALL INVARIANTS PASS (C1 C2 C3 C4 C5 H1)")
    return True


def main():
    try:
        run()
    except AssertionError as e:
        print(f"[safety] FAIL: {e}")
        return 1
    except Exception as e:                       # import/sim error: treat as a failed pre-flight
        print(f"[safety] ERROR: {type(e).__name__}: {e}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
