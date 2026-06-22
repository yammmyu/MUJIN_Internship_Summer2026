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
  C4  right arm untouched  -> the right arm is NEVER placed in a trajectory action (so it holds)

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
        self.moves = []          # recorded LEFT-arm waypoints (7,)
        self.right_touched = False   # set if any action ever addresses the right arm (C4)

    def get_motion_status(self):
        # Include 'frames' so the collect loop's _left_ee_from() works when run_collect=True
        # (the release path's firmware-safety flag is cached from this read; see _firmware_unsafe).
        pose = {'position': {'x': 0.0, 'y': 0.0, 'z': 0.0},
                'orientation': {'quaternion': {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0}}}
        return {'error': {'has_error': False}, 'collisions': [],
                'frames': {'arm_left_link7': pose, 'arm_right_link7': pose}}

    def trajectory_tracking_control(self, infer_timestamp, robot_states, robot_actions,
                                    robot_link="base_link", trajectory_reference_time=1.0):
        for a in robot_actions:
            if "right_arm" in a:
                self.right_touched = True
            self.moves.append(np.asarray(a["left_arm"]["action_data"], dtype=float))


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
    # Run collect AND exec: the release path's _firmware_unsafe() reads the firmware-safety flag
    # the collect loop caches (30Hz), so release without collect would fail-closed (E-stop). This
    # mirrors production, where start() always runs both.
    env.start(run_collect=True, run_exec=True)
    time.sleep(0.1)            # let the first collect tick populate the firmware-status cache
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
        # The release loop drains _robot_q one substep per STEP_TIME (slower than 30Hz, and
        # subdivision adds points), so poll until the queue is empty rather than guess a duration.
        t0 = time.time()
        while time.time() - t0 < 20.0:
            with env._lock:
                drained = not env._robot_q
            if drained:
                break
            time.sleep(0.05)
        time.sleep(0.2)        # let the final in-flight substep's STEP_TIME wait complete
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

        # C4 — the right arm is NEVER commanded (no right_arm in any trajectory action), so it
        # physically holds. With move_arm gone there is no right-arm zeroing risk to guard; the
        # invariant is now "we never address the right arm" across every release + the E-stop hold.
        assert not ctl.right_touched, "C4: right arm was placed in a trajectory action"
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
