#!/usr/bin/env python3
"""Safety-invariant pre-flight for the diffusion-policy -> G1 release pipeline.

Runs fully offline with fakes (no robot, no policy server, headless PyBullet) and asserts the
non-negotiable properties that gate hardware motion. A failure means a safety regression and
MUST block deployment:

  C1  no sim running       -> nothing can ever be released to the robot
  C2  validation           -> steps the sim, self-collision-checks, records sim-ACHIEVED joints
  C5  rate limit / ramp    -> released steps are <= MAX_JOINT_STEP (BOTH arms), ramped from pose
  C6  large-rotation cutoff -> a single commanded jump > WATCHDOG_MAX_JOINT_JUMP is REFUSED (E-stop
                              latched), not ramped through — the arm never swings a large rotation
  C7  EE safe region       -> a commanded EE outside the data-estimated EE_SAFE_REGION latches the
                              E-stop (the arm never leaves the box it was ever demonstrated in)
  H1  one-shot release     -> a validation can be released once (no re-release snap-back)
  C3  E-stop               -> latched, actively holds, refuses release until reset
  C4  DUAL-ARM commanded   -> BOTH arms are placed in every trajectory action and the right arm is
                              also velocity-bounded (dual-arm inference; the old "right arm never
                              moves" invariant is intentionally retired now that we drive both arms)

Self-contained: targets are the FK of a small joint sweep near a safe seed for EACH arm, so they
are reachable and self-consistent without needing a recording.

Run standalone:
    .venv/bin/python tests/test_safety_invariants.py
It also runs automatically as a launch pre-flight in robot_control_gui.py; a failure blocks
the GUI from starting. Set HUMANOID_SKIP_SAFETY_PREFLIGHT=1 to bypass (logs a loud warning).
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# A safe, clearly-away-from-torso arm seed (same as the sim replay runner's seed). Used for BOTH
# arms (clipped to each arm's own limits), so the synthetic dual-arm targets are reachable per side.
SAFE_SEED = np.array([0.0, -0.3, 0.0, -0.6, 0.0, 0.3, 0.0])


def _synthetic_actions(env, seed, n=24):
    """Reachable DUAL action rows (20-col): L[pos3,rot6d6,grip1] ++ R[pos3,rot6d6,grip1] from the FK
    of a small joint sweep near `seed` on EACH arm (clipped to that arm's limits)."""
    import pinocchio as pin

    def _arm_row(m, q):
        pos, quat = m.fk(q)                       # EE pose in the firmware frame
        R = pin.Quaternion(np.asarray(quat, dtype=float)).matrix()
        return list(pos) + list(R[:, 0]) + list(R[:, 1]) + [0.0]

    mL, mR = env.solver.m, env.solver_r.m
    rows = []
    for t in range(n):
        s = 0.12 * np.sin(2 * np.pi * t / n) * np.ones(7)
        qL = np.clip(seed + s, mL.lower, mL.upper)
        qR = np.clip(seed + s, mR.lower, mR.upper)
        rows.append(_arm_row(mL, qL) + _arm_row(mR, qR))     # 20-col dual row
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
        self.moves_r = []        # recorded RIGHT-arm waypoints (7,)
        self.right_touched = False   # set once any action addresses the right arm (dual-arm C4)
        self.left_only = False       # set if any action omits the right arm (would be a bug now)

    def get_motion_status(self):
        return {'error': {'has_error': False}, 'collisions': []}

    def trajectory_tracking_control(self, infer_timestamp, robot_states, robot_actions,
                                    robot_link="base_link", trajectory_reference_time=1.0):
        for a in robot_actions:
            if "right_arm" in a:
                self.right_touched = True
                self.moves_r.append(np.asarray(a["right_arm"]["action_data"], dtype=float))
            else:
                self.left_only = True
            self.moves.append(np.asarray(a["left_arm"]["action_data"], dtype=float))


def run(verbose=True):
    """Run all invariant checks. Returns True on success, raises AssertionError on a violation."""
    import real_world.postprocess as planner_mod
    from real_world.humanoid_env import HumanoidEnv
    from real_world.timing import MAX_JOINT_STEP, WATCHDOG_MAX_JOINT_JUMP
    from real_world.sim_backend import SimEnv

    # Isolate the RELEASE-pipeline invariants from the workspace-envelope check (H4 is config,
    # tested separately): widen BOTH arms' envelopes so synthetic FK targets aren't rejected by them.
    # The envelopes now live in real_world.postprocess (PostProcessor.solve_chunk_ik reads them).
    planner_mod.WORKSPACE_AABB = ((-9, 9), (-9, 9), (-9, 9))
    planner_mod.WORKSPACE_AABB_RIGHT = ((-9, 9), (-9, 9), (-9, 9))
    # Likewise widen the C7 EE safe region so the synthetic seed's FK (away from the real task box)
    # doesn't trip the runtime EE watchdog during the C2–C6 release checks; the C7 block below
    # narrows it back to prove the watchdog fires.
    planner_mod.EE_SAFE_REGION_LEFT = ((-9, 9), (-9, 9), (-9, 9))
    planner_mod.EE_SAFE_REGION_RIGHT = ((-9, 9), (-9, 9), (-9, 9))

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
        # C5 — every released step of BOTH arms is <= MAX_JOINT_STEP.
        dmaxL = float(np.max(np.abs(np.diff(np.array(ctl.moves)[:, :7], axis=0))))
        assert dmaxL <= MAX_JOINT_STEP + 1e-6, f"C5(L): step {dmaxL:.4f} exceeds cap {MAX_JOINT_STEP}"
        assert ctl.moves_r, "C5(R): right arm was never commanded (dual-arm release expected)"
        dmaxR = float(np.max(np.abs(np.diff(np.array(ctl.moves_r)[:, :7], axis=0))))
        assert dmaxR <= MAX_JOINT_STEP + 1e-6, f"C5(R): step {dmaxR:.4f} exceeds cap {MAX_JOINT_STEP}"

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

        # C4 (dual-arm) — BOTH arms are driven: every trajectory action addresses left_arm AND
        # right_arm together (in lockstep), and the E-stop hold holds both. The retired invariant
        # "right arm never moves" no longer applies now that dual-arm inference commands both; the
        # right arm's safety is the SAME as the left's — sim-validated (C2) + velocity-bounded (C5).
        assert ctl.right_touched, "C4: right arm was never commanded (dual-arm release expected)"
        assert not ctl.left_only, "C4: a trajectory action omitted the right arm (arms must move together)"

        # C6 (large-rotation watchdog) — a single commanded target that jumps more than
        # WATCHDOG_MAX_JOINT_JUMP from the last commanded config on any joint is REFUSED: the
        # watchdog latches the E-stop, drops the queue, and never dispatches the oversized target
        # (vs C5, which would ramp it through as a slow but large sweep). Drive it directly since
        # the whole point is a target upstream subdivision failed to bound.
        assert WATCHDOG_MAX_JOINT_JUMP > 0, "C6: watchdog disabled (WATCHDOG_MAX_JOINT_JUMP <= 0)"
        env.reset_estop()                                   # clean, un-latched state; guard re-seeds
        lower = np.asarray(env._jlower14, dtype=float)
        upper = np.asarray(env._jupper14, dtype=float)
        env._last_cmd_q14 = lower.copy()                    # seed the guard at the lower limit
        big = upper.copy()                                  # full-range jump, in-limits by construction
        assert float(np.max(np.abs(big - lower))) > WATCHDOG_MAX_JOINT_JUMP, \
            "C6: test target does not exceed the cutoff (arm range too small?)"
        n_before = len(ctl.moves)
        env.run_trajectory_control([(big, None)])           # one oversized waypoint
        assert env.estopped, "C6: watchdog did not latch the E-stop on a large joint jump"
        assert env.robot_pending == 0, "C6: watchdog did not drop the queue"
        sent = np.array(ctl.moves[n_before:]) if len(ctl.moves) > n_before else np.zeros((0, 7))
        if len(sent):                                       # only the hold pose may be sent, never `big`
            assert not np.any(np.all(np.isclose(sent, big[:7], atol=1e-6), axis=1)), \
                "C6: the over-cutoff target was dispatched despite the watchdog"
        env.reset_estop()

        # C7 (EE safe-region watchdog) — a commanded target whose FK end-effector is outside the
        # data-estimated EE_SAFE_REGION latches the E-stop. Force it with an unreachable region so
        # ANY EE is outside, and command the live pose (zeros) so the C6 joint-jump watchdog stays
        # silent and C7 is unambiguously the trip. Restore the region before asserting.
        env.reset_estop()
        saved_L = planner_mod.EE_SAFE_REGION_LEFT
        planner_mod.EE_SAFE_REGION_LEFT = ((9.0, 9.1), (9.0, 9.1), (9.0, 9.1))   # nothing inside
        n_before = len(ctl.moves)
        near_live = np.clip(np.zeros(14), env._jlower14, env._jupper14)   # ~live pose -> C6 silent
        try:
            env.run_trajectory_control([(near_live, None)])
        finally:
            planner_mod.EE_SAFE_REGION_LEFT = saved_L
        assert env.estopped, "C7: watchdog did not latch on an out-of-region EE"
        assert env.robot_pending == 0, "C7: watchdog did not drop the queue"
        env.reset_estop()
    finally:
        env.stop()
        sim.disconnect()

    if verbose:
        print("[safety] ALL INVARIANTS PASS (C1 C2 C3 C4 C5 C6 C7 H1)")
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
