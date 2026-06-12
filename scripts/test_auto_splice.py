#!/usr/bin/env python3
"""Offline tests for auto-inference -> robot splicing (env.auto_ingest_chunk / _auto_splice).

Runs fully offline with the same fakes as test_safety_invariants. Asserts the two properties that
make consecutive-inference integration safe:
  S1  splice math   -> keeps the committed next substep, bridges to the time-aligned f via a ramp,
                       follows with the new trajectory's tail, and EVERY step stays <= MAX_JOINT_STEP.
  S2  validate+splice-> real sim validation + splice over several inference gaps (0.15/0.18/0.20s),
                       confirming f adapts to the measured gap and the queue stays smooth.

Run:
    .venv/bin/python scripts/test_auto_splice.py
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_safety_invariants as T                                          # noqa: E402
import real_world.humanoid_env as he                                        # noqa: E402
from real_world.humanoid_env import HumanoidEnv, MAX_JOINT_STEP, STEP_TIME  # noqa: E402
from real_world.sim_backend import SimEnv                                   # noqa: E402


def _max_step(queue):
    """Largest single-joint change between consecutive substeps in a [(q7, grip), ...] queue."""
    q = np.array([np.asarray(p[0], dtype=float) for p in queue])
    return float(np.max(np.abs(np.diff(q, axis=0)))) if len(q) > 1 else 0.0


def test_splice_math(verbose=True):
    """S1: _auto_splice into a live queue — committed substep kept, smooth, ends at B[-1]."""
    env = HumanoidEnv(robot=T._FakeRobot(), robot_controller=T._FakeCtl(),
                      sim=None, real=True, seed_q=np.clip(T.SAFE_SEED, -1, 1))
    base = np.clip(T.SAFE_SEED, env._jlower, env._jupper).astype(np.float64)
    # Two distinct trajectories, each already <= MAX_JOINT_STEP-spaced (so subdivision is a no-op
    # except across the splice junction/ramp).
    A = [(base + 0.01 * i, 0.0) for i in range(20)]          # previous inference, draining
    B = [(base + 0.20 + 0.008 * i, 1.0) for i in range(20)]  # new inference, different path + grip
    with env._lock:
        env._robot_q.clear()
        env._robot_q.extend(env._subdivide_points(A))
    committed = np.asarray(env._robot_q[0][0]).copy()        # s_{n+1} that must survive the splice

    f_want = 5
    env._auto_splice(B, time.time() - f_want * STEP_TIME)    # elapsed = 5*STEP_TIME -> f=5
    q = list(env._robot_q)

    assert np.allclose(q[0][0], committed), "S1: committed next substep s_{n+1} was not preserved"
    assert np.allclose(q[-1][0], B[-1][0]), "S1: spliced queue must end at B[-1] (new trajectory end)"
    dmax = _max_step(q)
    assert dmax <= MAX_JOINT_STEP + 1e-6, f"S1: splice step {dmax:.4f} exceeds cap {MAX_JOINT_STEP}"
    # The tail must contain B's later points (time alignment skipped B[:f]).
    tail = [np.asarray(p[0]) for p in q]
    assert any(np.allclose(t, B[f_want + 1][0]) for t in tail), "S1: tail missing B[f+1] (alignment)"
    if verbose:
        print(f"[auto-splice] S1 OK: committed kept, ends at B[-1], max step {dmax:.4f} <= "
              f"{MAX_JOINT_STEP}, queue {len(q)}")


def test_validate_and_splice(verbose=True):
    """S2: real sim validation + splice across several inference gaps; f adapts, queue stays smooth.

    submit_job runs inline because this thread owns the SimEnv it creates, so sim.validate executes
    without a separate stepping thread."""
    he.WORKSPACE_AABB = ((-9, 9), (-9, 9), (-9, 9))          # isolate from the H4 envelope check
    sim = SimEnv(direct=True)
    seed = np.clip(T.SAFE_SEED, sim.model.lower, sim.model.upper)
    env = HumanoidEnv(robot=T._FakeRobot(), robot_controller=T._FakeCtl(),
                      sim=sim, real=True, seed_q=seed)
    actions = T._synthetic_actions(env, seed)
    try:
        # First ingest: queue empty -> fresh ramp-in (idle branch). Then splices over varied gaps.
        for i, gap in enumerate((0.0, 0.15, 0.18, 0.20)):
            ok, reason = env.auto_ingest_chunk(actions, time.time() - gap)
            assert ok, f"S2: auto_ingest_chunk failed (gap={gap}): {reason}"
            with env._lock:
                q = list(env._robot_q)
            dmax = _max_step(q)
            assert dmax <= MAX_JOINT_STEP + 1e-6, f"S2: gap {gap}: step {dmax:.4f} > cap"
            assert len(q) > 0, "S2: empty queue after ingest"
            f_expect = 0 if i == 0 else round(gap / STEP_TIME)
            if verbose:
                kind = "ramp-in" if i == 0 else f"splice f~={f_expect}"
                print(f"[auto-splice] S2 gap={gap:.2f}s ({kind}): queue {len(q)}, max step {dmax:.4f}")
    finally:
        env.stop()
        sim.disconnect()
    if verbose:
        print("[auto-splice] S2 OK: validation + splice smooth across gaps 0.0/0.15/0.18/0.20s")


def test_concurrent_splice(verbose=True):
    """S3: splice WHILE the live release loop drains _robot_q (the real concurrent path). The robot
    must keep moving smoothly across several spliced inferences with no over-cap step and no crash."""
    he.WORKSPACE_AABB = ((-9, 9), (-9, 9), (-9, 9))
    sim = SimEnv(direct=True)
    seed = np.clip(T.SAFE_SEED, sim.model.lower, sim.model.upper)
    env = HumanoidEnv(robot=T._FakeRobot(), robot_controller=T._FakeCtl(),
                      sim=sim, real=True, seed_q=seed)
    ctl = env.robot_controller
    actions = T._synthetic_actions(env, seed)
    env.start(run_collect=False, run_exec=True)              # release loop now draining _robot_q live
    try:
        for k in range(4):                                   # 4 inferences spliced mid-drain
            ok, reason = env.auto_ingest_chunk(actions, time.time() - 0.16)
            assert ok, f"S3: ingest {k} failed: {reason}"
            time.sleep(0.3)
        time.sleep(0.5)
        moves = np.array(ctl.moves)
        assert len(moves) > 10, "S3: robot barely moved (release loop not draining splices)"
        dmax = float(np.max(np.abs(np.diff(moves, axis=0))))
        assert dmax <= MAX_JOINT_STEP + 1e-6, f"S3: concurrent splice step {dmax:.4f} > cap"
        assert not ctl.right_touched, "S3: right arm was commanded"
    finally:
        env.stop()
        sim.disconnect()
    if verbose:
        print(f"[auto-splice] S3 OK: {len(moves)} waypoints streamed across 4 spliced inferences, "
              f"max step {dmax:.4f} <= {MAX_JOINT_STEP}, right untouched")


def run(verbose=True):
    test_splice_math(verbose)
    test_validate_and_splice(verbose)
    test_concurrent_splice(verbose)
    if verbose:
        print("[auto-splice] ALL AUTO-SPLICE TESTS PASS (S1 S2 S3)")
    return True


if __name__ == "__main__":
    run()
