#!/usr/bin/env python3
"""Offline tests for auto-inference -> robot ingest (env.auto_ingest_chunk) and the master-ID
temporal ensemble. Runs fully offline with the same fakes as test_safety_invariants.

  R1  ramp ingest (idle)  -> validate a chunk and lay a ramp-in from the live pose to its rows,
                             every step <= MAX_JOINT_STEP, row ids monotonic from obs_step_id.
  R2  master-ID drop      -> a chunk whose obs row id is BEHIND the current row id keeps only the
                             still-future rows (drops the part the arm already passed).
  R3  concurrent ingest   -> ingest WHILE the release loop drains: smooth, right arm untouched.
  E1  ensemble by id      -> rows sharing an absolute master id are averaged (recency-weighted);
                             non-overlapping rows pass through (identity); gripper re-binarised.

Run:
    .venv/bin/python scripts/test_auto_splice.py
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_safety_invariants as T                                              # noqa: E402
import real_world.humanoid_env as he                                           # noqa: E402
from real_world.humanoid_env import HumanoidEnv, MAX_JOINT_STEP, SUBSTEPS_PER_ROW  # noqa: E402
from real_world.sim_backend import SimEnv                                       # noqa: E402


def _max_step(queue):
    """Largest single-joint change between consecutive substeps in a [(q7, grip, *_), ...] queue."""
    q = np.array([np.asarray(p[0], dtype=float) for p in queue])
    return float(np.max(np.abs(np.diff(q, axis=0)))) if len(q) > 1 else 0.0


class _ClosedLoopFake:
    """Robot + controller fake whose arm_joint_states returns the LAST COMMANDED left arm (closed
    loop). The auto path re-reads the arm each ingest to anchor its ramp from the arm's ACTUAL pose;
    the real robot and sim_infer_eval's _SimRobot both satisfy that. (test_safety's _FakeRobot
    returns constant zeros — fine for the single-shot manual path, but it breaks multi-ingest auto,
    where the arm has actually moved.)"""

    def __init__(self, q0):
        self._left = np.asarray(q0, dtype=float).copy()
        self.moves = []
        self.right_touched = False

    def arm_joint_states(self):
        return (np.concatenate([self._left, np.zeros(7)]).tolist(), 0)

    def gripper_states(self):
        return ([0.0, 0.0], 0)

    def waist_joint_states(self):
        return ([0.0, 0.0], 0)

    def head_joint_states(self):
        return ([0.0, 0.0], 0)

    def move_gripper(self, positions):
        pass

    def get_motion_status(self, timestamp=None):
        return {'error': {'has_error': False}, 'collisions': []}

    def trajectory_tracking_control(self, infer_timestamp, robot_states, robot_actions,
                                    robot_link="base_link", trajectory_reference_time=1.0):
        for a in robot_actions:
            if "right_arm" in a or "right_gripper" in a:
                self.right_touched = True
            self._left = np.asarray(a["left_arm"]["action_data"], dtype=float)   # closed loop
            self.moves.append(self._left.copy())


def _env_with_sim():
    he.WORKSPACE_AABB = ((-9, 9), (-9, 9), (-9, 9))      # isolate from the H4 envelope check
    sim = SimEnv(direct=True)
    seed = np.clip(T.SAFE_SEED, sim.model.lower, sim.model.upper)
    fake = _ClosedLoopFake(seed)                         # one object as both robot + controller
    env = HumanoidEnv(robot=fake, robot_controller=fake, sim=sim, real=True, seed_q=seed)
    return env, sim, seed


def test_ramp_ingest_idle(verbose=True):
    """R1: ingest into an idle queue -> ramp from the live pose, smooth, row ids monotonic from 0."""
    env, sim, seed = _env_with_sim()
    actions = T._synthetic_actions(env, seed)
    try:
        ok, reason = env.auto_ingest_chunk(actions, obs_step_id=0)
        assert ok, f"R1: ingest failed: {reason}"
        with env._lock:
            q = list(env._robot_q)
        assert len(q) > 0, "R1: empty queue after ingest"
        dmax = _max_step(q)
        assert dmax <= MAX_JOINT_STEP + 1e-6, f"R1: step {dmax:.4f} exceeds cap {MAX_JOINT_STEP}"
        ids = [p[2] for p in q]                          # every element is (q7, grip, row_id)
        assert ids == sorted(ids), "R1: row ids not monotonic"
        assert ids[0] == 0, f"R1: first row id should be obs_step_id 0, got {ids[0]}"
    finally:
        env.stop()
        sim.disconnect()
    if verbose:
        print(f"[auto-ingest] R1 OK: idle ramp-in, queue {len(q)}, ids {ids[0]}..{ids[-1]}, "
              f"max step {dmax:.4f} <= {MAX_JOINT_STEP}")


def test_master_id_drop(verbose=True):
    """R2: chunk observed at row 0 while the arm is already on a later row keeps only future rows."""
    env, sim, seed = _env_with_sim()
    actions = T._synthetic_actions(env, seed)            # N rows
    n = len(actions)
    behind = 3                                           # arm is 3 rows ahead of the obs
    try:
        with env._lock:
            env._current_row_id = behind
        ok, reason = env.auto_ingest_chunk(actions, obs_step_id=0)
        assert ok, f"R2: ingest failed: {reason}"
        with env._lock:
            q = list(env._robot_q)
        ids = [p[2] for p in q]
        assert min(ids) == behind, f"R2: kept rows should start at current row {behind}, got {min(ids)}"
        assert max(ids) == n - 1, f"R2: should follow to the last row {n-1}, got {max(ids)}"
        assert _max_step(q) <= MAX_JOINT_STEP + 1e-6, "R2: over-cap step"
    finally:
        env.stop()
        sim.disconnect()
    if verbose:
        print(f"[auto-ingest] R2 OK: dropped rows < {behind}, kept ids {behind}..{n-1}")


def test_concurrent_ingest(verbose=True):
    """R3: ingest several chunks WHILE the release loop drains; smooth + right arm untouched."""
    env, sim, seed = _env_with_sim()
    ctl = env.robot_controller
    actions = T._synthetic_actions(env, seed)
    env.start(run_collect=False, run_exec=True)          # release loop now draining _robot_q live
    try:
        obs_id = 0
        for k in range(4):
            ok, reason = env.auto_ingest_chunk(actions, obs_step_id=obs_id)
            assert ok, f"R3: ingest {k} failed: {reason}"
            time.sleep(0.3)
            obs_id = env._current_row_id                 # next obs anchored at the arm's actual row
        time.sleep(0.5)
        moves = np.array(ctl.moves)
        assert len(moves) > 10, "R3: robot barely moved (release loop not draining)"
        dmax = float(np.max(np.abs(np.diff(moves, axis=0))))
        assert dmax <= MAX_JOINT_STEP + 1e-6, f"R3: concurrent step {dmax:.4f} > cap"
        assert not ctl.right_touched, "R3: right arm was commanded"
    finally:
        env.stop()
        sim.disconnect()
    if verbose:
        print(f"[auto-ingest] R3 OK: {len(moves)} waypoints across 4 ingests, "
              f"max step {dmax:.4f} <= {MAX_JOINT_STEP}, right untouched")


def test_ensemble_by_id(verbose=True):
    """E1: master-id ensemble averages rows with equal absolute id; non-overlap = identity."""
    from real_world.inference_controller import InferenceController, TE_M
    ctl = InferenceController(env=object())              # _temporal_ensemble only touches _te_buffer
    n = 8
    # chunk A anchored at id 0; chunk B (newest) at id 3. col 0 encodes the row value; gripper col 9:
    # A all closed (1), B all open (0) -> tests linear avg + gripper re-binarisation.
    a = np.zeros((n, 10)); a[:, 0] = 100 + np.arange(n); a[:, 9] = 1.0
    b = np.zeros((n, 10)); b[:, 0] = np.arange(n);       b[:, 9] = 0.0
    ctl._te_buffer.append((0, a.copy()))
    ctl._te_buffer.append((3, b.copy()))
    out = ctl._temporal_ensemble(3, b.copy())            # ensembling B (id_new=3)
    # k=5 -> target id 8 -> A has no row 8 (only the newest covers it) -> identity.
    assert np.allclose(out[5], b[5]), "E1: non-overlapping row should be identity"
    # k=0 -> target id 3 -> A row 3 (=103) averaged with B row 0 (=0), recency-weighted (A 1 older).
    w_a, w_b = np.exp(-TE_M * 1), np.exp(-TE_M * 0)
    exp0 = (w_a * 103 + w_b * 0) / (w_a + w_b)
    assert abs(out[0, 0] - exp0) < 1e-6, f"E1: overlap avg {out[0,0]:.3f} != {exp0:.3f}"
    # gripper (w_a*1 + w_b*0)/(w_a+w_b) ~0.495 < 0.5 -> 0
    assert out[0, 9] == 0.0, "E1: gripper should re-binarise to 0"
    if verbose:
        print(f"[auto-ingest] E1 OK: overlap avg {out[0,0]:.3f}, non-overlap identity, "
              f"gripper re-binarised")


def run(verbose=True):
    test_ramp_ingest_idle(verbose)
    test_master_id_drop(verbose)
    test_concurrent_ingest(verbose)
    test_ensemble_by_id(verbose)
    if verbose:
        print("[auto-ingest] ALL AUTO-INGEST TESTS PASS (R1 R2 R3 E1)")
    return True


if __name__ == "__main__":
    run()
