#!/usr/bin/env python3
"""Offline tests for auto-inference -> robot ingest (env.auto_ingest_chunk) and the master-ID
temporal ensemble. Runs fully offline with the same fakes as test_safety_invariants.

  R1  ramp ingest (idle)  -> validate a chunk and lay a ramp-in from the live pose to its rows,
                             every step <= MAX_JOINT_STEP, row ids monotonic from obs_step_id.
  R2  master-ID drop      -> a chunk whose obs row id is BEHIND the current row id keeps only the
                             still-future rows (drops the part the arm already passed).
  R3  concurrent ingest   -> ingest WHILE the release loop drains: BOTH arms commanded (dual-arm),
                             every per-substep joint step within the C5 cap on each arm.
  E1  ensemble by id      -> _rebuild_buffer averages rows sharing an absolute master id
                             (recency-weighted), passes non-overlapping rows through (identity), and
                             re-binarises the gripper; _materialize returns the contiguous run.

Run:
    .venv/bin/python tests/test_auto_splice.py
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_safety_invariants as T                                              # noqa: E402
import real_world.postprocess as planner_mod                                    # noqa: E402 (workspace envelopes live here now)
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
        q0 = np.asarray(q0, dtype=float)
        self._left = q0[:7].copy()
        self._right = np.zeros(7)          # closed-loop on BOTH arms (dual-arm pipeline)
        self.moves = []                    # left-arm waypoints
        self.moves_r = []                  # right-arm waypoints
        self.right_touched = False

    def arm_joint_states(self):
        return (np.concatenate([self._left, self._right]).tolist(), 0)

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
                if "right_arm" in a:                                             # closed loop (right)
                    self._right = np.asarray(a["right_arm"]["action_data"], dtype=float)
                    self.moves_r.append(self._right.copy())
            self._left = np.asarray(a["left_arm"]["action_data"], dtype=float)   # closed loop (left)
            self.moves.append(self._left.copy())


def _env_with_sim():
    planner_mod.WORKSPACE_AABB = ((-9, 9), (-9, 9), (-9, 9))   # isolate from the H4 envelope check
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
    """R3: ingest several chunks WHILE the release loop drains; the dual-arm pipeline commands
    BOTH arms and every per-substep joint step stays within the C5 cap on each arm."""
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
        assert dmax <= MAX_JOINT_STEP + 1e-6, f"R3: concurrent LEFT step {dmax:.4f} > cap"
        # Dual-arm: the right arm IS commanded now, in lockstep with the left and equally bounded.
        assert ctl.right_touched, "R3: right arm should be commanded (dual-arm pipeline)"
        moves_r = np.array(ctl.moves_r)
        dmax_r = float(np.max(np.abs(np.diff(moves_r, axis=0)))) if len(moves_r) > 1 else 0.0
        assert dmax_r <= MAX_JOINT_STEP + 1e-6, f"R3: concurrent RIGHT step {dmax_r:.4f} > cap"
    finally:
        env.stop()
        sim.disconnect()
    if verbose:
        print(f"[auto-ingest] R3 OK: {len(moves)} waypoints across 4 ingests, both arms commanded, "
              f"max step L {dmax:.4f} / R {dmax_r:.4f} <= {MAX_JOINT_STEP}")


def test_ensemble_by_id(verbose=True):
    """E1: the master-id temporal ensemble (_rebuild_buffer -> _materialize) averages rows sharing
    an absolute master id (recency-weighted), passes non-overlapping rows through unchanged, and
    re-binarises the gripper. Radius 0 isolates the cross-chunk average from the along-id Gaussian."""
    from real_world.postprocess import PostProcessor, TE_M, TE_SIGMA
    pp = PostProcessor()                                 # the merge lives here now (no env needed)
    pp.set_smoothing(radius=0, sigma=TE_SIGMA, m=TE_M)   # disable along-id smoothing -> pure per-id avg
    n = 8
    # chunk A anchored at id 0; chunk B (newest) at id 3. col 0 encodes the row value; gripper col 9:
    # A all closed (1), B all open (0) -> tests linear avg + gripper re-binarisation.
    a = np.zeros((n, 10)); a[:, 0] = 100 + np.arange(n); a[:, 9] = 1.0
    b = np.zeros((n, 10)); b[:, 0] = np.arange(n);       b[:, 9] = 0.0
    pp._te_buffer.append((0, a.copy()))
    pp._te_buffer.append((3, b.copy()))
    pp._rebuild_buffer(queued_through=-1)                # nothing frozen -> rebuild all ids
    base_id, buf = pp._materialize()                     # contiguous run master_id 0..10
    assert base_id == 0, f"E1: base id {base_id} != 0"
    # master id 8 -> only B covers it (B row 5 = 5) -> identity (no averaging).
    assert np.allclose(buf[8], b[5]), "E1: non-overlapping row should be identity"
    # master id 3 -> A row 3 (=103) averaged with B row 0 (=0), recency-weighted (A one older).
    w_a, w_b = np.exp(-TE_M * 1), np.exp(-TE_M * 0)
    exp3 = (w_a * 103 + w_b * 0) / (w_a + w_b)
    assert abs(buf[3, 0] - exp3) < 1e-5, f"E1: overlap avg {buf[3,0]:.3f} != {exp3:.3f}"
    # gripper (w_a*1 + w_b*0)/(w_a+w_b) ~0.495 < 0.5 -> 0
    assert buf[3, 9] == 0.0, "E1: gripper should re-binarise to 0"
    if verbose:
        print(f"[auto-ingest] E1 OK: overlap avg {buf[3,0]:.3f}, non-overlap identity, "
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
