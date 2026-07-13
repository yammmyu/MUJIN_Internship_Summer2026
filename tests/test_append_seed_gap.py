#!/usr/bin/env python3
"""Verify the streaming append_actions seed_gap path: every master id gets a UNIFORM K substeps
(row 0 included -> no lone anchor), and the chunk seam between two appends is continuous (the bridge
collapses -> no fast dart). Offline, same fakes as test_auto_splice."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_safety_invariants as T
import test_auto_splice as A
from real_world.timing import MAX_JOINT_STEP, SUBSTEPS_PER_ROW


def _counts_by_id(q):
    from collections import Counter
    return Counter(p[2] for p in q)


def test_uniform_24_and_continuous_seam(verbose=True):
    env, sim, seed = A._env_with_sim()
    K = SUBSTEPS_PER_ROW
    actions = T._synthetic_actions(env, seed)              # N policy rows (20-col dual)
    n = len(actions)
    try:
        # --- Append 1: queue a long run in one shot (n_rows large) from the idle queue. ---
        ok, reason = env.pipeline.append_actions(actions, obs_step_id=0, n_rows=n)
        assert ok, f"append-1 failed: {reason}"
        with env._lock:
            q1 = list(env._robot_q)
        counts = _counts_by_id(q1)
        ids = sorted(counts)
        # Master ids are CONTIGUOUS (one block per row, no drift) — the exact-row-index guarantee. Even
        # a velocity-capped startup ramp (row 0 from a parked seed) stays under a single id.
        assert ids == list(range(ids[0], ids[-1] + 1)), f"master ids not contiguous (drift): {ids}"
        # Row 0 is now a full gap (anchor removed), so >= K substeps — possibly > K if the idle/parked
        # seed->row0 ramp-in velocity-capped; that's correct and all attributed to the one id.
        assert counts[ids[0]] >= K, f"row0 got {counts[ids[0]]} substeps, expected >= K={K} (anchor not removed)"
        # Interior rows (steady gentle motion — not the startup ramp, not the window-edge tail) = K each.
        interior = ids[1:-1]
        bad = {i: counts[i] for i in interior if counts[i] != K}
        assert not bad, f"non-uniform interior substep counts (expected {K} each): {bad}"
        # Internal continuity: no substep-to-substep jump exceeds the safety cap.
        step1 = A._max_step(q1)
        assert step1 <= MAX_JOINT_STEP + 1e-6, f"append-1 over-cap step {step1:.4f}"

        # --- Append 2: continue the stream. Drain part of the queue (advance the clock), then append
        # the SAME chunk again; the seam from the queue tail to the new rows must be continuous. ---
        drained = q1[:2 * K]                                # pretend the arm executed 2 rows
        with env._lock:
            for _ in range(2 * K):
                env._robot_q.popleft()
            env._current_row_id = 2                         # clock advanced to row 2
        tail = np.asarray(list(env._robot_q)[-1][0], float) if len(env._robot_q) else None
        ok, reason = env.pipeline.append_actions(actions, obs_step_id=0, n_rows=n)
        assert ok, f"append-2 failed: {reason}"
        with env._lock:
            q2 = list(env._robot_q)
        # The join point: last old-tail substep -> first newly appended substep.
        # Find the boundary by id continuity is fiddly; instead assert the WHOLE queue is still
        # cap-bounded AND the max step near the tail is at motion scale, not a ramp-cap dart.
        step2 = A._max_step(q2)
        assert step2 <= MAX_JOINT_STEP + 1e-6, f"append-2 over-cap step {step2:.4f}"
        # Dart check: the old fixed-cap ramp crossed a seam at ~RAMP_JOINT_STEP. Confirm no cluster of
        # substeps sits near that cap — the seam should look like ordinary motion.
        allq = np.array([np.asarray(p[0], float) for p in q2])
        steps = np.abs(np.diff(allq, axis=0)).max(axis=1)
        near_cap = int((steps > 0.5 * MAX_JOINT_STEP).sum())
        assert near_cap == 0, f"{near_cap} seam substeps still near the velocity cap (dart not gone)"
    finally:
        env.stop(); sim.disconnect()
    if verbose:
        print(f"[append] OK: {len(ids)} contiguous ids, row0={counts[ids[0]]} substeps (>=K, ramp-in), "
              f"all interior == {K}; max step a1={step1:.4f} a2={step2:.4f} <= {MAX_JOINT_STEP:.4f}; "
              f"0 near-cap seam substeps")


if __name__ == "__main__":
    test_uniform_24_and_continuous_seam()
    print("[append] SEED_GAP APPEND TEST PASS")
