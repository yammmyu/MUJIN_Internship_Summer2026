#!/usr/bin/env python3
"""The sim returns an EXACT per-substep row index (`rows`), so master-id tagging is one-id-per-row
even when a velocity-capped row emits more than K substeps. Contrast with the old s//K formula, which
drifts as soon as any row exceeds K substeps."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_safety_invariants as T
import test_auto_splice as A
from real_world.timing import SUBSTEPS_PER_ROW as K


def test_rows_index_exact(verbose=True):
    env, sim, seed = A._env_with_sim()
    seed14 = np.concatenate([np.asarray(seed, float)[:7], np.zeros(7)])
    actions = T._synthetic_actions(env, seed)
    configs, kept_ids, ok, r = env.pipeline.solve_chunk_ik(actions[:6], seed14)
    assert ok, r
    assert kept_ids == list(range(len(configs))), "no rows skipped here -> kept_ids is identity"
    n = len(configs)
    try:
        # Force velocity-capping on EVERY gap by using a tiny max_joint_step, so most rows emit
        # K*factor > K substeps -> the fixed-K assumption is maximally violated.
        traj, ok, reason, rows = env.sim.validate(
            configs, seed14, max_joint_step=0.003, fast=True, substeps_per_row=K, seed_gap=True)
        assert ok, f"validate failed: {reason}"
        assert len(rows) == len(traj), "rows must be parallel to traj"

        # EXACT tags: start_id + rows[s]. One contiguous block per config row, covering 0..n-1 with
        # no gaps and no drift, regardless of how many substeps each row took.
        start_id = 100
        ids = [start_id + int(rows[s]) for s in range(len(traj))]
        assert ids == sorted(ids), "ids must be monotonic"
        assert min(ids) == start_id and max(ids) == start_id + n - 1, \
            f"ids should span {start_id}..{start_id+n-1}, got {min(ids)}..{max(ids)}"
        assert sorted(set(ids)) == list(range(start_id, start_id + n)), "every row id present exactly once as a block"
        # At least one row actually exceeded K (otherwise the test proves nothing).
        from collections import Counter
        blk = Counter(rows)
        assert max(blk.values()) > K, f"no row exceeded K={K} substeps; cap not exercised (max {max(blk.values())})"

        # The OLD s//K formula would DRIFT: with rows emitting > K substeps, s//K runs past n-1.
        old_ids = [start_id + int(s // K) for s in range(len(traj))]
        assert max(old_ids) > start_id + n - 1, "old s//K should have overshot the real last row (drift)"
        drift = max(old_ids) - (start_id + n - 1)
    finally:
        env.stop(); sim.disconnect()
    if verbose:
        print(f"[tagging] OK: {len(traj)} substeps over {n} rows; capped block max={max(blk.values())} (>K={K}); "
              f"EXACT ids span {min(ids)}..{max(ids)} (one block/row); old s//K would drift +{drift} ids")


if __name__ == "__main__":
    test_rows_index_exact()
    print("[tagging] EXACT ROW-INDEX TAGGING TEST PASS")
