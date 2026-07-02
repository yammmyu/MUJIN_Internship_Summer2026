"""Offline smoothness / velocity analysis for the auto-inference pipeline.

The per-inference logs (chunks.jsonl, buffer.jsonl) re-emit OVERLAPPING master ids every inference,
so plotting them in file order is a sawtooth artifact, NOT the executed motion. This tool collapses
them to one value PER MASTER ID (the apples-to-apples view) and measures:

  * executed-trajectory smoothness  (buffer frozen-per-id  vs  raw newest-chunk-per-id  vs  the
    single-chunk internal baseline — the policy's own smoothness, the best achievable);
  * per-substep joint velocity from released_substeps.jsonl, flagging the fast seam spikes and
    whether they land on chunk-row boundaries (the ramp signature).

Usage:
    python -m scripts.analyze_smoothness [DIR]      # DIR holds the *.jsonl (default: cwd)
"""
import json
import os
import sys
import pathlib

import numpy as np

# Allow `python scripts/analyze_smoothness.py` (not just `-m scripts.analyze_smoothness`)
# to resolve `real_world.*` by putting the repo root on the path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load(path):
    return [json.loads(l) for l in open(path) if l.strip()] if path.exists() else []


def _d2rms(v):
    return float(np.sqrt((np.diff(v, n=2) ** 2).mean())) if len(v) >= 3 else float("nan")


def per_id_smoothness(chunks, buf):
    """Collapse to one row per master id, then report |Δ²|rms per EE component."""
    buf_final = {}                                   # last write per id == its frozen value
    for e in buf:
        b0 = e["base_id"]
        for i, row in enumerate(e["buffer"]):
            buf_final[b0 + i] = row
    raw_newest = {}                                  # newest chunk's value per id (old pipeline)
    for e in chunks:
        sid = e["obs_ts"]
        for k, row in enumerate(e["action"]):
            raw_newest[sid + k] = row

    def traj(d, c):
        return np.array([d[i][c] for i in sorted(d)])

    print("=== executed-trajectory smoothness (|Δ²|rms per master id; lower = smoother) ===")
    comps = [(0, "eef x"), (1, "eef y"), (2, "eef z"), (3, "rot6d c0x")]
    # single-chunk internal baseline (the policy's own smoothness)
    if chunks:
        base = {c: np.mean([_d2rms(np.array([r[c] for r in e["action"]])) for e in chunks])
                for c, _ in comps}
    else:
        base = {c: float("nan") for c, _ in comps}
    hdr = f"{'component':12s} {'single-chunk':>13s} {'raw newest/id':>14s} {'BUFFER/id':>12s}"
    print(hdr)
    for c, lbl in comps:
        print(f"{lbl:12s} {base[c]:13.6f} "
              f"{_d2rms(traj(raw_newest, c)):14.6f} {_d2rms(traj(buf_final, c)):12.6f}")
    print(f"(ids: raw={len(raw_newest)}, buffer={len(buf_final)})\n")


def substep_velocity(rel, control_hz=None, max_joint_step=None):
    if control_hz is None or max_joint_step is None:
        try:
            from real_world.timing import CONTROL_HZ, MAX_JOINT_STEP
            control_hz, max_joint_step = CONTROL_HZ, MAX_JOINT_STEP
        except Exception:
            control_hz, max_joint_step = 120.0, 4.0 / 120.0
    """Per-substep joint velocity from released_substeps.jsonl; flag seam spikes."""
    if not rel:
        print("=== substep velocity: no released_substeps.jsonl ===\n")
        return
    q = np.array([r["q7"] for r in rel], dtype=float)
    sid = np.array([(-(10 ** 9) if r["step_id"] is None else r["step_id"]) for r in rel])
    dq = np.abs(np.diff(q, axis=0)).max(axis=1)      # max over 7 joints, per tick
    vel = dq * control_hz                            # rad/s
    boundary = sid[1:] != sid[:-1]                   # tick crosses a master-id (row/seam) boundary
    print("=== per-substep joint velocity (released_substeps.jsonl) ===")
    print(f"velocity rad/s: median={np.median(vel):.3f} p95={np.percentile(vel,95):.3f} "
          f"p99={np.percentile(vel,99):.3f} max={vel.max():.3f}  "
          f"(safety cap = {max_joint_step*control_hz:.2f})")
    if boundary.any() and (~boundary).any():
        print(f"mean |Δq| at id-boundary={dq[boundary].mean():.5f}  vs  mid-row={dq[~boundary].mean():.5f} "
              f"(ratio {dq[boundary].mean()/max(dq[~boundary].mean(),1e-9):.1f}x => ramp spikes if >>1)")
    top = np.argsort(dq)[::-1][:50]
    print(f"of the 50 fastest substeps, {int(boundary[top].sum())} are on a master-id boundary\n")


def main():
    d = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    chunks = _load(d / "chunks.jsonl")
    buf = _load(d / "buffer.jsonl")
    rel = _load(d / "released_substeps.jsonl")
    qts = [e.get("queued_through") for e in buf]
    if qts and any(b < a for a, b in zip(qts, qts[1:])):
        print("NOTE: queued_through is non-monotonic -> the run had E-stop resets; "
              "per-id metrics still valid, but a clean (reset-free) run is better for tuning.\n")
    per_id_smoothness(chunks, buf)
    substep_velocity(rel)


if __name__ == "__main__":
    main()
