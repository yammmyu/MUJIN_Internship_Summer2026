#!/usr/bin/env python3
"""Estimate the C7 EE safe region from recorded episodes.

The C7 runtime watchdog (real_world/postprocess.py: EE_SAFE_REGION_LEFT/RIGHT) latches the E-stop
when a commanded end-effector position leaves the box where the robot has ever safely operated.
This script (re)derives that box from a directory of recordings so it stays honest to the actual
task/workcell — run it whenever the demos or the setup change and paste the printed tuples into
postprocess.py.

Method: per arm, take the full per-axis min/max of the demonstrated EE positions (dropping
non-finite frames), expand by MARGIN metres for legitimate policy extrapolation, and round OUTWARD
to the centimetre so rounding can never shrink the box below the data. Frames are the recorded
left_pos / right_pos (firmware EE frame) — the SAME frame the runtime FK produces (verified:
FK(recorded joints) matches recorded EE to sub-mm), so the box is directly usable at dispatch.

Usage:
    .venv/bin/python scripts/estimate_ee_region.py [RECORDINGS_DIR] [--margin 0.12]
"""

import argparse
import glob
import math
import os

import numpy as np

DEFAULT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "MDM_data_collection", "recordings")


def _load_positions(recordings_dir):
    """Concatenate finite left/right EE positions across every <dir>/*/robot_states.npz."""
    paths = sorted(glob.glob(os.path.join(recordings_dir, "*", "robot_states.npz")))
    if not paths:
        raise SystemExit(f"no robot_states.npz under {recordings_dir}")
    L, R = [], []
    for p in paths:
        d = np.load(p, allow_pickle=True)
        lp = np.asarray(d["left_pos"], float)
        rp = np.asarray(d["right_pos"], float)
        L.append(lp[np.isfinite(lp).all(1)])
        R.append(rp[np.isfinite(rp).all(1)])
    return np.concatenate(L), np.concatenate(R), len(paths)


def estimate_box(positions, margin):
    """(x,y,z) min/max of `positions` expanded by `margin` and rounded OUTWARD to the cm."""
    box = []
    for i in range(3):
        lo = math.floor((positions[:, i].min() - margin) * 100) / 100
        hi = math.ceil((positions[:, i].max() + margin) * 100) / 100
        box.append((lo, hi))
    return tuple(box)


def _fraction_inside(positions, box):
    m = np.ones(len(positions), bool)
    for i, (lo, hi) in enumerate(box):
        m &= (positions[:, i] >= lo) & (positions[:, i] <= hi)
    return float(m.mean())


def main():
    ap = argparse.ArgumentParser(description="Estimate the C7 EE safe region from recordings.")
    ap.add_argument("recordings_dir", nargs="?", default=DEFAULT_DIR)
    ap.add_argument("--margin", type=float, default=0.12,
                    help="metres added beyond the demonstrated min/max on every face (default 0.12)")
    args = ap.parse_args()

    L, R, n = _load_positions(args.recordings_dir)
    bl = estimate_box(L, args.margin)
    br = estimate_box(R, args.margin)
    print(f"# {n} recordings, {len(L)} left / {len(R)} right frames, margin={args.margin} m")
    print(f"# demos inside: left={_fraction_inside(L, bl):.4f}  right={_fraction_inside(R, br):.4f} "
          f"(must be 1.0000)")
    print("# paste into real_world/postprocess.py:")
    print(f"EE_SAFE_REGION_LEFT  = {bl}")
    print(f"EE_SAFE_REGION_RIGHT = {br}")


if __name__ == "__main__":
    main()
