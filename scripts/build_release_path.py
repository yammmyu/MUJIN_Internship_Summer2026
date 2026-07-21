#!/usr/bin/env python3
"""Build a smoothed "reach + release" joint path for the flip / no-flip place macros.

Both macros (real_world/flip_place.py, real_world/no_flip_place.py) take an (M,14) absolute-joint
waypoint file that is ONLY the there-and-release trajectory: the arm reaches out to the release pose
where the gripper opens. The macro plays it forward, opens the gripper at the end, then plays it in
REVERSE to come back — so the recorded return/retreat is NOT wanted here.

Processing (same idea as the original assets/flip_release_path.npy):
  1. load arm_joints (N,14) [left7,right7] + gripper (N,2) from a recording's robot_states.npz
  2. find the RELEASE frame = where the right gripper first opens (drops from its closed value)
  3. trim to the reach: leading still frames removed, everything after release removed
  4. lightly Gaussian-smooth each joint column (repo default radius=3, sigma=1.5), endpoints pinned
     so the exact release config (last row) and start are preserved
  5. save float32 (M,14)

Usage:
    python scripts/build_release_path.py ~/Downloads/recording205 real_world/assets/no_flip_release_path.npy
    python scripts/build_release_path.py ~/Downloads/recording206 real_world/assets/flip_release_path.npy
"""
import argparse
import os

import numpy as np

R_GRIP_IDX = 1          # right channel in [left, right] gripper


def gaussian_kernel(radius=3, sigma=1.5):
    ks = np.arange(-radius, radius + 1)
    g = np.exp(-(ks ** 2) / (2.0 * sigma ** 2))
    return g / g.sum()


def smooth_cols(seg, radius=3, sigma=1.5):
    """Per-column 1-D Gaussian smoothing with reflect padding; first/last rows pinned to raw."""
    if len(seg) <= 2:
        return seg.copy()
    k = gaussian_kernel(radius, sigma)
    pad = np.pad(seg, ((radius, radius), (0, 0)), mode="reflect")
    out = np.empty_like(seg)
    for j in range(seg.shape[1]):
        out[:, j] = np.convolve(pad[:, j], k, mode="valid")
    out[0] = seg[0]         # pin start (shape only; the macro re-anchors it to the live pose anyway)
    out[-1] = seg[-1]       # pin the exact release config -> correct release EE via FK
    return out


def build(rec_dir, radius=3, sigma=1.5, move_eps=2e-3):
    st = np.load(os.path.join(rec_dir, "robot_states.npz"))
    aj = st["arm_joints"].astype(np.float64)          # (N,14)
    grip = st["gripper"].astype(np.float64)           # (N,2)
    n = len(aj)
    gr = grip[:, R_GRIP_IDX]
    closed = float(gr.max())

    # release = first frame the right gripper leaves 'closed' (starts opening)
    opening = np.where(gr < 0.97 * closed)[0]
    release_idx = int(opening[0]) if len(opening) else n - 1

    # motion start = first frame within the reach where the right arm actually begins to move
    step = np.linalg.norm(np.diff(aj[:release_idx + 1, 7:14], axis=0), axis=1)
    moving = np.where(step > move_eps)[0]
    start_idx = int(moving[0]) if len(moving) else 0

    seg = aj[start_idx:release_idx + 1]                # reach -> release (inclusive), no retreat
    out = smooth_cols(seg, radius, sigma).astype(np.float32)
    return out, dict(n=n, start_idx=start_idx, release_idx=release_idx, kept=len(out), closed=closed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recording", help="path to a recordingNNN dir (has robot_states.npz)")
    ap.add_argument("out", help="output .npy path")
    ap.add_argument("--radius", type=int, default=3)
    ap.add_argument("--sigma", type=float, default=1.5)
    args = ap.parse_args()

    out, info = build(args.recording, args.radius, args.sigma)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.save(args.out, out)
    print(f"{args.recording}: N={info['n']}  reach[{info['start_idx']}:{info['release_idx']}]  "
          f"-> {info['kept']} smoothed waypoints (radius={args.radius}, sigma={args.sigma})")
    print(f"  saved {out.shape} float32 -> {args.out}")
    print(f"  release (last) right-arm config: {np.round(out[-1, 7:14], 3).tolist()}")


if __name__ == "__main__":
    main()
