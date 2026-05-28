"""
Trajectory smoothing / filtering for recorded robot episodes.

Input:  recordings/episode_NNN/robot_states.npz
Output: recordings_filtered/episode_NNN/robot_states.npz  (same keys, smoothed values)

Camera videos and metadata.json are copied unchanged — only the NPZ is modified.

Usage:
    python data_filter.py                          # recordings/ → recordings_filtered/
    python data_filter.py --src my_recs --dst out  # custom paths

Filter parameters to tune:
  SG_WINDOW    : Savitzky-Golay window length (odd, >= SG_POLYORDER + 1)
  SG_POLYORDER : polynomial order (3 = cubic, good for smooth trajectories)

Quaternion smoothing note:
  Component-wise SG + renormalisation is an approximation. For strict geodesic
  smoothing implement SLERP-based averaging (TODO).
"""
import argparse
import pathlib
import shutil

import numpy as np
from scipy.signal import savgol_filter

# ─── Parameters (tune these) ──────────────────────────────────────────────────
SG_WINDOW    = 7   # must be odd and >= SG_POLYORDER + 1
SG_POLYORDER = 3


def _sg(arr: np.ndarray) -> np.ndarray:
    w = min(SG_WINDOW, arr.shape[0] if arr.shape[0] % 2 == 1 else arr.shape[0] - 1)
    w = max(w, SG_POLYORDER + 1)
    return savgol_filter(arr, window_length=w, polyorder=SG_POLYORDER, axis=0)


def filter_positions(arr: np.ndarray) -> np.ndarray:
    """Smooth (N, 3) position trajectory."""
    return _sg(arr)


def filter_quaternions(arr: np.ndarray) -> np.ndarray:
    """
    Approximate quaternion smoothing: SG per component + renormalise.
    TODO: replace with proper SLERP-based geodesic smoothing.
    """
    smoothed = _sg(arr)
    norms    = np.linalg.norm(smoothed, axis=1, keepdims=True)
    return smoothed / np.maximum(norms, 1e-8)


def filter_joints(arr: np.ndarray) -> np.ndarray:
    """Smooth (N, 14) joint angle trajectory."""
    return _sg(arr)


def filter_episode(src_ep: pathlib.Path, dst_ep: pathlib.Path) -> None:
    dst_ep.mkdir(parents=True, exist_ok=True)
    data = dict(np.load(src_ep / 'robot_states.npz'))

    data['left_pos']   = filter_positions(data['left_pos'])
    data['right_pos']  = filter_positions(data['right_pos'])
    data['left_quat']  = filter_quaternions(data['left_quat'])
    data['right_quat'] = filter_quaternions(data['right_quat'])
    data['arm_joints'] = filter_joints(data['arm_joints'])
    # timestamps and gripper are left unchanged

    np.savez(dst_ep / 'robot_states.npz', **data)

    if (src_ep / 'cameras').exists():
        shutil.copytree(src_ep / 'cameras', dst_ep / 'cameras', dirs_exist_ok=True)
    if (src_ep / 'metadata.json').exists():
        shutil.copy2(src_ep / 'metadata.json', dst_ep / 'metadata.json')

    print(f"  filtered {src_ep.name}  ({len(data['timestamps'])} frames)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', default='recordings',          help='Source directory')
    parser.add_argument('--dst', default='recordings_filtered', help='Output directory')
    args = parser.parse_args()

    src = pathlib.Path(args.src)
    dst = pathlib.Path(args.dst)

    episodes = sorted(src.glob('episode_*'))
    if not episodes:
        print(f"No episodes found in {src}")
        raise SystemExit(1)

    for ep in episodes:
        filter_episode(ep, dst / ep.name)

    print(f"\nDone. {len(episodes)} episodes written to {dst}/")
