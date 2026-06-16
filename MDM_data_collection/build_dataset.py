"""
Converts filtered robot recordings into a Zarr dataset compatible with
diffusion_policy/config/task/left_arm_ee_image.yaml.

Left arm only.

Output Zarr schema:
  data/agentview_image              (N, 84, 84, 3)  uint8    — head camera (RGB)
  data/robot0_eye_in_hand_image     (N, 84, 84, 3)  uint8    — hand_left camera (RGB)
  data/robot0_eef_pos               (N, 3)          float32  — EE position (metres)
  data/robot0_eef_quat              (N, 4)          float32  — EE quaternion [qx,qy,qz,qw]
  data/robot0_left_joint            (N, 8)          float32  — 7 left arm joint angles (rad) + 1 left gripper
  data/action                       (N, 10)         float32  — [pos(3) + rot6d(6) + gripper(1)]
  meta/episode_ends                 (E,)            int64    — cumulative frame boundaries

Usage:
    python build_dataset.py                          # recordings_filtered/ → robot_dataset.zarr
    python build_dataset.py --src recordings --out my.zarr
"""
import argparse
import pathlib

import cv2
import numcodecs
import numpy as np
import zarr
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R

# ─── Configuration ────────────────────────────────────────────────────────────
IMG_SIZE      = 84
CHUNK_T       = 100
COMPRESSOR    = numcodecs.Blosc(cname='zstd', clevel=5, shuffle=numcodecs.Blosc.BITSHUFFLE)
SMOOTH_SIGMA  = 1.7   # Gaussian sigma in frames; set to 0 to disable smoothing
FROZEN_JOINT_EPS = 1e-3  # rad; per-episode joint range below this ⇒ frozen arm_joint_states


# ─── Rotation conversion ─────────────────────────────────────────────────────

def quat_to_rot6d(quat: np.ndarray) -> np.ndarray:
    """
    (N, 4) [qx, qy, qz, qw] → (N, 6) continuous 6D rotation representation.
    First two columns of the rotation matrix, row-major.
    """
    mat = R.from_quat(quat).as_matrix()                          # (N, 3, 3)
    return np.concatenate([mat[..., 0], mat[..., 1]], axis=-1)   # (N, 6)


def smooth_pos(pos: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smooth (N, 3) position along time axis."""
    if sigma <= 0:
        return pos
    return gaussian_filter1d(pos, sigma=sigma, axis=0)


def _sign_fix_and_smooth_quat(quat: np.ndarray, sigma: float) -> np.ndarray:
    """
    Return a Gaussian-smoothed (N, 4) quaternion for a single episode.
    Sign-consistency fix runs before filtering so the kernel never blends
    q with -q. Re-normalisation after filtering returns the result to S³.
    Must be called per-episode — do NOT pass a concatenated multi-episode array.
    """
    q = quat.copy()
    if sigma > 0:
        for i in range(1, len(q)):
            if np.dot(q[i], q[i - 1]) < 0:
                q[i] *= -1
        q = gaussian_filter1d(q, sigma=sigma, axis=0)
        q /= np.linalg.norm(q, axis=1, keepdims=True)
    return q


# ─── Video loading ────────────────────────────────────────────────────────────

def read_video(path: pathlib.Path, size: int = IMG_SIZE) -> np.ndarray:
    """Returns (N, size, size, 3) uint8 RGB, or empty array if file missing."""
    if not path.exists():
        return np.empty((0, size, size, 3), dtype=np.uint8)
    cap    = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (size, size))
        frames.append(frame.astype(np.uint8))
    cap.release()
    return np.stack(frames) if frames else np.empty((0, size, size, 3), dtype=np.uint8)


# ─── Zarr helpers ─────────────────────────────────────────────────────────────

def init_zarr(out_path: pathlib.Path) -> tuple[zarr.Group, dict]:
    store = zarr.open_group(str(out_path), mode='w', zarr_format=2)
    store.require_group('data')
    store.require_group('meta')

    kw = dict(compressor=COMPRESSOR)

    def arr(name, *shape_rest, dtype='f4'):
        return store['data'].empty(
            name=name,
            shape      = (0, *shape_rest),
            chunks     = (CHUNK_T, *shape_rest),
            dtype      = dtype,
            zarr_format= 2,
            **kw,
        )

    arrays = {
        'agentview_image':          arr('agentview_image',          IMG_SIZE, IMG_SIZE, 3, dtype='u1'),
        'robot0_eye_in_hand_image': arr('robot0_eye_in_hand_image', IMG_SIZE, IMG_SIZE, 3, dtype='u1'),
        'robot0_eef_pos':           arr('robot0_eef_pos',           3),
        'robot0_eef_quat':          arr('robot0_eef_quat',          4),
        'robot0_left_joint':        arr('robot0_left_joint',        8),
        'action':                   arr('action',                   10),
    }
    return store, arrays


def append_to(arr: zarr.Array, data: np.ndarray) -> None:
    n_old = arr.shape[0]
    n_new = data.shape[0]
    arr.resize((n_old + n_new, *arr.shape[1:]))
    arr[n_old:] = data


# ─── Main ─────────────────────────────────────────────────────────────────────

def build(src_dir: pathlib.Path, out_path: pathlib.Path) -> None:
    episodes = sorted(src_dir.glob('recording*'))
    if not episodes:
        raise FileNotFoundError(f"No episodes found in {src_dir}")
    print(f"Found {len(episodes)} episodes in {src_dir}\n")

    store, arrays = init_zarr(out_path)
    episode_ends  = []
    total         = 0

    # accumulated for smoothing-comparison NPZ
    raw_pos_all, raw_quat_all     = [], []
    smooth_pos_all, smooth_quat_all = [], []

    for ep in episodes:
        print(f"Processing {ep.name}...")

        states = np.load(ep / 'robot_states.npz')
        n      = len(states['timestamps'])

        # ── Images ──────────────────────────────────────────────────────────
        head_frames = read_video(ep / 'cameras' / 'head.mp4')      # (N, 84, 84, 3)
        hand_frames = read_video(ep / 'cameras' / 'hand_left.mp4') # (N, 84, 84, 3)

        # Guard against minor frame-count drift between video and NPZ
        n = min(n, len(head_frames), len(hand_frames))
        if n == 0:
            print(f"  SKIP — no usable frames")
            continue

        head_frames = head_frames[:n]
        hand_frames = hand_frames[:n]

        # ── Left arm state ───────────────────────────────────────────────────
        left_pos   = states['left_pos'][:n]        # (N, 3)
        left_quat  = states['left_quat'][:n]       # (N, 4)  [qx,qy,qz,qw]
        gripper    = states['gripper'][:n, :1]     # (N, 1)  left gripper value
        left_joint = states['arm_joints'][:n, :7]  # (N, 7)  left arm joint angles (rad)

        # Guard: arm_joint_states() freezes (stale, constant) unless a Slam()
        # instance was alive in the collection process. A frozen episode yields a
        # near-constant robot0_left_joint, which is useless as an observation.
        if n > 1:
            joint_span = float(np.ptp(left_joint, axis=0).max())
            if joint_span < FROZEN_JOINT_EPS:
                print(f"  WARNING — arm_joints look FROZEN "
                      f"(max range {joint_span:.2e} rad < {FROZEN_JOINT_EPS:.0e}); "
                      f"was Slam() running during collection?")

        # ── Action = [pos(3) + rot6d(6) + gripper(1)] ───────────────────────
        # Observations stay raw (must match camera ground truth).
        # Actions use smoothed trajectories to reduce teleoperation jerk.
        pos_smooth  = smooth_pos(left_pos, SMOOTH_SIGMA)
        quat_smooth = _sign_fix_and_smooth_quat(left_quat, SMOOTH_SIGMA)
        rot6d       = quat_to_rot6d(quat_smooth)                            # (N, 6)
        action      = np.concatenate([pos_smooth, rot6d, gripper], axis=1)  # (N, 10)

        raw_pos_all.append(left_pos)
        raw_quat_all.append(left_quat)
        smooth_pos_all.append(pos_smooth)
        smooth_quat_all.append(quat_smooth)

        # left_joint = 7 arm joint angles + 1 gripper, to satisfy config shape [8]
        robot0_left_joint = np.concatenate([left_joint, gripper], axis=1)  # (N, 8)

        # ── Append ───────────────────────────────────────────────────────────
        append_to(arrays['agentview_image'],          head_frames)
        append_to(arrays['robot0_eye_in_hand_image'], hand_frames)
        append_to(arrays['robot0_eef_pos'],           left_pos)
        append_to(arrays['robot0_eef_quat'],          left_quat)
        append_to(arrays['robot0_left_joint'],        robot0_left_joint)
        append_to(arrays['action'],                   action)

        total += n
        episode_ends.append(total)
        print(f"  {n} frames  (total so far: {total})")

    # ── Smoothing comparison NPZ ──────────────────────────────────────────────
    npz_path = out_path.with_suffix('.smooth_debug.npz')
    np.savez(
        npz_path,
        raw_pos          = np.concatenate(raw_pos_all,    axis=0),
        raw_quat         = np.concatenate(raw_quat_all,   axis=0),
        smooth_pos       = np.concatenate(smooth_pos_all, axis=0),
        smooth_quat      = np.concatenate(smooth_quat_all, axis=0),
        episode_ends     = np.array(episode_ends, dtype=np.int64),
    )
    print(f"  Smoothing debug data → {npz_path}")

    # ── Episode ends ──────────────────────────────────────────────────────────
    ep_ends = np.array(episode_ends, dtype=np.int64)
    meta_arr = store['meta'].empty(
        name='episode_ends',
        shape=ep_ends.shape,
        chunks=ep_ends.shape,
        dtype=np.int64,
        zarr_format=2,
        compressor=COMPRESSOR,
    )
    meta_arr[:] = ep_ends

    print(f"\nDataset written to {out_path}")
    print(f"  Total frames : {total}")
    print(f"  Episodes     : {len(episode_ends)}")
    print(f"  action shape : {arrays['action'].shape}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', default='recordings_filtered',
                        help='Source dir (falls back to recordings/ if not found)')
    parser.add_argument('--out', default='robot_dataset.zarr',
                        help='Output Zarr store path')
    parser.add_argument('--sigma', type=float, default=SMOOTH_SIGMA,
                        help='Gaussian sigma (frames) for action smoothing; 0 = disabled')
    args = parser.parse_args()
    SMOOTH_SIGMA = args.sigma

    src = pathlib.Path(args.src)
    if not src.exists():
        fallback = pathlib.Path('recordings')
        print(f"'{src}' not found, falling back to '{fallback}'")
        src = fallback

    build(src, pathlib.Path(args.out))
