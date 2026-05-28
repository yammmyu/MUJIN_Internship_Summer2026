"""
Converts filtered robot recordings into a Zarr dataset compatible with
diffusion_policy/config/task/lift_image_abs.yaml.

Left arm only.

Output Zarr schema:
  data/agentview_image              (N, 84, 84, 3)  float32  — head camera (RGB)
  data/robot0_eye_in_hand_image     (N, 84, 84, 3)  float32  — hand_left camera (RGB)
  data/robot0_eef_pos               (N, 3)          float32  — EE position (metres)
  data/robot0_eef_quat              (N, 4)          float32  — EE quaternion [qx,qy,qz,qw]
  data/robot0_gripper_qpos          (N, 2)          float32  — left gripper (duplicated to match shape [2])
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
from scipy.spatial.transform import Rotation as R

# ─── Configuration ────────────────────────────────────────────────────────────
IMG_SIZE   = 84
CHUNK_T    = 100
COMPRESSOR = numcodecs.Blosc(cname='zstd', clevel=5, shuffle=numcodecs.Blosc.BITSHUFFLE)


# ─── Rotation conversion ─────────────────────────────────────────────────────

def quat_to_rot6d(quat: np.ndarray) -> np.ndarray:
    """
    (N, 4) [qx, qy, qz, qw] → (N, 6) continuous 6D rotation representation.
    First two columns of the rotation matrix, row-major.
    """
    mat = R.from_quat(quat).as_matrix()                          # (N, 3, 3)
    return np.concatenate([mat[..., 0], mat[..., 1]], axis=-1)   # (N, 6)


# ─── Video loading ────────────────────────────────────────────────────────────

def read_video(path: pathlib.Path, size: int = IMG_SIZE) -> np.ndarray:
    """Returns (N, size, size, 3) float32 RGB, or empty array if file missing."""
    if not path.exists():
        return np.empty((0, size, size, 3), dtype=np.float32)
    cap    = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (size, size))
        frames.append(frame.astype(np.float32))
    cap.release()
    return np.stack(frames) if frames else np.empty((0, size, size, 3), dtype=np.float32)


# ─── Zarr helpers ─────────────────────────────────────────────────────────────

def init_zarr(out_path: pathlib.Path) -> tuple[zarr.Group, dict]:
    store = zarr.open_group(str(out_path), mode='w')
    store.require_group('data')
    store.require_group('meta')

    kw = dict(compressor=COMPRESSOR)

    def arr(name, *shape_rest, dtype='f4'):
        return store['data'].empty(
            name,
            shape  = (0, *shape_rest),
            chunks = (CHUNK_T, *shape_rest),
            dtype  = dtype,
            **kw,
        )

    arrays = {
        'agentview_image':          arr('agentview_image',          IMG_SIZE, IMG_SIZE, 3),
        'robot0_eye_in_hand_image': arr('robot0_eye_in_hand_image', IMG_SIZE, IMG_SIZE, 3),
        'robot0_eef_pos':           arr('robot0_eef_pos',           3),
        'robot0_eef_quat':          arr('robot0_eef_quat',          4),
        'robot0_gripper_qpos':      arr('robot0_gripper_qpos',      2),
        'action':                   arr('action',                   10),
    }
    return store, arrays


def append_to(arr: zarr.Array, data: np.ndarray) -> None:
    n_old = arr.shape[0]
    n_new = data.shape[0]
    arr.resize(n_old + n_new, *arr.shape[1:])
    arr[n_old:] = data


# ─── Main ─────────────────────────────────────────────────────────────────────

def build(src_dir: pathlib.Path, out_path: pathlib.Path) -> None:
    episodes = sorted(src_dir.glob('episode_*'))
    if not episodes:
        raise FileNotFoundError(f"No episodes found in {src_dir}")
    print(f"Found {len(episodes)} episodes in {src_dir}\n")

    store, arrays = init_zarr(out_path)
    episode_ends  = []
    total         = 0

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
        left_pos  = states['left_pos'][:n]    # (N, 3)
        left_quat = states['left_quat'][:n]   # (N, 4)  [qx,qy,qz,qw]
        gripper   = states['gripper'][:n, :1] # (N, 1)  left gripper value

        # ── Action = [pos(3) + rot6d(6) + gripper(1)] ───────────────────────
        rot6d  = quat_to_rot6d(left_quat)                           # (N, 6)
        action = np.concatenate([left_pos, rot6d, gripper], axis=1) # (N, 10)

        # gripper_qpos duplicated to satisfy config shape [2]
        gripper_qpos = np.repeat(gripper, 2, axis=1)                # (N, 2)

        # ── Append ───────────────────────────────────────────────────────────
        append_to(arrays['agentview_image'],          head_frames)
        append_to(arrays['robot0_eye_in_hand_image'], hand_frames)
        append_to(arrays['robot0_eef_pos'],           left_pos)
        append_to(arrays['robot0_eef_quat'],          left_quat)
        append_to(arrays['robot0_gripper_qpos'],      gripper_qpos)
        append_to(arrays['action'],                   action)

        total += n
        episode_ends.append(total)
        print(f"  {n} frames  (total so far: {total})")

    # ── Episode ends ──────────────────────────────────────────────────────────
    store['meta'].array(
        'episode_ends',
        np.array(episode_ends, dtype=np.int64),
        compressor=COMPRESSOR,
    )

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
    args = parser.parse_args()

    src = pathlib.Path(args.src)
    if not src.exists():
        fallback = pathlib.Path('recordings')
        print(f"'{src}' not found, falling back to '{fallback}'")
        src = fallback

    build(src, pathlib.Path(args.out))
