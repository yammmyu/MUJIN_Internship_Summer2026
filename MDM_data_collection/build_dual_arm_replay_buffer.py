"""Build OUR robot recordings -> diffusion_policy dual-arm JOINT ReplayBuffer (zarr).

Concrete `iter_demos()` for this repo's recording format, matching the shape_meta in
diffusion_policy/config/task/dual_arm_grasp_image.yaml (trained by
train_diffusion_unet_ddim_dual_arm_workspace.yaml). THREE cameras stored at 224x224 (the
workspace random-crops each to 200 at train time), plus per-arm joint state and a 16-d action:

    head_img       : (T, S, S, 3) uint8    # head camera (global workspace view), square SxS, RGB
    left_wrist_img : (T, S, S, 3) uint8    # hand_left  camera
    right_wrist_img: (T, S, S, 3) uint8    # hand_right camera
    left_joint     : (T, 8)       float32  # 7 arm joints + 1 gripper bit (0=open, 1=closed)
    right_joint    : (T, 8)       float32
    action         : (T, 16)      float32  # next-state target (one-step shift):
        #   [0:7]   left  arm joint targets
        #   [7]     left  gripper command (0/1)
        #   [8:15]  right arm joint targets
        #   [15]    right gripper command

`--img_size` MUST match the task yaml's image_shape (default 224). This is the JOINT-space sibling
of build_dataset.py (the EE-pose dataset) — same recordings, but absolute joint targets + the
three 224 camera views the dual-arm joint task expects.

Our recording layout (per episode dir, produced by MDM_data_collection/robot_data_collect.py):
    <raw_root>/recordingNNN/
        robot_states.npz    # arm_joints (T,14) [L:0:7, R:7:14], gripper (T,2) [L,R] raw, timestamps
        cameras/head.mp4  cameras/hand_left.mp4  cameras/hand_right.mp4   # 30 Hz, same T
        metadata.json          # {fps, n_frames, camera_names}

Action, like the reference, is derived by a one-step shift (teleop logs state, not a separate
action): action[t] = state[t+1], so each episode loses its last frame (T -> T-1). With --fps below
the shift happens on the DOWNSAMPLED sequence, so an action row spans `stride` recorded frames.

Writes the ReplayBuffer zarr format DIRECTLY (no diffusion_policy import, no zarr-v2 env needed —
see the writer note below), so it runs in the same environment you record/build in.

Run:
    python MDM_data_collection/build_dual_arm_replay_buffer.py \
        --raw_root MDM_data_collection/recordings \
        --out      data/dual_arm_grasp/replay_buffer.zarr \
        --img_size 224 --fps 30
"""

import argparse
import os
import pathlib
from typing import Dict, Iterator

import cv2
import numcodecs
import numpy as np
import zarr

# Gripper binarisation threshold — SOURCE OF TRUTH is real_world/humanoid_env.py
# (GRIPPER_CLOSE_THRESH). Hardcoded here to keep this builder dependency-light (no pinocchio / SDK
# import just to read one constant). Keep in sync if that value ever changes.
GRIPPER_CLOSE_THRESH = 10.0

SOURCE_HZ = 30          # recordings are captured at 30 Hz (metadata.json "fps")

# We write the ReplayBuffer on-disk layout DIRECTLY (data/<key> concatenated across episodes +
# meta/episode_ends, zarr_format=2) instead of importing diffusion_policy.ReplayBuffer. That class
# targets zarr 2.x (zarr.DirectoryStore / zarr.copy_store), so importing it forces a zarr-v2 env;
# writing the group ourselves works with any zarr (incl. v3 via zarr_format=2) and produces the
# IDENTICAL format that MDM_data_collection/build_dataset.py already trains on. Same helpers as there.
CHUNK_T    = 100
COMPRESSOR = numcodecs.Blosc(cname='zstd', clevel=5, shuffle=numcodecs.Blosc.BITSHUFFLE)


def _append(store: zarr.Group, key: str, data: np.ndarray) -> None:
    """Append (n, ...) `data` to data/<key>, creating the resizable array on first use."""
    data = np.ascontiguousarray(data)
    if key in store['data']:
        a = store['data'][key]
    else:
        a = store['data'].empty(
            name=key, shape=(0, *data.shape[1:]), chunks=(CHUNK_T, *data.shape[1:]),
            dtype=data.dtype, zarr_format=2, compressor=COMPRESSOR)
    n0 = a.shape[0]
    a.resize((n0 + len(data), *a.shape[1:]))
    a[n0:] = data


def _read_video_square(path: pathlib.Path, img_size: int) -> np.ndarray:
    """Decode an mp4 to (N, img_size, img_size, 3) uint8 RGB (square resize, matching the
    reference's iter_demos). Returns an empty array if the file is missing."""
    empty = np.empty((0, img_size, img_size, 3), dtype=np.uint8)
    if not path.exists():
        return empty
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if rgb.shape[:2] != (img_size, img_size):
            rgb = cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_AREA)
        frames.append(rgb.astype(np.uint8))
    cap.release()
    return np.stack(frames) if frames else empty


def _grip_bit(raw: np.ndarray) -> np.ndarray:
    """Raw gripper reading (~[0, 85]) -> binary closed bit (1 = closed), column vector (N, 1)."""
    return (raw >= GRIPPER_CLOSE_THRESH).astype(np.float32).reshape(-1, 1)


def iter_demos(raw_root: str, img_size: int, stride: int) -> Iterator[Dict[str, np.ndarray]]:
    """Yield one dict per recording: left/right wrist imgs, left/right joint state (7+grip), and
    the one-step-shift action (16). Episodes shorter than 2 usable (downsampled) frames are skipped."""
    root = pathlib.Path(raw_root)
    episode_dirs = sorted(
        d for d in root.iterdir()
        if d.is_dir() and (d / 'robot_states.npz').exists())
    if not episode_dirs:
        raise SystemExit(f"no recordings (dirs with robot_states.npz) found under {root}")

    for ep in episode_dirs:
        states = np.load(ep / 'robot_states.npz')
        arm_joints = states['arm_joints'].astype(np.float32)   # (T, 14) [L:0:7, R:7:14]
        grip = states['gripper'].astype(np.float32)            # (T, 2)  [L, R] raw
        n = len(arm_joints)

        head = _read_video_square(ep / 'cameras' / 'head.mp4', img_size)
        left_wrist = _read_video_square(ep / 'cameras' / 'hand_left.mp4', img_size)
        right_wrist = _read_video_square(ep / 'cameras' / 'hand_right.mp4', img_size)

        # Align lengths (videos can drift a frame from the NPZ), then downsample to the target rate.
        n = min(n, len(head), len(left_wrist), len(right_wrist))
        if n == 0:
            print(f"  SKIP {ep.name}: no usable frames")
            continue
        sel = np.arange(0, n, stride)
        if len(sel) < 2:
            print(f"  SKIP {ep.name}: too short after --fps downsample ({len(sel)} frames)")
            continue

        arm_joints = arm_joints[sel]
        grip = grip[sel]
        head = head[sel]
        left_wrist = left_wrist[sel]
        right_wrist = right_wrist[sel]

        # State per arm: 7 joints + 1 binarised gripper bit -> (N, 8).
        left_state = np.concatenate([arm_joints[:, 0:7], _grip_bit(grip[:, 0])], axis=1)
        right_state = np.concatenate([arm_joints[:, 7:14], _grip_bit(grip[:, 1])], axis=1)

        # Action = next state (both arms), one-step shift; drop the final frame that has no next.
        state16 = np.concatenate([left_state, right_state], axis=1)   # (N, 16) [L8 ++ R8]
        action = state16[1:].copy()                                  # (N-1, 16)
        left_state = left_state[:-1]
        right_state = right_state[:-1]
        head = head[:-1]
        left_wrist = left_wrist[:-1]
        right_wrist = right_wrist[:-1]

        yield {
            'head_img':        head,
            'left_wrist_img':  left_wrist,
            'right_wrist_img': right_wrist,
            'left_joint':      left_state,
            'right_joint':     right_state,
            'action':          action,
        }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--raw_root',
                    default=str(pathlib.Path(__file__).parent / 'recordings'),
                    help='Directory of recordingNNN/ dirs (default: MDM_data_collection/recordings).')
    ap.add_argument('--out', required=True,
                    help='Output zarr path, e.g. data/dual_arm_joint/replay_buffer.zarr')
    ap.add_argument('--img_size', type=int, default=224,
                    help='Square camera-image size; MUST match image_shape in the task yaml '
                         '(dual_arm_grasp_image.yaml uses 224).')
    ap.add_argument('--fps', type=int, default=SOURCE_HZ,
                    help=f'Output row rate (recordings are {SOURCE_HZ} Hz). A model trained at N Hz '
                         f'must be deployed at RECORD_HZ==N. Default: native {SOURCE_HZ}.')
    args = ap.parse_args()

    if args.fps <= 0 or SOURCE_HZ % args.fps != 0:
        raise SystemExit(f"--fps must be a positive divisor of {SOURCE_HZ} (got {args.fps})")
    stride = SOURCE_HZ // args.fps

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(args.out):
        raise SystemExit(f"{args.out} already exists. Delete it or pick another path.")

    # ReplayBuffer-format group: data/<key> arrays (grown per episode) + meta/episode_ends.
    store = zarr.open_group(args.out, mode='w', zarr_format=2)
    store.require_group('data')
    store.require_group('meta')

    n_eps = 0
    total = 0
    episode_ends = []
    for ep in iter_demos(args.raw_root, args.img_size, stride):
        for key, value in ep.items():
            _append(store, key, value)
        n_eps += 1
        total += len(ep['action'])
        episode_ends.append(total)
        print(f"  episode {n_eps}: T={len(ep['action'])}, total steps={total}")

    ep_ends = np.array(episode_ends, dtype=np.int64)
    meta = store['meta'].empty(name='episode_ends', shape=ep_ends.shape,
                               chunks=ep_ends.shape, dtype=np.int64,
                               zarr_format=2, compressor=COMPRESSOR)
    meta[:] = ep_ends

    print(f"Done. Wrote {n_eps} episodes / {total} steps to {args.out} "
          f"(img {args.img_size}x{args.img_size}, {args.fps} Hz, stride {stride}).")


if __name__ == '__main__':
    main()
