"""
Converts filtered robot recordings into a Zarr dataset compatible with
diffusion_policy/config/task/dual_arm_ee_image.yaml.

Dual arm (left + right).

Output Zarr schema (16:9, H x W = 144 x 256, matching task/dual_arm_ee_image.yaml's
image_shape [C, H, W] = [3, 144, 256]):
  data/agentview_image            (N, 144, 256, 3)  uint8  — head camera (RGB, top-cropped to 16:9)
  data/robotl_eye_in_hand_image   (N, 144, 256, 3)  uint8  — hand_left camera (RGB)
  data/robotr_eye_in_hand_image   (N, 144, 256, 3)  uint8  — hand_right camera (RGB)
  data/robotl_eef_pos             (N, 9)          float32  — left  EE [pos(3) + rot6d(6)] (raw)
  data/robotr_eef_pos             (N, 9)          float32  — right EE [pos(3) + rot6d(6)] (raw)
  data/robot0_grip                (N, 2)          float32  — [left, right] gripper (0=open 1=closed)
  data/action                     (N, 20)         float32  — L[pos(3)+rot6d(6)+grip(1)] ++ R[pos(3)+rot6d(6)+grip(1)]
  meta/episode_ends               (E,)            int64    — cumulative frame boundaries

Observations (robot*_eef_pos, robot0_grip) stay RAW so they match the camera
ground truth. Actions use Gaussian-smoothed EE trajectories to reduce
teleoperation jerk. Position/rotation layout is pos-first in both obs and action.

Usage:
    python build_dataset.py                          # recordings_filtered/ → dual_ee.zarr
    python build_dataset.py --src recordings --out my.zarr
"""
import argparse
import os
import pathlib
import sys

import cv2
import numcodecs
import numpy as np
import zarr
from scipy.ndimage import gaussian_filter1d

# Single source of truth for the obs/transform pipeline, shared with INFERENCE
# (real_world/build_data.py). Importing the image crop/resize, the quaternion->6D rotation,
# the EE-row layout and the IMG_* / AGENT_CROP_ZOOM constants from there GUARANTEES the training
# data and the deployed observations are built by the exact same code (parity by construction,
# not by hand-synced copies). The only train/deploy gap left is the image codec (mp4 vs JPEG).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from real_world.build_data import (  # noqa: E402
    IMG_H, IMG_W, preprocess_frame, quat_to_rot6d, build_ee_pose,
)

# ─── Configuration ────────────────────────────────────────────────────────────
CHUNK_T       = 100
COMPRESSOR    = numcodecs.Blosc(cname='zstd', clevel=5, shuffle=numcodecs.Blosc.BITSHUFFLE)
SMOOTH_SIGMA  = 1.7   # Gaussian sigma in frames; set to 0 to disable smoothing


# ─── Action smoothing (TRAINING ONLY — obs stays raw, see build_data) ────────

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


def arm_action(pos: np.ndarray, quat: np.ndarray, grip: np.ndarray,
               sigma: float) -> np.ndarray:
    """One arm's action: smoothed [pos(3) + rot6d(6) + grip(1)] → (N,10).

    pos/quat are smoothed; the gripper command is kept raw (≈binary). Training-only: the action
    target is smoothed, but the OBS (eef_pose via build_data.build_ee_pose) stays raw."""
    pos_s  = smooth_pos(pos, sigma)
    quat_s = _sign_fix_and_smooth_quat(quat, sigma)
    return np.concatenate([pos_s, quat_to_rot6d(quat_s), grip], axis=1)  # (N, 10)


# ─── Video loading ────────────────────────────────────────────────────────────

# The head camera is recorded at its native 1280x800 (16:10); the hand cameras at 848x480
# (16:9). The target is 16:9 (IMG_W x IMG_H = 256 x 144). The crop+resize itself is
# build_data.preprocess_frame — THE shared transform used at inference too — so training and
# deployment pixels match (up to mp4-vs-JPEG). The head is too tall for 16:9, so it is cropped
# from the TOP only (keep="bottom"), preserving the workspace at the bottom; the hand cams are
# already ~16:9 and get the default centered few-row crop.

def read_video(path: pathlib.Path, keep: str = "center") -> np.ndarray:
    """Returns (N, IMG_H, IMG_W, 3) uint8 RGB, or empty array if file missing.

    `keep` is the vertical crop anchor passed to the shared preprocess_frame (see build_data):
    "bottom" for the head (drop rows from the TOP), "center" for the already-16:9 wrist cams."""
    if not path.exists():
        return np.empty((0, IMG_H, IMG_W, 3), dtype=np.uint8)
    cap    = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(preprocess_frame(rgb, keep).astype(np.uint8))
    cap.release()
    return np.stack(frames) if frames else np.empty((0, IMG_H, IMG_W, 3), dtype=np.uint8)


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
        'agentview_image':           arr('agentview_image',           IMG_H, IMG_W, 3, dtype='u1'),
        'robotl_eye_in_hand_image':  arr('robotl_eye_in_hand_image',  IMG_H, IMG_W, 3, dtype='u1'),
        'robotr_eye_in_hand_image':  arr('robotr_eye_in_hand_image',  IMG_H, IMG_W, 3, dtype='u1'),
        'robotl_eef_pos':            arr('robotl_eef_pos',            9),
        'robotr_eef_pos':            arr('robotr_eef_pos',            9),
        'robot0_grip':               arr('robot0_grip',               2),
        'action':                    arr('action',                    20),
    }
    return store, arrays


def append_to(arr: zarr.Array, data: np.ndarray) -> None:
    n_old = arr.shape[0]
    n_new = data.shape[0]
    arr.resize((n_old + n_new, *arr.shape[1:]))
    arr[n_old:] = data


# ─── Main ─────────────────────────────────────────────────────────────────────

def build(src_dir: pathlib.Path, out_path: pathlib.Path) -> None:
    episodes = sorted(p for p in src_dir.glob('recording*') if p.is_dir())
    if not episodes:
        raise FileNotFoundError(f"No episodes found in {src_dir}")
    print(f"Found {len(episodes)} episodes in {src_dir}\n")

    store, arrays = init_zarr(out_path)
    episode_ends  = []
    total         = 0

    # accumulated for the smoothing-comparison NPZ (per arm)
    raw_pos_all  = {'l': [], 'r': []}
    raw_quat_all = {'l': [], 'r': []}
    sm_pos_all   = {'l': [], 'r': []}
    sm_quat_all  = {'l': [], 'r': []}

    for ep in episodes:
        print(f"Processing {ep.name}...")

        states = np.load(ep / 'robot_states.npz')
        n      = len(states['timestamps'])

        # ── Images ──────────────────────────────────────────────────────────
        # head: native 16:10 -> crop the TOP to 16:9 (keep the bottom) then downsize (see read_video).
        # hand_*: already ~16:9, so the centered crop is a few rows and the resize keeps aspect.
        head_frames  = read_video(ep / 'cameras' / 'head.mp4', keep="bottom")
        handl_frames = read_video(ep / 'cameras' / 'hand_left.mp4')
        handr_frames = read_video(ep / 'cameras' / 'hand_right.mp4')

        # Guard against minor frame-count drift between videos and NPZ
        n = min(n, len(head_frames), len(handl_frames), len(handr_frames))
        if n == 0:
            print(f"  SKIP — no usable frames")
            continue

        head_frames  = head_frames[:n]
        handl_frames = handl_frames[:n]
        handr_frames = handr_frames[:n]

        # ── Arm state ─────────────────────────────────────────────────────────
        left_pos   = states['left_pos'][:n]      # (N, 3)
        left_quat  = states['left_quat'][:n]     # (N, 4)  [qx,qy,qz,qw]
        right_pos  = states['right_pos'][:n]     # (N, 3)
        right_quat = states['right_quat'][:n]    # (N, 4)
        grip       = states['gripper'][:n, :2]   # (N, 2)  [left, right]
        left_grip  = grip[:, :1]                 # (N, 1)
        right_grip = grip[:, 1:2]                # (N, 1)

        # ── Observations (RAW) ────────────────────────────────────────────────
        # build_ee_pose is the SHARED obs layout (real_world/build_data.py) — the live env /
        # replay build their obs rows from the same function, so train == deploy by construction.
        robotl_eef_pos = build_ee_pose(left_pos,  left_quat)   # (N, 9)
        robotr_eef_pos = build_ee_pose(right_pos, right_quat)  # (N, 9)

        # ── Action = L[pos+rot6d+grip] ++ R[pos+rot6d+grip] = (N, 20) ─────────
        action_l = arm_action(left_pos,  left_quat,  left_grip,  SMOOTH_SIGMA)   # (N,10)
        action_r = arm_action(right_pos, right_quat, right_grip, SMOOTH_SIGMA)   # (N,10)
        action   = np.concatenate([action_l, action_r], axis=1)                 # (N,20)

        for tag, pos, quat in (('l', left_pos, left_quat), ('r', right_pos, right_quat)):
            raw_pos_all[tag].append(pos)
            raw_quat_all[tag].append(quat)
            sm_pos_all[tag].append(smooth_pos(pos, SMOOTH_SIGMA))
            sm_quat_all[tag].append(_sign_fix_and_smooth_quat(quat, SMOOTH_SIGMA))

        # ── Append ────────────────────────────────────────────────────────────
        append_to(arrays['agentview_image'],          head_frames)
        append_to(arrays['robotl_eye_in_hand_image'], handl_frames)
        append_to(arrays['robotr_eye_in_hand_image'], handr_frames)
        append_to(arrays['robotl_eef_pos'],           robotl_eef_pos)
        append_to(arrays['robotr_eef_pos'],           robotr_eef_pos)
        append_to(arrays['robot0_grip'],              grip)
        append_to(arrays['action'],                   action)

        total += n
        episode_ends.append(total)
        print(f"  {n} frames  (total so far: {total})")

    # ── Smoothing comparison NPZ ──────────────────────────────────────────────
    npz_path = out_path.with_suffix('.smooth_debug.npz')
    np.savez(
        npz_path,
        raw_pos_l    = np.concatenate(raw_pos_all['l'],  axis=0),
        raw_quat_l   = np.concatenate(raw_quat_all['l'], axis=0),
        smooth_pos_l = np.concatenate(sm_pos_all['l'],   axis=0),
        smooth_quat_l= np.concatenate(sm_quat_all['l'],  axis=0),
        raw_pos_r    = np.concatenate(raw_pos_all['r'],  axis=0),
        raw_quat_r   = np.concatenate(raw_quat_all['r'], axis=0),
        smooth_pos_r = np.concatenate(sm_pos_all['r'],   axis=0),
        smooth_quat_r= np.concatenate(sm_quat_all['r'],  axis=0),
        episode_ends = np.array(episode_ends, dtype=np.int64),
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
    parser.add_argument('--out', default='dual_ee.zarr',
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
