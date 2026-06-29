"""Feed a recorded episode to the inference loop in place of live cameras (SDK-free).

`RecordedObsSource` decodes a recording's head + hand_left + hand_right videos and
reconstructs the dual-arm EE/gripper obs from robot_states.npz, exposing the same
`get_obs()` / `inf_ready` surface the live HumanoidEnv does. The offline sim runner injects
it into InferenceController so the policy runs on recorded perception, with predicted actions
driven into the sim.

Imports only numpy + cv2 (+ build_data), so it loads without `a2d_sdk`.
"""

import os
import time

import cv2
import numpy as np

# Same proprioception row layout as the live env (training zarr layout, defined once).
from real_world.build_data import build_ee_pose_row, build_grip_row

# Role mapping (mirrors humanoid_env AGENT_CAMERA / HAND_CAMERA_*).
AGENT_VIDEO = "head.mp4"            # -> agent_imgs
HAND_LEFT_VIDEO = "hand_left.mp4"   # -> handl_imgs
HAND_RIGHT_VIDEO = "hand_right.mp4"  # -> handr_imgs


class RecordedObsSource:
    """Replays a recording's obs at the inference rate.

    get_obs() returns the same dict shape as HumanoidEnv.get_obs() (dual_arm_ee_image):
        agent_imgs:     [head_{t-1}, head_t]   (RGB, to match encode_image)
        handl_imgs:     [hand_left_{t-1},  hand_left_t]
        handr_imgs:     [hand_right_{t-1}, hand_right_t]
        robotl_eef_pos: [row_{t-1}, row_t]   each [pos(3) + rot6d(6)]  (9 dims)
        robotr_eef_pos: [row_{t-1}, row_t]   each [pos(3) + rot6d(6)]  (9 dims)
        robot0_grip:    [g_{t-1}, g_t]       each [left, right]        (2 dims)
        timestamp:      float (wall clock; always advances so inference doesn't dedup)
        step_id:        int (recording row index — the master-id anchor for ensemble/ingest)
    Returns None and sets `done` once the videos are exhausted.
    """

    def __init__(self, recording, recordings_dir, record_hz=30, inference_hz=10):
        rec_dir = os.path.join(recordings_dir, recording)
        npz_path = os.path.join(rec_dir, "robot_states.npz")
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"no recording at {npz_path}")
        npz = np.load(npz_path)
        left_pos = npz["left_pos"].astype(np.float64)      # (N, 3)
        left_quat = npz["left_quat"].astype(np.float64)    # (N, 4) xyzw
        right_pos = npz["right_pos"].astype(np.float64)    # (N, 3)
        right_quat = npz["right_quat"].astype(np.float64)  # (N, 4) xyzw
        grip = npz["gripper"].astype(np.float64)[:, :2]    # (N, 2) [left, right]
        self.arm_joints = npz["arm_joints"].astype(np.float64)  # (N, 14) — for IK seeding
        # Same row builders as the live env, so replay obs are byte-for-byte the training layout
        # (plain Python floats -> JSON-serialisable for the policy request).
        n = len(left_pos)
        self.robotl_eef = [build_ee_pose_row(left_pos[i], left_quat[i]) for i in range(n)]
        self.robotr_eef = [build_ee_pose_row(right_pos[i], right_quat[i]) for i in range(n)]
        self.grips = [build_grip_row(grip[i, 0], grip[i, 1]) for i in range(n)]
        self.n = n

        self.agent_cap = cv2.VideoCapture(os.path.join(rec_dir, "cameras", AGENT_VIDEO))
        self.handl_cap = cv2.VideoCapture(os.path.join(rec_dir, "cameras", HAND_LEFT_VIDEO))
        self.handr_cap = cv2.VideoCapture(os.path.join(rec_dir, "cameras", HAND_RIGHT_VIDEO))
        if not (self.agent_cap.isOpened() and self.handl_cap.isOpened()
                and self.handr_cap.isOpened()):
            raise FileNotFoundError(
                f"could not open {AGENT_VIDEO}/{HAND_LEFT_VIDEO}/{HAND_RIGHT_VIDEO} "
                f"under {rec_dir}/cameras")

        self.stride = max(1, round(record_hz / inference_hz))  # frames to advance per obs
        self.cursor = 0           # frames consumed from the videos
        self.done = False
        self._cur = None          # last (agent, handl, handr, robotl, robotr, grip) emitted

    @property
    def inf_ready(self):
        return not self.done

    def seed_left_joints(self):
        """The recording's first left-arm joints (IK warm-start / sim seed)."""
        return self.arm_joints[0, :7].copy()

    def _read_rgb(self, cap):
        ok, frame = cap.read()
        if not ok:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   # encode_image expects RGB

    def get_obs(self):
        if self.done:
            return None
        # Advance `stride` frames; the last triple read becomes the new 'cur'.
        agent = handl = handr = None
        idx = self.cursor
        for _ in range(self.stride):
            a = self._read_rgb(self.agent_cap)
            hl = self._read_rgb(self.handl_cap)
            hr = self._read_rgb(self.handr_cap)
            if a is None or hl is None or hr is None:
                self.done = True
                return None
            agent, handl, handr = a, hl, hr
            idx += 1
        si = min(idx - 1, self.n - 1)
        cur = (agent, handl, handr, self.robotl_eef[si], self.robotr_eef[si], self.grips[si])
        prev = self._cur if self._cur is not None else cur   # first obs: [s0, s0]
        self._cur = cur
        self.cursor = idx
        return {
            'agent_imgs': [prev[0], cur[0]],
            'handl_imgs': [prev[1], cur[1]],
            'handr_imgs': [prev[2], cur[2]],
            'robotl_eef_pos': [prev[3], cur[3]],
            'robotr_eef_pos': [prev[4], cur[4]],
            'robot0_grip': [prev[5], cur[5]],
            'timestamp': time.time(),
            # Master row id for alignment: the recording row index of this obs (the recorded frames
            # ARE the policy-row timeline). Replaces wall-clock as the ensemble/ingest anchor. In the
            # live env this comes from the robot's execution clock; here it's the replay cursor.
            'step_id': si,
        }

    def close(self):
        self.agent_cap.release()
        self.handl_cap.release()
        self.handr_cap.release()
