"""Feed a recorded episode to the inference loop in place of live cameras (SDK-free).

`RecordedObsSource` decodes a recording's head + hand_left videos and reconstructs the EE
state from robot_states.npz, exposing the same `get_obs()` / `inf_ready` surface the live
HumanoidEnv does. The offline sim runner injects it into InferenceController so the policy
runs on recorded perception, with predicted actions driven into the sim.

Imports only numpy + cv2, so it loads without `a2d_sdk`.
"""

import os
import time

import cv2
import numpy as np

# Role mapping (mirrors humanoid_env AGENT_CAMERA / HAND_CAMERA).
AGENT_VIDEO = "head.mp4"        # -> agent_imgs
HAND_VIDEO = "hand_left.mp4"    # -> hand_imgs


class RecordedObsSource:
    """Replays a recording's obs at the inference rate.

    get_obs() returns the same dict shape as HumanoidEnv.get_obs():
        agent_imgs:  [head_{t-1}, head_t]   (RGB, to match encode_image)
        hand_imgs:   [hand_left_{t-1}, hand_left_t]
        state:       [s_{t-1}, s_t]   each [pos(3), quat xyzw(4), grip(1)]  (8 dims)
        joint_state: [j_{t-1}, j_t]   each [7 left-arm joints (rad), grip]  (8 dims)
        timestamp:   float (wall clock; always advances so inference doesn't dedup)
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
        grip = npz["gripper"].astype(np.float64)[:, 0]     # (N,)
        self.arm_joints = npz["arm_joints"].astype(np.float64)  # (N, 14) — for IK seeding
        # Plain Python floats (not np.float64) so the state is JSON-serialisable for the
        # policy request, matching the live env's SDK-derived floats.
        self.states = [[float(v) for v in (*left_pos[i], *left_quat[i], grip[i])]
                       for i in range(len(left_pos))]
        # robot0_left_joint obs: [7 LEFT-arm joints (rad), gripper], mirroring the live env's
        # joint_state (arm14[:7] + grip). Required by InferenceController (obs['joint_state']).
        self.joint_states = [[float(v) for v in self.arm_joints[i, :7]] + [float(grip[i])]
                             for i in range(len(left_pos))]
        self.n = len(self.states)

        self.agent_cap = cv2.VideoCapture(os.path.join(rec_dir, "cameras", AGENT_VIDEO))
        self.hand_cap = cv2.VideoCapture(os.path.join(rec_dir, "cameras", HAND_VIDEO))
        if not (self.agent_cap.isOpened() and self.hand_cap.isOpened()):
            raise FileNotFoundError(
                f"could not open {AGENT_VIDEO}/{HAND_VIDEO} under {rec_dir}/cameras")

        self.stride = max(1, round(record_hz / inference_hz))  # frames to advance per obs
        self.cursor = 0           # frames consumed from the videos
        self.done = False
        self._cur = None          # last (agent, hand, state) emitted

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
        # Advance `stride` frames; the last pair read becomes the new 'cur'.
        agent = hand = None
        idx = self.cursor
        for _ in range(self.stride):
            a = self._read_rgb(self.agent_cap)
            h = self._read_rgb(self.hand_cap)
            if a is None or h is None:
                self.done = True
                return None
            agent, hand = a, h
            idx += 1
        si = min(idx - 1, self.n - 1)
        cur = (agent, hand, self.states[si], self.joint_states[si])
        prev = self._cur if self._cur is not None else cur   # first obs: [s0, s0]
        self._cur = cur
        self.cursor = idx
        return {
            'agent_imgs': [prev[0], cur[0]],
            'hand_imgs': [prev[1], cur[1]],
            'state': [prev[2], cur[2]],
            'joint_state': [prev[3], cur[3]],
            'timestamp': time.time(),
        }

    def close(self):
        self.agent_cap.release()
        self.hand_cap.release()
