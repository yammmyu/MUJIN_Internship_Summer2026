"""Observation collector for HumanoidEnv (the producer-side obs state).

Extracted from HumanoidEnv. An ObsCollector owns the rolling dual-arm EE-pose / gripper obs
buffers, the freshness/staleness tracking (H2), and the ``get_obs`` snapshot + ``inf_ready``
readiness — everything the policy observation is built from. It holds its own lock.

The env's collect thread still takes the raw per-tick SDK reads (it owns the robot/controller)
and feeds them here via ``ingest``; ``get_obs`` pulls the camera frames from the CameraHub it is
handed. Camera ROLES (which camera is agentview vs each wrist) are injected so this module stays
decoupled from the SDK camera set and the env's re-exported role constants.

Scope is deliberately observation-only: the control path (ramp-in / release / splice) takes its
own fresh synchronous reads because it needs the arm's pose AT command time, not a value up to one
tick old.
"""

import copy
import threading
import time

import numpy as np

from real_world.build_data import build_ee_pose_row, build_grip_row


def extract_pose(frame):
    """One motion-status link frame -> (x, y, z, qx, qy, qz, qw). Same extraction the recorder and
    the right-EE obs both use (shared so recorded rows match the obs source exactly)."""
    pos = frame['position']
    quat = frame['orientation']['quaternion']
    return (
        pos['x'], pos['y'], pos['z'],
        quat['x'], quat['y'], quat['z'], quat['w'],
    )


class ObsCollector:
    """Producer-side observation buffers + snapshot for one policy (dual_arm_ee_image)."""

    def __init__(self, solver, solver_r, inference_cameras, agent_camera, hand_left, hand_right,
                 stale_timeout):
        self.solver = solver                        # LEFT-arm model, for live-joint EE FK
        self.solver_r = solver_r                    # RIGHT-arm model, for live-joint EE FK (mirror)
        self.inference_cameras = list(inference_cameras)
        self.agent_camera = agent_camera
        self.hand_left = hand_left
        self.hand_right = hand_right
        self.stale_timeout = stale_timeout

        self._lock = threading.Lock()
        self._log_ts = {}                           # warmup-print throttle

        # Dual-arm EE-pose obs (policy input), rolling [row_{t-1}, row_t], each row 9-dim
        # [pos(3) + rot6d(6)] in the training robot*_eef_pos layout: left from calibrated FK on the
        # live joints, right from the firmware status frame.
        self._last_two_robotl_eef = None
        self._last_two_robotr_eef = None
        # robot0_grip obs: rolling [g_{t-1}, g_t], each = [left, right] RAW gripper (2-dim).
        self._last_two_grip = None
        self._obs_timestamp = 0.0
        self._obs_row_id = 0

        # Liveness/staleness (H2): _fresh_mono advances ONLY when the EE pose or a policy camera
        # frame ACTUALLY changes, so a frozen feed is detectable even though every tick is "recent".
        self._fresh_mono = 0.0
        self._last_ee_sig = None
        self._last_cam_sig = {}
        self._firmware_has_error = False

    def _throttled(self, key, interval=1.0):
        """True at most once per `interval` seconds for `key` (rate-limits warmup prints)."""
        now = time.monotonic()
        if now - self._log_ts.get(key, 0.0) >= interval:
            self._log_ts[key] = now
            return True
        return False

    def _left_ee_fk(self, arm7):
        """Left EE obs row [pos(3) + rot6d(6)] (9-dim) via pinocchio FK on the LIVE arm joints.

        The obs EE comes from FK, NOT status['frames']['arm_left_link7']: the firmware parks its
        EE/FK estimator while an ABS_JOINT trajectory executes (auto-inference), so that frame
        FREEZES even though arm_joint_states() stays live. FK from the live joints keeps the EE
        tracking the arm and matches the training EE frame (~3mm vs firmware FK, calibrated via
        fk_calibration.json — recordings were collected under EE-space teleop while the firmware
        FK was live). The gripper is published separately (robot0_grip), not in this row."""
        pos, quat = self.solver.m.fk(np.asarray(arm7, dtype=np.float64))
        return build_ee_pose_row(pos, quat)

    def _right_ee_fk(self, arm7):
        """Right EE obs row [pos(3) + rot6d(6)] via pinocchio FK on the LIVE right joints — the mirror
        of _left_ee_fk. Uses FK, NOT status['frames']['arm_right_link7']: now that the dual-arm policy
        DRIVES the right arm, the firmware parks its right-EE/FK estimator while that ABS_JOINT
        trajectory executes, FREEZING the status frame at the pre-motion pose (arm_joint_states stays
        live). That stale frame made the policy re-plan the right arm from an OLD pose ('snaps back').
        solver_r carries its own calibrated base_offset (fk_calibration_right.json), so this matches the
        training right-EE frame the same way the left does."""
        pos, quat = self.solver_r.m.fk(np.asarray(arm7, dtype=np.float64))
        return build_ee_pose_row(pos, quat)

    @staticmethod
    def _frame_sig(frame):
        """Cheap change signature for a camera frame (H2): a coarse subsample, not the whole
        image, so per-tick freeze detection stays light. None frame -> None."""
        if frame is None:
            return None
        a = np.asarray(frame)
        return a[::64, ::64].tobytes() if a.ndim >= 2 else a.tobytes()

    def ingest(self, now, now_mono, status, grip, arm14, frames, current_row_id):
        """Fold one producer tick into the obs buffers. The env's collect thread passes the raw
        reads it already took: `status` (motion status), `grip` ([left,right] raw), `arm14` (both
        arms, or None), this tick's camera `frames` ({name: [prev,cur]}), and the master row id the
        obs is anchored to. Latest-wins; None inputs are simply skipped."""
        # BOTH EE rows from FK on the LIVE joints (9-dim [pos(3) + rot6d(6)]). The right arm is now
        # commanded by the dual-arm policy, so — like the left — its firmware EE frame parks during the
        # ABS_JOINT trajectory; FK from the live joints keeps the right EE tracking the real arm instead
        # of freezing at the pre-motion pose (which made the policy plan the right arm from a stale pose).
        robotl_eef = self._left_ee_fk(arm14[:7]) if arm14 is not None else None
        robotr_eef = self._right_ee_fk(arm14[7:]) if arm14 is not None else None
        grip_state = build_grip_row(grip[0], grip[1]) if grip is not None else None

        # Freshness (H2): did the EE pose or a policy camera frame actually CHANGE? Only then bump
        # _fresh_mono, so a frozen feed stops advancing it even though `now` keeps moving.
        changed = False
        ee_sig = tuple(np.round(robotl_eef, 6)) if robotl_eef is not None else None
        if ee_sig is not None and ee_sig != self._last_ee_sig:
            changed = True
        cam_sigs = {n: self._frame_sig(frames[n][-1]) for n in self.inference_cameras
                    if n in frames}
        for n, sig in cam_sigs.items():
            if sig != self._last_cam_sig.get(n):
                changed = True
        fw_error = bool(status and status.get('error', {}).get('has_error'))

        with self._lock:
            self._obs_timestamp = now
            self._obs_row_id = current_row_id       # master-ID this obs is anchored to
            self._firmware_has_error = fw_error
            if ee_sig is not None:
                self._last_ee_sig = ee_sig
            self._last_cam_sig.update(cam_sigs)
            if changed or self._fresh_mono == 0.0:
                self._fresh_mono = now_mono
            if robotl_eef is not None:
                self._last_two_robotl_eef = ([self._last_two_robotl_eef[-1], robotl_eef]
                                             if self._last_two_robotl_eef else [robotl_eef, robotl_eef])
            if robotr_eef is not None:
                self._last_two_robotr_eef = ([self._last_two_robotr_eef[-1], robotr_eef]
                                             if self._last_two_robotr_eef else [robotr_eef, robotr_eef])
            if grip_state is not None:
                self._last_two_grip = ([self._last_two_grip[-1], grip_state]
                                       if self._last_two_grip else [grip_state, grip_state])

    def inf_ready(self, camhub):
        """Ready for inference once the three policy cameras + both EE states are populated."""
        frames_ready = camhub.have_frames(self.inference_cameras)
        with self._lock:
            return frames_ready and self._last_two_robotl_eef is not None \
                and self._last_two_robotr_eef is not None

    def get_obs(self, camhub):
        """Time-aligned snapshot for one inference, or None if not ready yet.

        Returns the dual_arm_ee_image obs (see build_data.build_predict_request):
            agent_imgs:     [head_{t-1}, head_t]
            handl_imgs:     [hand_left_{t-1},  hand_left_t]
            handr_imgs:     [hand_right_{t-1}, hand_right_t]
            robotl_eef_pos: [row_{t-1}, row_t]   each [pos(3) + rot6d(6)] (9)
            robotr_eef_pos: [row_{t-1}, row_t]   each [pos(3) + rot6d(6)] (9)
            robot0_grip:    [g_{t-1}, g_t]       each [left, right] (2)
            timestamp:  float (seconds)
            age:        seconds since the EE pose / cameras last actually changed (H2)
            stale:      True if age > stale_timeout or the firmware reports an error
            firmware_error: bool
        """
        # Keep the three policy cameras warm while inference polls get_obs().
        for cam in self.inference_cameras:
            camhub.request(cam)
        # Camera frames snapshot (hub lock) — deepcopied [prev, cur] pairs, or None per camera while
        # warming up. Taken before our lock; obs buffers are latest-wins, so a sub-tick skew between
        # the frame snapshot and the EE/grip snapshot below is acceptable.
        imgs = camhub.snapshot(self.inference_cameras)
        if any(imgs[n] is None for n in self.inference_cameras):
            if self._throttled("warmup_cam", 2.0):
                missing = [n for n in self.inference_cameras if imgs[n] is None]
                print(f"[ObsCollector] warming up: waiting for cameras {missing}…")
            return None
        with self._lock:
            if (self._last_two_robotl_eef is None or self._last_two_robotr_eef is None
                    or self._last_two_grip is None):
                if self._throttled("warmup_ee", 2.0):
                    print("[ObsCollector] warming up: waiting for EE / gripper states…")
                return None
            age = (time.monotonic() - self._fresh_mono) if self._fresh_mono else float('inf')
            stale = age > self.stale_timeout or self._firmware_has_error
            return {
                'agent_imgs': imgs[self.agent_camera],
                'handl_imgs': imgs[self.hand_left],
                'handr_imgs': imgs[self.hand_right],
                'robotl_eef_pos': copy.deepcopy(self._last_two_robotl_eef),
                'robotr_eef_pos': copy.deepcopy(self._last_two_robotr_eef),
                'robot0_grip': copy.deepcopy(self._last_two_grip),
                'timestamp': self._obs_timestamp,
                'step_id': self._obs_row_id,    # master row-ID for alignment (see ensemble/ingest)
                'age': age,
                'stale': bool(stale),
                'firmware_error': bool(self._firmware_has_error),
            }
