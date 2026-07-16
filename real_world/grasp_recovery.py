"""Standalone CCDP-style grasp-failure recovery for the dual-arm policy.

Self-contained: loads the wrist-camera grasp detector (data/grasp_detector/detector.pt,
trained in the diffusion_policy repo) and drives failure recovery, with NO GUI coupling
and only three small call-ins from InferenceController. Needs only torch + torchvision +
the .pt checkpoint on the client machine (detection runs LOCALLY, not on the policy server).

WHAT IT DOES
------------
Watches the RIGHT gripper command in each policy chunk. On an open->close transition it
arms a grasp attempt and, after `settle_sec` (default 5 s — deliberately late so the check
never perturbs the grasp itself), runs the detector on the right wrist frame
(obs['handr_imgs'][-1]). If the gripper closed on nothing it enters RECOVERY:

    1. clear the robot queue (drop the policy's "I grasped, now lift" rows)
    2. OPEN the right gripper, then move BOTH arms to a fixed HOME joint pose by ABSOLUTE joint
       angles (env.command_gripper + env.move_to_joints) -- a synchronous, velocity-bounded,
       E-stop-aware move; no EE-space lift / IK / streaming.
    3. hand control back -> the stochastic policy re-plans a fresh approach from home on its own.

No failure demos, no high-level planner: each grasp attempt is a sub-problem and the
recovery simply resets to home + open so the stochastic policy can try again.

INTEGRATION (see the three call-ins in InferenceController):
    __init__: self.recovery = GraspRecoveryMonitor("data/grasp_detector/detector.pt",
                                                    open_grip=..., closed_grip_min=...)
    _run_inference (after obs, before post_predict):
        if rec and rec.maybe_start(env, obs): return True  # miss -> recovery (open + home) done inline
    _run_inference (after `action`):
        if rec: rec.note_action(action)                    # track the grip command



"""

import logging
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18

from real_world.retreat import retreat_to_nearest

log = logging.getLogger(__name__)

# 20-col dual_arm_ee_image action row: L[pos3,rot6d6,grip1] ++ R[pos3,rot6d6,grip1]
L_EE = slice(0, 9)      # left [pos3, rot6d6]  (held during retreat)
L_GRIP = 9
R_POS = slice(10, 13)   # right pos3
R_ROT = slice(13, 19)   # right rot6d6
R_GRIP = 19

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class _Detector:
    """Inlined GraspDetector: ResNet18 head on the fixed finger-gap ROI crop."""

    def __init__(self, ckpt_path, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        # weights_only=False: the checkpoint stores plain python values (roi/threshold/size) next to
        # the state_dict, which torch>=2.6's weights_only default rejects. It's our own trusted file.
        ck = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.roi = tuple(ck["roi"])
        self.empty_idx = ck["empty_idx"]
        self.threshold = ck["threshold"]
        m = resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, 2)
        m.load_state_dict(ck["state_dict"])
        self.model = m.to(self.device).eval()
        self.tf = transforms.Compose([
            transforms.Resize((ck["size"], ck["size"])),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    @torch.no_grad()
    def p_empty(self, frame) -> float:
        if not isinstance(frame, Image.Image):
            frame = Image.fromarray(np.asarray(frame)[..., :3].astype(np.uint8))
        x = self.tf(frame.convert("RGB").crop(self.roi)).unsqueeze(0).to(self.device)
        return torch.softmax(self.model(x), dim=1)[0, self.empty_idx].item()


class GraspRecoveryMonitor:
    def __init__(self, detector_path, *,
                 settle_sec=4.0,
                 # Pre-grasp APPROACH waypoints (n, 14) [left7, right7] the recovery retreats to (open
                 # gripper + move BOTH arms), ordered start -> pre-grasp. Passed in by the controller
                 # (real_world/config/retreat_waypoints.json). The retreat picks the nearest waypoint
                 # not ahead of the arm's current phase, so it no longer always snaps to the start. A
                 # single (14,) pose is accepted (legacy fixed-home behaviour); None -> gripper only.
                 retreat_waypoints=None,
                 open_grip=0.0,                      # 0 = open; the recovery opens the right gripper
                 closed_grip_min=10.0,               # = postprocess.GRIPPER_CLOSE_THRESH: note_action
                                                     # reads the RAW server grip [0,~85] (pre-binarize)
                 device=None):
        self.det = _Detector(detector_path, device=device)
        self.settle_sec = settle_sec
        # Normalize to (n, 14): accept a waypoint list/array or a single legacy (14,) home pose.
        self.retreat_waypoints = (np.asarray(retreat_waypoints, dtype=float).reshape(-1, 14)
                                  if retreat_waypoints is not None else None)
        self.open_grip = float(open_grip)
        self.closed_grip_min = float(closed_grip_min)

        # grasp-attempt tracking
        self._prev_closed = False
        self._close_t = None          # monotonic time of the last open->close transition
        self._checked = False         # detector already run for this closure?

        # recovery state. The retreat is now SYNCHRONOUS (open gripper + move_to_joints home in
        # _begin_retreat), so _retreating never latches True and pump() is a no-op — kept only for the
        # auto-loop interface.
        self._retreating = False
        self._log_ts = {}             # key -> last monotonic log time (rate-limits hot-loop logs)
        log.info("GraspRecoveryMonitor ready: settle=%.1fs closed_grip_min=%.1f roi=%s thr=%.2f "
                 "device=%s", settle_sec, self.closed_grip_min, self.det.roi, self.det.threshold,
                 self.det.device)

    def _throttled(self, key, interval=2.0):
        now = time.monotonic()
        if now - self._log_ts.get(key, 0.0) >= interval:
            self._log_ts[key] = now
            return True
        return False

    # -- hook 1: every policy chunk, to track the right-grip command ------------
    def note_action(self, action_chunk):
        a = np.asarray(action_chunk, dtype=float)
        if a.ndim != 2 or a.shape[1] <= R_GRIP:
            if self._throttled("shape", 5.0):
                log.warning("[grasp-check] action chunk shape %s has no right-grip col (need 2-D with "
                            ">%d cols, i.e. a 20-col dual-arm row) -> grip tracking OFF, detector will "
                            "never fire", a.shape, R_GRIP)
            return
        rgrip = float(a[-1, R_GRIP])
        closed_now = bool(rgrip >= self.closed_grip_min)
        if self._throttled("grip", 2.0):                  # heartbeat: proves note_action is running
            log.info("[grasp-check] right grip=%.1f (close>=%.1f) closed=%s armed=%s checked=%s",
                     rgrip, self.closed_grip_min, closed_now, self._close_t is not None, self._checked)
        if closed_now and not self._prev_closed:          # open -> close: attempt starts
            self._close_t = time.monotonic()
            self._checked = False
            log.info("[grasp-check] open->close (grip=%.1f) -> attempt ARMED, detector check in %.1fs",
                     rgrip, self.settle_sec)
        elif not closed_now and self._prev_closed and not self._checked:
            self._close_t = None                          # released before the check -> cancel it
            log.info("[grasp-check] released before the %.1fs check -> attempt cancelled", self.settle_sec)
        self._prev_closed = closed_now

    # -- hook 2: every loop BEFORE predict. True => entered recovery (skip predict)
    def maybe_start(self, env, obs) -> bool:
        if self._retreating or self._close_t is None or self._checked:
            return False
        waited = time.monotonic() - self._close_t
        if waited < self.settle_sec:
            if self._throttled("settle", 1.0):
                log.info("[grasp-check] grasp armed, settling %.1f/%.1fs before detector runs",
                         waited, self.settle_sec)
            return False
        self._checked = True
        log.info("[grasp-check] settle elapsed -> running detector on right wrist frame")

        frame = obs.get("handr_imgs")
        if isinstance(frame, (list, tuple)):
            frame = frame[-1] if frame else None
        if frame is None:
            log.warning("[grasp-check] no 'handr_imgs' right-wrist frame in obs (keys=%s) -> "
                        "cannot run detector", list(obs.keys()))
            return False

        p = self.det.p_empty(frame)
        if p < self.det.threshold:
            log.info("[grasp-check] detector P_empty=%.2f < thr=%.2f -> grasp HELD, no recovery",
                     p, self.det.threshold)
            return False

        self._begin_retreat(env, obs)
        log.warning("[recovery] missed grasp (P_empty=%.2f >= thr=%.2f) -> opened gripper + moved home",
                    p, self.det.threshold)
        return True

    @property
    def is_retreating(self) -> bool:
        return self._retreating

    # -- hook 3: kept for the auto-loop interface. The retreat is now synchronous (done inside
    # _begin_retreat), so there is nothing to stream here.
    def pump(self, env):
        return

    # -- internals --------------------------------------------------------------
    def _begin_retreat(self, env, obs):
        """Recovery = let go + reset. OPEN the right gripper + move BOTH arms to the nearest pre-grasp
        approach waypoint not ahead of the arm's current phase (shared retreat_to_nearest), so the
        retreat backs off only as far as needed rather than all the way to the start. Pins the
        per-substep delta so it cruises at RETREAT_JOINT_VEL_FRAC of MAX_JOINT_VEL. Blocks until the
        arm arrives; policy re-approaches."""
        if self.retreat_waypoints is None:
            log.warning("[recovery] no retreat waypoints configured -> gripper-only recovery")
            if hasattr(env, "command_gripper"):
                env.command_gripper(gr=self.open_grip)
            return
        retreat_to_nearest(env, self.retreat_waypoints, open_grip=self.open_grip)
        self._finish()                                     # synchronous recovery done; policy resumes

    def _finish(self):
        self._retreating = False
        # gripper is open now; re-arm grip tracking so the next close is a fresh attempt
        self._prev_closed = False
        self._close_t = None

    def reset(self):
        self._finish()
