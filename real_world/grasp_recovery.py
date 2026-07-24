"""Standalone CCDP-style grasp-failure recovery for the dual-arm policy.

Self-contained: loads a wrist-camera YOLO gripper-state detector (data/grasp_detector/detector.pt,
a 3-class bbox model: open / closed-gripped / closed-empty) and drives failure recovery, with NO
GUI coupling and only three small call-ins from InferenceController. Needs only ultralytics +
the .pt checkpoint on the client machine (detection runs LOCALLY, not on the policy server).

WHAT IT DOES
------------
Whenever the RIGHT gripper is commanded CLOSED, runs the YOLO detector on the right wrist frame
(obs['handr_imgs'][-1]) EVERY loop — no settle wait, no one-shot latch. The instant it reports
'closed-empty' (gripper closed on nothing) for `confirm_frames` consecutive checks (default 1 =
immediately) it enters RECOVERY:

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
from PIL import Image
from ultralytics import YOLO

from real_world.retreat import retreat_to_nearest

log = logging.getLogger(__name__)

# 20-col dual_arm_ee_image action row: L[pos3,rot6d6,grip1] ++ R[pos3,rot6d6,grip1]
L_EE = slice(0, 9)      # left [pos3, rot6d6]  (held during retreat)
L_GRIP = 9
R_POS = slice(10, 13)   # right pos3
R_ROT = slice(13, 19)   # right rot6d6
R_GRIP = 19

# 3-class wrist-camera gripper-state model. Recovery fires ONLY when the detector reports this
# state (gripper closed but holding nothing). "open" and "closed-gripped" -> hold, no recovery.
RECOVERY_STATE = "closed-empty"


class _Detector:
    """Inlined YOLO gripper-state detector: a 3-class bbox model (open / closed-gripped /
    closed-empty) over the right wrist frame. classify() returns the highest-confidence detected
    state name and its confidence, or (None, 0.0) when nothing clears the conf threshold."""

    def __init__(self, ckpt_path, *, conf=0.40, device=None):
        self.conf = float(conf)
        self.device = device                       # None -> ultralytics auto-selects (cuda if avail)
        self.model = YOLO(ckpt_path)
        self.names = {int(i): str(n) for i, n in self.model.names.items()}
        if RECOVERY_STATE not in self.names.values():
            log.warning("[grasp-yolo] model classes %s do not include '%s' -> recovery can NEVER "
                        "fire; retrain or rename the model's classes", self.names, RECOVERY_STATE)

    def classify(self, frame, imgsz=None):
        """(state_name, confidence). state_name is None when no box clears self.conf. `imgsz` lets a
        caller run a cheaper forward pass (e.g. the live GUI pill at the wrist frame's native 320px)
        without touching the full-res recovery check."""
        if not isinstance(frame, Image.Image):
            frame = Image.fromarray(np.asarray(frame)[..., :3].astype(np.uint8))
        kw = {"conf": self.conf, "device": self.device, "verbose": False}
        if imgsz is not None:
            kw["imgsz"] = int(imgsz)
        # Pass a PIL image: ultralytics treats it as RGB (obs frames are RGB), so no BGR channel swap.
        res = self.model.predict(frame.convert("RGB"), **kw)[0]
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            return None, 0.0
        confs = boxes.conf.cpu().numpy()
        i = int(confs.argmax())                    # highest-confidence detection is the gripper state
        return self.names.get(int(boxes.cls[i].item())), float(confs[i])

    @property
    def available(self):
        """True when a YOLO model is loaded (detection can run). Mirrors YoloGate.available so the
        detector-tuning view can treat the gripper and no-flip detectors uniformly."""
        return self.model is not None

    def analyze(self, frame, floor=0.05):
        """Detector-tuning helper: run YOLO at a low confidence FLOOR and return every box as a dict
        {x, y, w, h, reason, text} in the tuning view's schema — reason is None for boxes that clear the
        live self.conf (drawn green) and a short string for below-threshold near-misses (drawn red), so
        the same render path used for the no-flip YoloGate renders the gripper model too."""
        if not isinstance(frame, Image.Image):
            frame = Image.fromarray(np.asarray(frame)[..., :3].astype(np.uint8))
        res = self.model.predict(frame.convert("RGB"), conf=float(floor), device=self.device,
                                 verbose=False)[0]
        out = []
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            return out
        xyxy = boxes.xyxy.cpu().numpy()
        cls = boxes.cls.cpu().numpy()
        cf = boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), c, p in zip(xyxy, cls, cf):
            name = self.names.get(int(c), str(int(c)))
            out.append({"x": int(x1), "y": int(y1), "w": int(x2 - x1), "h": int(y2 - y1),
                        "reason": None if p >= self.conf else f"conf {p:.2f}",
                        "text": f"{name} {p:.2f}"})
        return out


class GraspRecoveryMonitor:
    def __init__(self, detector_path, *,
                 confirm_frames=1,                   # consecutive closed-empty detections before it
                                                     # fires (1 = the instant one is seen; raise to
                                                     # debounce single-frame YOLO false positives)
                 # Pre-grasp APPROACH waypoints (n, 14) [left7, right7] the recovery retreats to (open
                 # gripper + move BOTH arms), ordered start -> pre-grasp. Passed in by the controller
                 # (real_world/config/retreat_waypoints.json). The retreat picks the nearest waypoint
                 # not ahead of the arm's current phase, so it no longer always snaps to the start. A
                 # single (14,) pose is accepted (legacy fixed-home behaviour); None -> gripper only.
                 retreat_waypoints=None,
                 open_grip=0.0,                      # 0 = open; the recovery opens the right gripper
                 closed_grip_min=10.0,               # = postprocess.GRIPPER_CLOSE_THRESH: note_action
                                                     # reads the RAW server grip [0,~85] (pre-binarize)
                 conf=0.40,                          # min YOLO box confidence to accept a state
                 device=None):
        self.det = _Detector(detector_path, conf=conf, device=device)
        self.confirm_frames = max(1, int(confirm_frames))
        # Normalize to (n, 14): accept a waypoint list/array or a single legacy (14,) home pose.
        self.retreat_waypoints = (np.asarray(retreat_waypoints, dtype=float).reshape(-1, 14)
                                  if retreat_waypoints is not None else None)
        self.open_grip = float(open_grip)
        self.closed_grip_min = float(closed_grip_min)

        # grasp-attempt tracking. The detector now runs EVERY tick while the right gripper is closed
        # (no settle wait, no one-shot latch) and fires the instant it sees closed-empty, so all we
        # keep is the consecutive-empty streak and the last commanded grip state (heartbeat fallback).
        self._prev_closed = False
        self._empty_streak = 0        # consecutive closed-empty detections while the gripper is closed
        # Cached last gripper read for the (cheap) GUI pill. Updated by maybe_start during auto and by
        # the controller's gripper_scan() when idle, so the pill NEVER runs its own forward pass on the
        # UI thread while auto is running (which would double-infer AND race the loop on one YOLO model).
        self._live = {"available": True, "frame_ok": False, "state": None, "conf": 0.0}

        # recovery state. The retreat is SYNCHRONOUS (open gripper + move_to_joints home in
        # _begin_retreat), so _retreating never latches True and pump() is a no-op — kept only for the
        # auto-loop interface.
        self._retreating = False
        self._log_ts = {}             # key -> last monotonic log time (rate-limits hot-loop logs)
        log.info("GraspRecoveryMonitor ready: confirm_frames=%d closed_grip_min=%.1f classes=%s "
                 "recover_on='%s' conf>=%.2f device=%s", self.confirm_frames, self.closed_grip_min,
                 self.det.names, RECOVERY_STATE, self.det.conf, self.det.device or "auto")

    def _throttled(self, key, interval=2.0):
        now = time.monotonic()
        if now - self._log_ts.get(key, 0.0) >= interval:
            self._log_ts[key] = now
            return True
        return False

    # -- hook 1: every policy chunk. Heartbeat + a fallback 'right gripper closed?' signal for envs
    #    that don't expose _last_grip_cmd (maybe_start prefers that; this is only a backstop).
    def note_action(self, action_chunk):
        a = np.asarray(action_chunk, dtype=float)
        if a.ndim != 2 or a.shape[1] <= R_GRIP:
            return                                        # not a 20-col dual-arm chunk -> ignore
        rgrip = float(a[-1, R_GRIP])
        self._prev_closed = bool(rgrip >= self.closed_grip_min)
        if self._throttled("grip", 2.0):                  # heartbeat: proves note_action is running
            log.info("[grasp-check] right grip=%.1f (close>=%.1f) closed=%s streak=%d",
                     rgrip, self.closed_grip_min, self._prev_closed, self._empty_streak)

    def _right_gripper_closed(self, env) -> bool:
        """Is the RIGHT gripper commanded closed right now? Prefer the env's EFFECTIVE binary grip
        command (_last_grip_cmd = [gl, gr], what no_flip/flip/package-gate all read); fall back to the
        state note_action tracked from the action chunk."""
        grip = getattr(env, "_last_grip_cmd", None)
        if grip is not None and len(grip) > 1:
            return bool(grip[1])
        return self._prev_closed

    # -- hook 2: every loop BEFORE predict. True => entered recovery (skip predict)
    def maybe_start(self, env, obs) -> bool:
        """Runs the gripper detector on the right-wrist frame EVERY tick while the right gripper is
        closed (a miss can only happen while closed) and, the instant it sees 'closed-empty' for
        `confirm_frames` consecutive checks, opens the gripper and retreats. No settle wait and no
        one-shot latch — it keeps watching so a miss is caught as soon as the model reports it."""
        if self._retreating:
            return False
        if not self._right_gripper_closed(env):           # open / approach -> nothing to recover
            self._empty_streak = 0
            self._live = {"available": True, "frame_ok": True, "state": "open", "conf": 0.0}
            return False

        frame = obs.get("handr_imgs")
        if isinstance(frame, (list, tuple)):
            frame = frame[-1] if frame else None
        if frame is None:
            if self._throttled("noframe", 2.0):
                log.warning("[grasp-check] gripper closed but no 'handr_imgs' right-wrist frame in obs "
                            "(keys=%s) -> cannot run detector", list(obs.keys()))
            return False

        state, conf = self.det.classify(frame)
        self._live = {"available": True, "frame_ok": True, "state": state, "conf": float(conf)}
        if state != RECOVERY_STATE:
            self._empty_streak = 0
            if self._throttled("held", 2.0):
                log.info("[grasp-check] closed; state=%s conf=%.2f (recover on '%s') -> HELD",
                         state, conf, RECOVERY_STATE)
            return False

        # closed-empty seen -> count consecutive detections; fire the moment we reach confirm_frames.
        self._empty_streak += 1
        log.warning("[grasp-check] closed-empty detected (conf=%.2f) %d/%d",
                    conf, self._empty_streak, self.confirm_frames)
        if self._empty_streak < self.confirm_frames:
            return False
        self._empty_streak = 0
        self._begin_retreat(env, obs)
        log.warning("[recovery] missed grasp (closed-empty conf=%.2f) -> opened gripper + retreated",
                    conf)
        return True

    @property
    def is_retreating(self) -> bool:
        return self._retreating

    def live_status(self):
        """Cheap read of the last gripper state for the GUI pill — NO forward pass. Reflects the auto
        loop's own detections during auto, or the controller's idle gripper_scan() otherwise."""
        d = dict(self._live)
        d["recover_on"] = RECOVERY_STATE
        d["classes"] = list(self.det.names.values())
        return d

    def scan_live(self, env, imgsz=320):
        """Run ONE cheap forward pass on the right-wrist frame and cache the result for the pill. Used
        only while auto is OFF (during auto, maybe_start already refreshes the cache). imgsz defaults to
        the wrist frame's native 320px so the idle pill is ~2.4x cheaper than the full recovery check."""
        frame = None
        if env is not None:
            try:
                frame = env.get_frame("hand_right")   # right wrist cam; request()s it -> keeps it ON
            except Exception:
                frame = None
        if not (isinstance(frame, np.ndarray) and frame.size > 0):
            self._live = {"available": True, "frame_ok": False, "state": None, "conf": 0.0}
            return
        try:
            state, conf = self.det.classify(frame, imgsz=imgsz)
        except Exception as e:
            log.debug("gripper live scan skipped: %r", e)
            self._live = {"available": True, "frame_ok": False, "state": None, "conf": 0.0}
            return
        self._live = {"available": True, "frame_ok": True, "state": state, "conf": float(conf)}

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
        self._empty_streak = 0

    def reset(self):
        self._finish()
