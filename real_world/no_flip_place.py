"""Scripted release macro for the NO-FLIP case: place the grasped object WITHOUT flipping it.

Sibling of flip_place.py. The flip variant runs after the policy's grab -> lift -> ~180 deg flip and
keys its trigger off that wrist swing. This variant is what runs when the robot decides the package
should NOT be flipped: the object is grabbed and lifted but left in its original orientation, so there
is no characteristic wrist rotation to watch for. Once the (TBD) detection says "place it now", this
module takes over INLINE in the auto loop exactly like the flip variant:

    1. stop predicting + clear ALL queues (so nothing stale runs)
    2. move the RIGHT arm from its LIVE pose to a FIXED release pose, following the shape recorded in
       assets, OPEN the gripper there (release), then reverse the exact same path back to the live start
    3. clear everything again and hand back -> auto inference resumes from the live pose

Design decisions (shared with the flip variant):
  * JOINT space, not EE: a fixed END joint config guarantees the same release EE point via FK with NO IK.
  * The start ADAPTS to wherever the arm ended up (varies), the END is FIXED: a decaying-offset warp
    re-anchors the recorded shape so out[0]=live pose and out[-1]=the recorded release config.
  * Slow + smooth: streamed at vel_frac of max joint velocity, subdivided, from the live pose (zero seam).
  * No snap on return: robot queue + staging + merge buffer + grip latch are ALL cleared before and
    after, and the arm ends at the live start, so the first resumed inference is fresh.

Detection cue — YOLO DETECTION, with a commit LATCH:
    A package that shows a printed label/barcode is oriented correctly and must NOT be flipped. The
    detector scans CONTINUOUSLY every auto tick (not gated by the grasp). When the target is detected on
    ``commit_count`` consecutive scans (default 20) the macro LATCHES: it now knows the next place is
    no-flip. The latch is dropped again in any of three ways: the scripted place fires (once the object
    is grasped), the target is MISSED commit_count scans in a row, or reset() (auto stop / E-stop). A
    brief occlusion (e.g. the gripper crossing the label during the grasp) is tolerated — only
    commit_count consecutive misses unlock.
    Detection uses a YOLO object detector (ultralytics) over the HEAD camera frame; see YoloGate. The
    gate loads a trained ``.pt`` model from assets and fires when any target box clears the confidence
    threshold. Both the package (ultralytics) and the weights file are loaded LAZILY, so the module
    still imports before the model exists — the gate then never fires (like a disabled detector) until
    the weights are dropped in. The port is open: pass your own detector=callable(env, arm14)->bool (or
    None) to swap the cue; BarcodeGate (zxing-cpp) remains available for barcodes/QR.

    "Continuous" = scan() runs every auto tick while running; the GUI also calls scan() on its refresh
    while auto is OFF, so the live indicator responds to a label even when idle (only scan() runs then
    — never the FIRE stage). The commit latch is reset when auto-run STARTS, so an idle pre-arm drives
    only the indicator and the real decision is re-made live during the run.

Integration (one call-in in InferenceController._run_auto_inference, top of loop):
    nfp = getattr(self, "no_flip_place", None)
    if nfp is not None and nfp.maybe_trigger(env):
        continue                      # macro ran this cycle; skip predicting
"""

import logging
import os
import time

import cv2
import numpy as np

log = logging.getLogger(__name__)

# Camera(s) to scan. Only the HEAD camera is used — the operator presents the labelled package to the
# head camera to decide no-flip. (Name per humanoid_env AGENT_CAMERA.)
SCAN_CAMERAS = ("head",)

# No-flip reach+release path, built from recording205 (see scripts/build_release_path.py). Distinct
# from the flip variant's flip_release_path.npy (recording206) — the two cases use different releases.
DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "assets", "no_flip_release_path.npy")

# Trained YOLO weights the default detector loads (ultralytics .pt). Drop the trained model here (or
# pass weights=... / call YoloGate.reload(path)); until it exists the gate loads but never fires.
DEFAULT_WEIGHTS = os.path.join(os.path.dirname(__file__), "assets", "no_flip_yolo.pt")

R_GRIP_IDX = 1          # right channel in a [gl, gr] gripper command


class YoloGate:
    """Detection port for the no-flip case: "does the head camera see the target (label/barcode)?".

    A package showing a printed label/barcode is oriented correctly and must NOT be flipped, so the
    macro should place it as-is. This gate runs a trained YOLO object detector (ultralytics) on the
    scan-camera frame and reports SEEN when any target box clears the confidence threshold. It is the
    default no-flip cue, replacing the old OpenCV text-cluster detector.

    Same port contract as BarcodeGate — callable(env, arm14)->bool plus the live-state attributes the
    GUI reads (last_text/last_camera/last_seen_monotonic/last_frame_cams, seen_within, available,
    debug) — so it drops straight into NoFlipPlaceMacro(detector=...), debounced by the commit_count
    latch. It also exposes analyze()/conf/iou/imgsz so the detector-tuning page can draw live boxes and
    the confidence threshold can be tuned from the GUI.

    Both ultralytics AND the weights file are loaded LAZILY and tolerantly: if the package is missing
    or the .pt does not exist yet, the gate loads, logs once, and simply never fires (exactly like a
    disabled detector). That is the intended state until the trained model is dropped in at ``weights``
    (or supplied via weights=... / reload()). `classes` optionally restricts which detections count
    (a set/list of class ids or names); None = any detection counts.
    """

    def __init__(self, cameras=SCAN_CAMERAS, *, weights=DEFAULT_WEIGHTS, conf=0.5, iou=0.45,
                 imgsz=640, classes=None, device=None):
        self.cameras = tuple(cameras)
        self.weights = str(weights)
        self.conf = float(conf)               # min box confidence to count as a detection (GUI-tunable)
        self.iou = float(iou)                 # NMS IoU threshold
        self.imgsz = int(imgsz)               # inference image size (longest side)
        self.classes = classes                # None => any class counts; else ids/names that qualify
        self.device = device                  # None => ultralytics default (auto CPU/GPU)
        # Live state for the GUI indicator: what/where the last detection was and when it was seen.
        self.last_text = None                 # "name conf" of the most recent hit (for the indicator)
        self.last_camera = None               # which camera saw it
        self.last_seen_monotonic = 0.0        # time.monotonic() of the last hit (0 = never)
        self.last_frame_cams = ()             # cameras that delivered a frame on the last scan (diag)
        self.debug = False                    # set True (via NoFlipPlaceMacro debug) for per-scan diag
        self._last_diag = 0.0                 # throttle timer for the debug print
        self._warned = False                  # one-shot guard for repeated predict errors
        self._model = None                    # loaded ultralytics YOLO (None => gate never fires)
        self._load_error = None               # human-readable reason the model is not loaded
        self._load()

    def _load(self):
        """Lazily import ultralytics and load the weights. Never raises: on any failure the gate stays
        inert (available == False) with the reason in self._load_error."""
        try:
            from ultralytics import YOLO
        except ImportError:
            self._load_error = "ultralytics not installed (`pip install ultralytics`)"
            log.error("[no-flip-place] %s; YOLO detection disabled -> macro will never fire",
                      self._load_error)
            return
        if not os.path.exists(self.weights):
            self._load_error = f"weights not found at {self.weights}"
            log.warning("[no-flip-place] %s; YOLO detection disabled until the trained model is added "
                        "-> macro will never fire", self._load_error)
            return
        try:
            model = YOLO(self.weights)
            if self.device is not None:
                model.to(self.device)
            self._model = model
            self._load_error = None
            log.info("[no-flip-place] YOLO model loaded from %s (classes=%s)",
                     self.weights, getattr(model, "names", None))
        except Exception as e:                # a corrupt/incompatible checkpoint must not kill import
            self._load_error = f"failed to load YOLO weights: {e}"
            log.error("[no-flip-place] %s -> macro will never fire", self._load_error)

    def reload(self, weights=None):
        """(Re)load the model — call after dropping a freshly trained .pt at `weights` so the running
        gate picks it up without a restart. Returns True if a model is now loaded."""
        if weights is not None:
            self.weights = str(weights)
        self._model = None
        self._load_error = None
        self._warned = False
        self._load()
        return self.available

    @property
    def available(self) -> bool:
        """True when the YOLO model is loaded (i.e. detection can actually run)."""
        return self._model is not None

    def seen_within(self, hold_s) -> bool:
        """Was a target detected in the last `hold_s` seconds? (drives the live GUI indicator)."""
        return self.last_seen_monotonic > 0.0 and (time.monotonic() - self.last_seen_monotonic) < hold_s

    def _class_ok(self, cls_id, names) -> bool:
        """Whether a detected class id passes the optional `classes` filter (ids or names)."""
        if self.classes is None:
            return True
        if cls_id in self.classes:
            return True
        name = names.get(cls_id) if isinstance(names, dict) else None
        return name is not None and name in self.classes

    def _detect(self, rgb):
        """Run the model on one uint8 RGB frame and return accepted detections as
        [(cls_id, conf, name, (x1, y1, x2, y2)), ...] (empty if none / no model). Never raises.

        The gate's public contract is RGB in (matching env.get_frame); ultralytics assumes a numpy
        array is BGR and flips it internally, so convert RGB->BGR right before predict to match."""
        if self._model is None:
            return []
        try:
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            res = self._model.predict(bgr, conf=self.conf, iou=self.iou, imgsz=self.imgsz,
                                      device=self.device, verbose=False)[0]
        except Exception as e:
            if not self._warned:
                log.warning("[no-flip-place] YOLO predict failed: %s", e)
                self._warned = True
            return None                        # signal an error to the caller (vs [] = ran, no hits)
        names = getattr(res, "names", None) or getattr(self._model, "names", {}) or {}
        boxes = getattr(res, "boxes", None)
        out = []
        if boxes is not None and len(boxes):
            for b in boxes:
                cid = int(b.cls[0])
                if not self._class_ok(cid, names):
                    continue
                cf = float(b.conf[0])
                x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
                out.append((cid, cf, names.get(cid, str(cid)) if isinstance(names, dict) else str(cid),
                            (x1, y1, x2, y2)))
        return out

    def __call__(self, env, arm14) -> bool:
        if self._model is None:
            self.last_frame_cams = ()
            self._diag(self._load_error or "YOLO model not loaded (reader missing)")
            return False
        got, diag = [], []                    # got: cams that delivered a frame; diag: per-cam lines
        for name in self.cameras:
            frame = env.get_frame(name)       # also request()s the camera -> switches it ON
            if frame is None:                 # not warmed up / not streaming yet
                diag.append(f"{name}=NO-FRAME")
                continue
            got.append(name)
            shape = getattr(frame, "shape", None)
            rgb = self._as_uint8_rgb(frame)   # coerce to the uint8 HxWx3 ultralytics expects
            if rgb is None:
                diag.append(f"{name}=BAD-FMT{shape}")
                continue
            dets = self._detect(rgb)
            if dets is None:                  # predict errored -> already logged; skip this camera
                diag.append(f"{name}=ERR")
                continue
            diag.append(f"{name}{shape}={len(dets)}det(s)"
                        + (f":{dets[0][2]} {dets[0][1]:.2f}" if dets else ""))
            if dets:
                cid, cf, cname, _xyxy = max(dets, key=lambda d: d[1])   # highest-confidence detection
                self.last_text = f"{cname} {cf:.2f}"
                self.last_camera = name
                self.last_seen_monotonic = time.monotonic()
                self.last_frame_cams = tuple(got)
                log.info("[no-flip-place] YOLO saw '%s' (%.2f) on '%s' -> place without flipping",
                         cname, cf, name)
                self._diag("  ".join(diag))
                return True
        self.last_frame_cams = tuple(got)     # for diagnostics: which cams are feeding frames
        self._diag("  ".join(diag) if diag else "(no cameras configured)")
        return False

    def analyze(self, frame):
        """Detector-tuning helper: run the model on an RGB frame and return per-box dicts the tuning
        page draws: {x, y, w, h, conf, text, reason}. reason is None for every returned box (all are
        accepted detections above `conf`), matching find-blocks semantics. Empty while no model is
        loaded, so the tuning view just shows the raw frame until the weights are added."""
        dets = self._detect(frame)
        if not dets:
            return []
        out = []
        for _cid, cf, cname, (x1, y1, x2, y2) in dets:
            out.append({"x": int(x1), "y": int(y1), "w": int(x2 - x1), "h": int(y2 - y1),
                        "conf": cf, "text": f"{cname} {cf:.2f}", "reason": None})
        return out

    def _diag(self, msg):
        """Throttled (~1 Hz) diagnostic print of what the scan saw, gated by self.debug."""
        if not self.debug:
            return
        now = time.monotonic()
        if now - self._last_diag >= 1.0:
            self._last_diag = now
            print(f"[no-flip scan] cams={list(self.cameras)}  {msg}")

    @staticmethod
    def _as_uint8_rgb(frame):
        """Coerce a camera frame to the uint8 HxWx3 ultralytics wants: drop an alpha channel, and treat
        a single-channel image as grayscale->RGB. Returns None for anything unusable."""
        try:
            if frame.dtype != np.uint8:
                return None
            if frame.ndim == 2:
                return cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            if frame.ndim == 3:
                if frame.shape[2] == 4:
                    return np.ascontiguousarray(frame[:, :, :3])
                if frame.shape[2] == 3:
                    return frame
            return None
        except (AttributeError, cv2.error):
            return None


class BarcodeGate:
    """Detection port for the no-flip case: "does a camera see a barcode on the grasped object?".

    A visible barcode means the package is oriented right-side-up and must NOT be flipped, so the
    macro should place it as-is. Uses the zxing-cpp library (`pip install zxing-cpp`), which decodes
    the common 1-D/2-D symbologies (Code128, EAN/UPC, QR, ...) straight from an RGB numpy frame and is
    far more robust than OpenCV's bundled detector. A barcode counts as SEEN when zxing decodes one;
    pass symbologies=("EAN13", ...) / a zxingcpp.BarcodeFormat to restrict which kinds count.

    zxing-cpp is imported lazily so the module still loads on a machine without it (the gate then
    logs once and never fires, exactly like a disabled detector). Callable, so it drops straight into
    NoFlipPlaceMacro(detector=...), which debounces it via the commit_count latch (a barcode must be
    detected that many scans before no-flip locks in).
    """

    def __init__(self, cameras=SCAN_CAMERAS, *, symbologies=None):
        self.cameras = tuple(cameras)
        self.symbologies = symbologies    # None => accept any; else a zxingcpp.BarcodeFormat mask
        self._warned = False
        # Live state for the GUI indicator: what/where the last barcode was and when it was seen.
        self.last_text = None             # decoded payload of the most recent hit
        self.last_format = None           # symbology name of that hit
        self.last_camera = None           # which camera saw it
        self.last_seen_monotonic = 0.0    # time.monotonic() of the last hit (0 = never)
        self.last_frame_cams = ()         # cameras that delivered a frame on the last scan (diagnostics)
        self.debug = False                # set True (via NoFlipPlaceMacro debug) to print per-scan diag
        self._last_diag = 0.0             # throttle timer for the debug print
        try:
            import zxingcpp
            self._zxing = zxingcpp
        except ImportError:
            self._zxing = None
            log.error("[no-flip-place] zxing-cpp not installed (`pip install zxing-cpp`); "
                      "barcode detection disabled -> macro will never fire")

    @property
    def available(self) -> bool:
        """True when the zxing-cpp backend loaded (i.e. detection can actually run)."""
        return self._zxing is not None

    def seen_within(self, hold_s) -> bool:
        """Was a barcode decoded in the last `hold_s` seconds? (drives the live GUI indicator)."""
        return self.last_seen_monotonic > 0.0 and (time.monotonic() - self.last_seen_monotonic) < hold_s

    def __call__(self, env, arm14) -> bool:
        if self._zxing is None:
            self.last_frame_cams = ()
            self._diag("zxing-cpp NOT available (reader missing)")
            return False
        got = []                                      # cameras that actually delivered a frame this scan
        diag = []                                     # per-camera one-liners for the debug print
        hit = False
        for name in self.cameras:
            frame = env.get_frame(name)               # also request()s the camera -> switches it ON
            if frame is None:                         # not warmed up / not streaming yet
                diag.append(f"{name}=NO-FRAME")
                continue
            got.append(name)
            shape = getattr(frame, "shape", None)
            frame = self._as_uint8_rgb(frame)         # coerce to what zxing expects (drop alpha, etc.)
            if frame is None:
                diag.append(f"{name}=BAD-FMT{shape}")
                continue
            try:
                if self.symbologies is not None:
                    results = self._zxing.read_barcodes(frame, formats=self.symbologies)
                else:
                    results = self._zxing.read_barcodes(frame)
            except Exception as e:                    # never let a decode hiccup kill the auto loop
                diag.append(f"{name}=ERR({e})")
                if not self._warned:
                    log.warning("[no-flip-place] barcode read failed on '%s': %s", name, e)
                    self._warned = True
                continue
            texts = [r.text for r in results if r.text]
            diag.append(f"{name}{shape}={len(texts)}code(s)" + (f":{texts[0]}" if texts else ""))
            for r in results:
                if r.text:                            # a barcode was decoded -> it is seen
                    self.last_text = r.text            # publish live state for the GUI indicator
                    self.last_format = str(r.format)
                    self.last_camera = name
                    self.last_seen_monotonic = time.monotonic()
                    self.last_frame_cams = tuple(got)
                    log.info("[no-flip-place] barcode seen on '%s' (%s: %s) -> place without flipping",
                             name, r.format, r.text)
                    self._diag("  ".join(diag))
                    return True
        self.last_frame_cams = tuple(got)             # for diagnostics: which cams are feeding frames
        self._diag("  ".join(diag) if diag else "(no cameras configured)")
        return hit

    def _diag(self, msg):
        """Throttled (~1 Hz) diagnostic print of what the scan saw, gated by self.debug."""
        if not self.debug:
            return
        now = time.monotonic()
        if now - self._last_diag >= 1.0:
            self._last_diag = now
            print(f"[no-flip scan] cams={list(self.cameras)}  {msg}")

    @staticmethod
    def _as_uint8_rgb(frame):
        """Coerce a camera frame to the uint8 HxWx3 zxing wants: drop an alpha channel, and treat a
        single-channel image as grayscale. Returns None for anything unusable (so the caller skips it)."""
        try:
            if frame.dtype != np.uint8:
                return None
            if frame.ndim == 2:
                return frame                          # grayscale is fine
            if frame.ndim == 3:
                if frame.shape[2] == 4:
                    return np.ascontiguousarray(frame[:, :, :3])
                if frame.shape[2] == 3:
                    return frame
            return None
        except AttributeError:
            return None


class NoFlipPlaceMacro:
    _UNSET = object()

    def __init__(self, path=DEFAULT_PATH, *,
                 detector=_UNSET,      # PORT: callable(env, arm14)->bool. default = YoloGate; None disables
                 commit_count=20,      # a label must be detected this many consecutive scans to LATCH no-flip
                 grip_delay_s=1.5,     # after the grasp closes, wait this long before firing the macro
                 vel_frac=0.5,         # streaming speed = fraction of MAX joint velocity
                 release_settle_s=0.4, # pause after opening the gripper before withdrawing
                 open_grip=0.0,        # 0 = open (release)
                 debug=True):          # print a throttled per-scan diagnostic (frames + decode results)
        self.path = np.load(path).astype(np.float64)      # (M, 14) absolute-joint waypoints
        if self.path.ndim != 2 or self.path.shape[1] != 14:
            raise ValueError(f"no-flip release path must be (M,14); got {self.path.shape}")
        # default cue = YOLO sees the target -> don't flip; pass detector=None to disable, BarcodeGate()
        # for zxing barcodes/QR, or your own callable
        self.detector = YoloGate() if detector is self._UNSET else detector
        if hasattr(self.detector, "debug"):
            self.detector.debug = bool(debug)             # route scan diagnostics to stdout
        self.commit_count = max(1, int(commit_count))
        self.grip_delay_s = float(grip_delay_s)
        self.vel_frac = float(vel_frac)
        self.release_settle_s = float(release_settle_s)
        self.open_grip = float(open_grip)
        self.enabled = True           # live on/off: False -> maybe_trigger is a no-op (macro never runs)
        # Continuous-scan + commit-latch state:
        self.committed = False        # THE LOCK: label detected commit_count scans -> "next place is no-flip".
                                      #   Cleared when the macro fires, when the label has been MISSED
                                      #   commit_count scans in a row, or on reset().
        self._seen_count = 0          # consecutive scans the label has been detected (lock counter)
        self._absent_count = 0        # consecutive scans the label has been missed (unlock counter)
        self._grip_closed_since = None  # monotonic time the grasp closed (grip-settle timer before firing)
        self._fired = False           # macro ran for the current committed cycle (for status display)
        log.info("NoFlipPlaceMacro ready: %d waypoints, commit=%d detections grip_delay=%.1fs vel=%.0f%% "
                 "max, detector=%s", len(self.path), self.commit_count, self.grip_delay_s,
                 self.vel_frac * 100,
                 type(self.detector).__name__ if self.detector is not None else "None (disabled)")

    def _should_place(self, env, arm14) -> bool:
        """Detection gate: True => place the object as-is (no flip).

        Default cue is YoloGate — a scan camera sees the target (label/barcode) on the grasped package,
        meaning it is oriented correctly and must not be flipped. The port stays open: pass detector=your
        own callable(env, arm14) -> bool to __init__ to swap the cue, or detector=None to disable.

        `arm14` is the live 14-vec arm_joints [left7, right7] (unused by the YOLO cue; available to
        custom detectors). Returns False when no detector is wired, so the macro is inert by default.
        """
        if self.detector is not None:
            return bool(self.detector(env, arm14))
        return False

    def scan(self, env, arm14=None) -> bool:
        """One detection tick: run the detector and update the commit latch. Does NOT fire / move the
        robot, so it is safe to call outside the auto loop — e.g. the GUI calls it while auto is OFF so
        the live indicator still responds to a label held under a camera. A label detected for
        commit_count consecutive scans LATCHES ``self.committed``; the latch is dropped again if the
        label is then MISSED commit_count scans in a row. Returns committed. No-op while disabled."""
        if not self.enabled:
            return self.committed
        if self._should_place(env, arm14):           # PORT: a label is visible this scan
            self._absent_count = 0                   # seen -> reset the unlock (miss) counter
            self._seen_count += 1
            if not self.committed and self._seen_count >= self.commit_count:
                self.committed = True                # LATCH: detected commit_count scans in a row
                self._fired = False                  # fresh committed cycle
                log.warning("[no-flip-place] label detected %dx -> LOCKED: next place is no-flip",
                            self.commit_count)
        else:
            self._seen_count = 0                     # streak broken; restart the seen counter
            self._absent_count += 1
            if self.committed and self._absent_count >= self.commit_count:
                self.committed = False               # UNLOCK: label missed commit_count scans in a row
                log.warning("[no-flip-place] label missed %dx -> UNLOCKED (no-flip latch dropped)",
                            self.commit_count)
        return self.committed

    def maybe_trigger(self, env) -> bool:
        """Auto-loop hook: True => the macro ran this cycle (caller should skip predicting). Two stages:

          1. scan() — continuous detection + commit latch (see scan()).
          2. FIRE — once committed AND the object has been grasped (right gripper commanded closed) for
             grip_delay_s, the scripted place runs; the latch then CLEARS so the next object must earn
             its own commit. The grip_delay_s wait lets the grasp/lift settle so the macro doesn't jump
             in the instant the fingers close.

        No-op (returns False) while disabled or while no detector is wired, so it is safe by default."""
        if not self.enabled:
            return False
        arm14 = env._read_arm14()
        self.scan(env, arm14)                        # 1. continuous scan + commit latch (no motion)

        # 2. fire once committed and the object has been grasped for grip_delay_s
        now = time.monotonic()
        grip = getattr(env, "_last_grip_cmd", None)
        grip_closed = grip is not None and grip[R_GRIP_IDX] >= 1
        if not grip_closed:
            self._grip_closed_since = None           # released -> reset the grip-settle timer
        elif self._grip_closed_since is None:
            self._grip_closed_since = now            # grasp just closed -> start the settle timer
        settled = (self._grip_closed_since is not None
                   and (now - self._grip_closed_since) >= self.grip_delay_s)
        if self.committed and grip_closed and settled and arm14 is not None:
            log.warning("[no-flip-place] committed + grasped %.1fs -> stop auto, run no-flip place",
                        now - self._grip_closed_since)
            self._run(env, np.asarray(arm14, dtype=np.float64))
            self.committed = False                   # lock resets after the macro fires
            self._seen_count = 0
            self._absent_count = 0
            self._grip_closed_since = None
            self._fired = True
            return True
        return False

    # -- internals --------------------------------------------------------------
    def _clear(self, env):
        """Drop EVERY source of stale motion so nothing snaps: robot queue, manual staging, streaming
        cursor, and the temporal-ensemble merge buffer."""
        with env._lock:
            env._robot_q.clear()
            env._staged_release.clear()
            env._queued_through = -1
        if getattr(env, "pipeline", None) is not None:
            env.pipeline.reset_merge()

    def _run(self, env, q_now):
        self._clear(env)                                  # 1. stop/clear before moving
        M = len(self.path)
        # 2. warp: out[i] = rec[i] + (q_now - rec[0]) * (1 - i/(M-1)) -> out[0]=q_now, out[-1]=release
        decay = 1.0 - np.arange(M) / (M - 1)
        fwd = self.path + np.outer(decay, q_now - self.path[0])
        fwd[:, :7] = q_now[:7]                             # HOLD the left arm at its current pose
        # forward: move out to the fixed release pose, gripper untouched (stays closed on the object)
        env.play_joint_path(fwd, vel_frac=self.vel_frac, grip=None)
        # release at the fixed point, let it settle, then withdraw along the same path
        if hasattr(env, "command_gripper"):
            env.command_gripper(gr=self.open_grip)
        time.sleep(self.release_settle_s)
        env.play_joint_path(fwd[::-1], vel_frac=self.vel_frac, grip=None)   # 3. come back to q_now
        self._clear(env)                                  # 4. clear again -> auto resumes fresh
        if hasattr(env, "reset_grip_latch"):
            env.reset_grip_latch()
        log.info("[no-flip-place] macro complete -> auto resumes from the live pose")

    def status(self, hold_s=1.5):
        """Live snapshot for the GUI indicator. `hold_s` = how long after a detection the target is still
        reported as "seen" (the detector samples once per auto tick). Returns a dict (the barcode_*
        keys are kept for GUI back-compat; they now carry the YOLO detection state):
            enabled         — armed?
            has_detector    — a detection cue is wired (vs detector=None)
            available       — the detector's backend is usable (YOLO model loaded); None if unknown
            barcode_seen    — the target was detected within hold_s (visible right now)
            barcode_text    — label+confidence of the most recent hit (or None)
            committed       — THE LOCK is set: the next place is committed to no-flip
            commit_progress — 0..1 toward the lock (seen_count / commit_count)
            seen_count      — consecutive scans the label has been detected so far
            commit_count    — detections-in-a-row needed to lock
            fired           — the macro has run for the current committed cycle
            frames_ok       — the last scan got a camera frame (distinguishes "camera not feeding"
                              from "frames arriving but no label found"); None if unknown
            frame_cams      — which cameras delivered a frame on the last scan
        """
        det = self.detector
        progress = min(1.0, self._seen_count / self.commit_count) if self.commit_count > 0 else 0.0
        frame_cams = getattr(det, "last_frame_cams", None)
        return {
            "enabled": self.enabled,
            "has_detector": det is not None,
            "available": getattr(det, "available", None),
            "barcode_seen": bool(det is not None and hasattr(det, "seen_within")
                                 and det.seen_within(hold_s)),
            "barcode_text": getattr(det, "last_text", None),
            "committed": self.committed,
            "commit_progress": progress,
            "seen_count": self._seen_count,
            "commit_count": self.commit_count,
            "fired": self._fired,
            "frames_ok": (len(frame_cams) > 0) if frame_cams is not None else None,
            "frame_cams": list(frame_cams) if frame_cams else [],
        }

    def reset(self):
        """Drop the commit latch + counters (e.g. on auto stop / E-stop) so a restart begins fresh."""
        self.committed = False
        self._seen_count = 0
        self._absent_count = 0
        self._grip_closed_since = None
        self._fired = False
