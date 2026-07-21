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

Detection cue — ARUCO MARKER, with a commit LATCH:
    A package that shows an ArUco marker is oriented correctly and must NOT be flipped. The detector
    scans CONTINUOUSLY every auto tick (not gated by the grasp). When a marker is seen for an unbroken
    ``commit_s`` (default 6 s) the macro LATCHES: it now knows the next place is no-flip. The latch is
    dropped again in any of three ways: the scripted place fires (once the object is grasped), the
    marker stays OUT of view for commit_s, or reset() (auto stop / E-stop). A brief occlusion (e.g. the
    gripper crossing the marker during the grasp) is tolerated — only a full commit_s of absence unlocks.
    Detection uses OpenCV's cv2.aruco (no extra dependency) over the HEAD camera frame; see ArucoGate.
    ArUco is chosen over a 1-D barcode / QR because it decodes at a far smaller on-screen size (~12 px
    vs ~200 px for EAN-13) and tolerates angle — the head camera's wide FOV makes 1-D codes unreadable
    unless held right against the lens. The port is open: pass your own detector=callable(env,
    arm14)->bool (or None) to swap the cue; BarcodeGate (zxing-cpp) remains available for that.

    "Continuous" = scan() runs every auto tick while running; the GUI also calls scan() on its refresh
    while auto is OFF, so the live indicator responds to a barcode even when idle (only scan() runs then
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

# ArUco dictionary: 4x4 markers (small, robust, 50 ids). Present the operator a printed marker from
# this dict (see scripts/barcode_webcam_test.py --make-aruco, or cv2.aruco.generateImageMarker).
ARUCO_DICT = "DICT_4X4_50"

# Camera(s) to scan for a barcode. Only the HEAD camera is used — the operator presents the barcode
# to the head camera to decide no-flip. (Name per humanoid_env AGENT_CAMERA.)
BARCODE_CAMERAS = ("head",)

# Placeholder: reuses the flip variant's recorded path so the macro is runnable today. Swap this for a
# dedicated no-flip placement path once one is recorded (the release EE point likely differs).
DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "assets", "flip_release_path.npy")

R_GRIP_IDX = 1          # right channel in a [gl, gr] gripper command


class ArucoGate:
    """Detection port for the no-flip case: "does the head camera see an ArUco marker?".

    A visible marker means the package is oriented right-side-up and must NOT be flipped, so the macro
    should place it as-is. Uses OpenCV's cv2.aruco (bundled with opencv-python; no extra dependency),
    which locates 4x4 markers down to ~12 px on-screen and tolerates rotation — far better at the head
    camera's distance/FOV than a 1-D barcode. A marker counts as SEEN when detected; pass ids=(7, ...)
    to only accept specific marker id(s), else any marker in the dictionary counts.

    Callable, so it drops straight into NoFlipPlaceMacro(detector=...). The commit_s latch debounces it
    (a marker must be held that long before no-flip locks in).
    """

    def __init__(self, cameras=BARCODE_CAMERAS, *, dict_name=ARUCO_DICT, ids=None):
        self.cameras = tuple(cameras)
        self.ids = set(ids) if ids is not None else None   # None => accept any marker id
        self.dict_name = dict_name
        self._warned = False
        self.last_text = None             # "id=<n>" of the most recent hit (for the GUI indicator)
        self.last_id = None
        self.last_camera = None
        self.last_seen_monotonic = 0.0
        self.last_frame_cams = ()
        self.debug = False
        self._last_diag = 0.0
        try:
            aruco = cv2.aruco
            dictionary = aruco.getPredefinedDictionary(getattr(aruco, dict_name))
            self._detector = aruco.ArucoDetector(dictionary, aruco.DetectorParameters())
        except Exception as e:
            self._detector = None
            log.error("[no-flip-place] cv2.aruco unavailable (%r); marker detection disabled -> "
                      "macro will never fire", e)

    @property
    def available(self) -> bool:
        """True when the ArUco detector built (i.e. detection can actually run)."""
        return self._detector is not None

    def seen_within(self, hold_s) -> bool:
        """Was a marker detected in the last `hold_s` seconds? (drives the live GUI indicator)."""
        return self.last_seen_monotonic > 0.0 and (time.monotonic() - self.last_seen_monotonic) < hold_s

    def __call__(self, env, arm14) -> bool:
        if self._detector is None:
            self.last_frame_cams = ()
            self._diag("cv2.aruco NOT available (reader missing)")
            return False
        got, diag = [], []
        for name in self.cameras:
            frame = env.get_frame(name)               # also request()s the camera -> switches it ON
            if frame is None:
                diag.append(f"{name}=NO-FRAME")
                continue
            got.append(name)
            shape = getattr(frame, "shape", None)
            gray = self._as_gray(frame)
            if gray is None:
                diag.append(f"{name}=BAD-FMT{shape}")
                continue
            _corners, ids, _rej = self._detector.detectMarkers(gray)
            found = [] if ids is None else [int(i) for i in ids.flatten()]
            if self.ids is not None:
                found = [i for i in found if i in self.ids]
            diag.append(f"{name}{shape}={len(found)}marker(s)" + (f":{found}" if found else ""))
            if found:
                self.last_id = found[0]
                self.last_text = f"id={found[0]}"
                self.last_camera = name
                self.last_seen_monotonic = time.monotonic()
                self.last_frame_cams = tuple(got)
                log.info("[no-flip-place] ArUco marker %s seen on '%s' -> place without flipping",
                         found, name)
                self._diag("  ".join(diag))
                return True
        self.last_frame_cams = tuple(got)
        self._diag("  ".join(diag) if diag else "(no cameras configured)")
        return False

    def _diag(self, msg):
        """Throttled (~1 Hz) diagnostic print of what the scan saw, gated by self.debug."""
        if not self.debug:
            return
        now = time.monotonic()
        if now - self._last_diag >= 1.0:
            self._last_diag = now
            print(f"[no-flip scan] cams={list(self.cameras)} dict={self.dict_name}  {msg}")

    @staticmethod
    def _as_gray(frame):
        """Coerce a camera frame to a uint8 single-channel image for cv2.aruco. Colour order is
        irrelevant (markers are black/white). Returns None for anything unusable."""
        try:
            if frame.dtype != np.uint8:
                return None
            if frame.ndim == 2:
                return frame
            if frame.ndim == 3:
                if frame.shape[2] == 4:
                    return cv2.cvtColor(frame, cv2.COLOR_RGBA2GRAY)
                if frame.shape[2] == 3:
                    return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
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
    NoFlipPlaceMacro(detector=...), which debounces it via the commit_s latch (a barcode must be held
    that long before no-flip locks in).
    """

    def __init__(self, cameras=BARCODE_CAMERAS, *, symbologies=None):
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
                 detector=_UNSET,      # PORT: callable(env, arm14)->bool. default = BarcodeGate; None disables
                 commit_s=6.0,         # a barcode must be seen CONTINUOUSLY this long to LATCH no-flip
                 grip_delay_s=1.5,     # after the grasp closes, wait this long before firing the macro
                 vel_frac=0.5,         # streaming speed = fraction of MAX joint velocity
                 release_settle_s=0.4, # pause after opening the gripper before withdrawing
                 open_grip=0.0,        # 0 = open (release)
                 debug=True):          # print a throttled per-scan diagnostic (frames + decode results)
        self.path = np.load(path).astype(np.float64)      # (M, 14) absolute-joint waypoints
        if self.path.ndim != 2 or self.path.shape[1] != 14:
            raise ValueError(f"no-flip release path must be (M,14); got {self.path.shape}")
        # default cue = ArUco marker seen -> don't flip; pass detector=None to disable, BarcodeGate()
        # for zxing barcodes/QR, or your own callable
        self.detector = ArucoGate() if detector is self._UNSET else detector
        if hasattr(self.detector, "debug"):
            self.detector.debug = bool(debug)             # route scan diagnostics to stdout
        self.commit_s = float(commit_s)
        self.grip_delay_s = float(grip_delay_s)
        self.vel_frac = float(vel_frac)
        self.release_settle_s = float(release_settle_s)
        self.open_grip = float(open_grip)
        self.enabled = True           # live on/off: False -> maybe_trigger is a no-op (macro never runs)
        # Continuous-scan + commit-latch state:
        self.committed = False        # THE LOCK: barcode held commit_s -> "the next place is no-flip".
                                      #   Cleared when the macro fires, when the barcode has been GONE
                                      #   for commit_s, or on reset().
        self._seen_since = None       # monotonic time the current unbroken barcode streak began
        self._absent_since = None     # monotonic time the barcode last went out of view (unlock timer)
        self._grip_closed_since = None  # monotonic time the grasp closed (grip-settle timer before firing)
        self._fired = False           # macro ran for the current committed cycle (for status display)
        log.info("NoFlipPlaceMacro ready: %d waypoints, commit=%.0fs grip_delay=%.1fs vel=%.0f%% max, "
                 "detector=%s", len(self.path), self.commit_s, self.grip_delay_s, self.vel_frac * 100,
                 type(self.detector).__name__ if self.detector is not None else "None (disabled)")

    def _should_place(self, env, arm14) -> bool:
        """Detection gate: True => place the object as-is (no flip).

        Default cue is BarcodeGate — a camera sees a barcode on the grasped package, meaning it is
        oriented correctly and must not be flipped. The port stays open: pass detector=your own
        callable(env, arm14) -> bool to __init__ to swap the cue, or detector=None to disable.

        `arm14` is the live 14-vec arm_joints [left7, right7] (unused by the barcode cue; available to
        custom detectors). Returns False when no detector is wired, so the macro is inert by default.
        """
        if self.detector is not None:
            return bool(self.detector(env, arm14))
        return False

    def scan(self, env, arm14=None) -> bool:
        """One detection tick: run the detector and update the commit latch. Does NOT fire / move the
        robot, so it is safe to call outside the auto loop — e.g. the GUI calls it while auto is OFF so
        the live indicator still responds to a barcode held under a camera. A barcode seen for an
        unbroken commit_s LATCHES ``self.committed``; the latch is dropped again if the barcode then
        stays OUT of view for commit_s. Returns the current committed state. No-op while disabled."""
        if not self.enabled:
            return self.committed
        now = time.monotonic()
        if self._should_place(env, arm14):           # PORT: a barcode is visible this tick
            self._absent_since = None                # seen -> reset the unlock (absence) timer
            if self._seen_since is None:
                self._seen_since = now
            if not self.committed and (now - self._seen_since) >= self.commit_s:
                self.committed = True                # LATCH: hold survived commit_s
                self._fired = False                  # fresh committed cycle
                log.warning("[no-flip-place] barcode held >= %.0fs -> LOCKED: next place is no-flip",
                            self.commit_s)
        else:
            self._seen_since = None                  # streak broken; restart the seen timer
            if self._absent_since is None:
                self._absent_since = now
            if self.committed and (now - self._absent_since) >= self.commit_s:
                self.committed = False               # UNLOCK: barcode gone for commit_s
                log.warning("[no-flip-place] barcode gone >= %.0fs -> UNLOCKED (no-flip latch dropped)",
                            self.commit_s)
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
            self._seen_since = None
            self._absent_since = None
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
        """Live snapshot for the GUI indicator. `hold_s` = how long after a decode the barcode is still
        reported as "seen" (the detector samples once per auto tick). Returns a dict:
            enabled         — armed?
            has_detector    — a detection cue is wired (vs detector=None)
            available       — the detector's backend is usable (zxing-cpp import ok); None if unknown
            barcode_seen    — a barcode was decoded within hold_s (visible right now)
            barcode_text    — payload of the most recent hit (or None)
            committed       — THE LOCK is set: the next place is committed to no-flip
            commit_progress — 0..1 toward the commit_s lock while a barcode is continuously held
            commit_s        — the hold-time threshold (s)
            fired           — the macro has run for the current committed cycle
            frames_ok       — the last scan got a camera frame (distinguishes "camera not feeding"
                              from "frames arriving but no code decoded"); None if unknown
            frame_cams      — which cameras delivered a frame on the last scan
        """
        det = self.detector
        if self._seen_since is not None and self.commit_s > 0:
            progress = min(1.0, (time.monotonic() - self._seen_since) / self.commit_s)
        else:
            progress = 0.0
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
            "commit_s": self.commit_s,
            "fired": self._fired,
            "frames_ok": (len(frame_cams) > 0) if frame_cams is not None else None,
            "frame_cams": list(frame_cams) if frame_cams else [],
        }

    def reset(self):
        """Drop the commit latch + timers (e.g. on auto stop / E-stop) so a restart begins fresh."""
        self.committed = False
        self._seen_since = None
        self._absent_since = None
        self._grip_closed_since = None
        self._fired = False
