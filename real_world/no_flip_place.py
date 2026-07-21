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

Detection cue — BARCODE:
    A package that shows a barcode is oriented correctly and must NOT be flipped. So the trigger here
    is "a camera sees a barcode on the grasped object" -> place it as-is. Detection uses the
    zxing-cpp library (`pip install zxing-cpp`) over the wrist + head frames; see BarcodeGate.
    It is injected as the default `detector`, but the port is still open: pass your own
    detector=callable(env, arm14)->bool (or None to disable) to swap the cue.

Integration (one call-in in InferenceController._run_auto_inference, top of loop):
    nfp = getattr(self, "no_flip_place", None)
    if nfp is not None and nfp.maybe_trigger(env):
        continue                      # macro ran this cycle; skip predicting
"""

import logging
import os
import time

import numpy as np

log = logging.getLogger(__name__)

# Cameras to scan for a barcode, in order. The right wrist eye-in-hand camera looks straight at the
# object the right gripper is holding; the head camera is a wider fallback. (Names per humanoid_env.)
BARCODE_CAMERAS = ("hand_right", "head")

# Placeholder: reuses the flip variant's recorded path so the macro is runnable today. Swap this for a
# dedicated no-flip placement path once one is recorded (the release EE point likely differs).
DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "assets", "flip_release_path.npy")

R_GRIP_IDX = 1          # right channel in a [gl, gr] gripper command


class BarcodeGate:
    """Detection port for the no-flip case: "does a camera see a barcode on the grasped object?".

    A visible barcode means the package is oriented right-side-up and must NOT be flipped, so the
    macro should place it as-is. Uses the zxing-cpp library (`pip install zxing-cpp`), which decodes
    the common 1-D/2-D symbologies (Code128, EAN/UPC, QR, ...) straight from an RGB numpy frame and is
    far more robust than OpenCV's bundled detector. A barcode counts as SEEN when zxing decodes one;
    pass symbologies=("EAN13", ...) / a zxingcpp.BarcodeFormat to restrict which kinds count.

    zxing-cpp is imported lazily so the module still loads on a machine without it (the gate then
    logs once and never fires, exactly like a disabled detector). Callable, so it drops straight into
    NoFlipPlaceMacro(detector=...). The env's per-tick debounce (settle_s) already requires the
    detection to hold across frames before the macro fires.
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
            return False
        for name in self.cameras:
            frame = env.get_frame(name)               # latest RGB frame (copy), or None while warming up
            if frame is None:
                continue
            try:
                if self.symbologies is not None:
                    results = self._zxing.read_barcodes(frame, formats=self.symbologies)
                else:
                    results = self._zxing.read_barcodes(frame)
            except Exception as e:                    # never let a decode hiccup kill the auto loop
                if not self._warned:
                    log.warning("[no-flip-place] barcode read failed on '%s': %s", name, e)
                    self._warned = True
                continue
            for r in results:
                if r.text:                            # a barcode was decoded -> it is seen
                    self.last_text = r.text            # publish live state for the GUI indicator
                    self.last_format = str(r.format)
                    self.last_camera = name
                    self.last_seen_monotonic = time.monotonic()
                    log.info("[no-flip-place] barcode seen on '%s' (%s: %s) -> place without flipping",
                             name, r.format, r.text)
                    return True
        return False


class NoFlipPlaceMacro:
    _UNSET = object()

    def __init__(self, path=DEFAULT_PATH, *,
                 detector=_UNSET,      # PORT: callable(env, arm14)->bool. default = BarcodeGate; None disables
                 settle_s=0.3,         # the place condition must hold this long before firing (debounce)
                 vel_frac=0.5,         # streaming speed = fraction of MAX joint velocity
                 release_settle_s=0.4, # pause after opening the gripper before withdrawing
                 open_grip=0.0):       # 0 = open (release)
        self.path = np.load(path).astype(np.float64)      # (M, 14) absolute-joint waypoints
        if self.path.ndim != 2 or self.path.shape[1] != 14:
            raise ValueError(f"no-flip release path must be (M,14); got {self.path.shape}")
        # default cue = barcode seen -> don't flip; pass detector=None to disable, or your own callable
        self.detector = BarcodeGate() if detector is self._UNSET else detector
        self.settle_s = float(settle_s)
        self.vel_frac = float(vel_frac)
        self.release_settle_s = float(release_settle_s)
        self.open_grip = float(open_grip)
        self.enabled = True           # live on/off: False -> maybe_trigger is a no-op (macro never runs)
        self._fired = False           # already ran for the current closed episode?
        self._above_since = None      # monotonic time the place condition first held (settle timer)
        log.info("NoFlipPlaceMacro ready: %d waypoints, settle=%.1fs vel=%.0f%% max, detector=%s",
                 len(self.path), self.settle_s, self.vel_frac * 100,
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

    def maybe_trigger(self, env) -> bool:
        """Loop hook: True => the macro ran this cycle (caller should skip predicting). Fires ONCE per
        closed grasp, when the right gripper is commanded closed AND _should_place() (the PORT) holds
        for settle_s. Resets on release. No-op (returns False) while disabled or while the detection
        port is not yet wired, so the operator can turn the macro off live and it is safe by default."""
        if not self.enabled:
            return False
        grip = getattr(env, "_last_grip_cmd", None)
        grip_closed = grip is not None and grip[R_GRIP_IDX] >= 1
        if not grip_closed:                          # released -> disarm; ready for the next grasp
            self._fired = False
            self._above_since = None
            return False
        arm14 = env._read_arm14()
        if arm14 is None:
            return False
        if self._fired:
            return False
        if not self._should_place(env, arm14):       # PORT: detection gate
            self._above_since = None                 # condition not met -> restart the settle timer
            return False
        now = time.monotonic()
        if self._above_since is None:
            self._above_since = now
            return False
        if now - self._above_since < self.settle_s:
            return False
        self._fired = True
        log.warning("[no-flip-place] place condition met (held >= %.1fs) -> stop auto, run macro",
                    self.settle_s)
        self._run(env, np.asarray(arm14, dtype=np.float64))
        return True

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
        reported as "seen" (the detector only samples once per auto tick). Returns a dict:
            enabled       — armed?
            has_detector  — a detection cue is wired (vs detector=None)
            available     — the detector's backend is usable (zxing-cpp import ok); None if unknown
            barcode_seen  — a barcode was decoded within hold_s (the "place flat now" condition)
            barcode_text  — payload of the most recent hit (or None)
            fired         — the macro has already run for the current grasp
        """
        det = self.detector
        return {
            "enabled": self.enabled,
            "has_detector": det is not None,
            "available": getattr(det, "available", None),
            "barcode_seen": bool(det is not None and hasattr(det, "seen_within")
                                 and det.seen_within(hold_s)),
            "barcode_text": getattr(det, "last_text", None),
            "fired": self._fired,
        }

    def reset(self):
        """Disarm (e.g. on auto stop / E-stop) so a restart doesn't immediately re-fire."""
        self._fired = False
        self._above_since = None
