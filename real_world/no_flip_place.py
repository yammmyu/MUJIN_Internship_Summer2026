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

Detection cue — BARCODE, with a commit LATCH:
    A package that shows a barcode is oriented correctly and must NOT be flipped. The detector scans
    CONTINUOUSLY every auto tick (not gated by the grasp). When a barcode is seen for an unbroken
    ``commit_s`` (default 6 s) the macro LATCHES: it now knows the next place is no-flip. The latch
    holds even if the barcode then leaves view (the gripper may occlude it during the grasp), and it
    fires the scripted place once the object is actually grasped, then CLEARS so the next object must
    earn its own 6-s barcode hold. Detection uses the zxing-cpp library (`pip install zxing-cpp`) over
    the wrist + head frames; see BarcodeGate. The port is open: pass your own
    detector=callable(env, arm14)->bool (or None to disable) to swap the cue.

    "Continuous" means every tick of the auto loop; the scan is still driven by that loop, so it runs
    only while auto-run is active (there is no separate always-on scanner thread).

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
                 commit_s=6.0,         # a barcode must be seen CONTINUOUSLY this long to LATCH no-flip
                 vel_frac=0.5,         # streaming speed = fraction of MAX joint velocity
                 release_settle_s=0.4, # pause after opening the gripper before withdrawing
                 open_grip=0.0):       # 0 = open (release)
        self.path = np.load(path).astype(np.float64)      # (M, 14) absolute-joint waypoints
        if self.path.ndim != 2 or self.path.shape[1] != 14:
            raise ValueError(f"no-flip release path must be (M,14); got {self.path.shape}")
        # default cue = barcode seen -> don't flip; pass detector=None to disable, or your own callable
        self.detector = BarcodeGate() if detector is self._UNSET else detector
        self.commit_s = float(commit_s)
        self.vel_frac = float(vel_frac)
        self.release_settle_s = float(release_settle_s)
        self.open_grip = float(open_grip)
        self.enabled = True           # live on/off: False -> maybe_trigger is a no-op (macro never runs)
        # Continuous-scan + commit-latch state:
        self.committed = False        # THE LOCK: barcode held commit_s -> "the next place is no-flip".
                                      #   Persists even if the barcode later leaves view; cleared only
                                      #   when the macro fires (or on reset()).
        self._seen_since = None       # monotonic time the current unbroken barcode streak began
        self._fired = False           # macro ran for the current committed cycle (for status display)
        log.info("NoFlipPlaceMacro ready: %d waypoints, commit=%.0fs vel=%.0f%% max, detector=%s",
                 len(self.path), self.commit_s, self.vel_frac * 100,
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
        """Loop hook: True => the macro ran this cycle (caller should skip predicting). Two stages:

          1. CONTINUOUS scan — every tick (NOT gated by the grasp) the detector runs; a barcode seen
             for an unbroken commit_s LATCHES ``self.committed`` = "the next place is no-flip". The
             latch persists even if the barcode then leaves view (e.g. the gripper occludes it).
          2. FIRE — once committed AND the object is grasped (right gripper commanded closed), the
             scripted place runs; the latch then CLEARS so the next object must earn its own commit.

        No-op (returns False) while disabled or while no detector is wired, so it is safe by default."""
        if not self.enabled:
            return False
        arm14 = env._read_arm14()

        # 1. continuous scan + commit latch
        now = time.monotonic()
        if self._should_place(env, arm14):           # PORT: a barcode is visible this tick
            if self._seen_since is None:
                self._seen_since = now
            if not self.committed and (now - self._seen_since) >= self.commit_s:
                self.committed = True                # LATCH: hold survived commit_s
                self._fired = False                  # fresh committed cycle
                log.warning("[no-flip-place] barcode held >= %.0fs -> LOCKED: next place is no-flip",
                            self.commit_s)
        else:
            self._seen_since = None                  # streak broken; the LATCH (if set) stays

        # 2. fire once committed and something is actually grasped
        grip = getattr(env, "_last_grip_cmd", None)
        grip_closed = grip is not None and grip[R_GRIP_IDX] >= 1
        if self.committed and grip_closed and arm14 is not None:
            log.warning("[no-flip-place] committed + grasped -> stop auto, run no-flip place")
            self._run(env, np.asarray(arm14, dtype=np.float64))
            self.committed = False                   # lock resets after the macro fires
            self._seen_since = None
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
        """
        det = self.detector
        if self._seen_since is not None and self.commit_s > 0:
            progress = min(1.0, (time.monotonic() - self._seen_since) / self.commit_s)
        else:
            progress = 0.0
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
        }

    def reset(self):
        """Drop the commit latch + streak (e.g. on auto stop / E-stop) so a restart begins fresh."""
        self.committed = False
        self._seen_since = None
        self._fired = False
