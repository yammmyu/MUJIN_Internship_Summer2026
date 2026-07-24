"""Package-presence gate: pause auto inference (and park the arm at home) when there is no package to
work on, and resume once a package reappears.

Companion to no_flip_place.py. The SAME trained YOLO model now emits two classes:
  * "barcode"  -> drives the no-flip release cue (unchanged; see no_flip_place.YoloGate)
  * "package"  -> drives THIS gate: is there a package present to manipulate at all?

Behaviour (per operator decision — PAUSE ONLY WHEN IDLE):
    The pause engages ONLY while the gripper is EMPTY. During a grasp/place the arm and hand routinely
    occlude the package in the head camera, so presence is deliberately NOT checked mid-task — pausing
    there would interrupt every grasp. Rule, evaluated each auto tick:
      * holding an item (gripper closed) -> treated as "present", counters reset, never pauses.
      * empty-handed + package MISSED `absent_to_pause` scans in a row -> the controller homes the arm
        to the auto start pose and SKIPS inference (paused).
      * empty-handed + package SEEN `present_to_resume` scans in a row -> normal inference resumes.
    The two debounce counts tolerate flicker / a frame the package is momentarily lost.

FAIL-OPEN (safety): the gate can only pause when it is actually CAPABLE of detecting packages — the
model is loaded AND has the "package" class. If ultralytics/weights are missing, or the loaded model is
the barcode-only model with no "package" class, `capable` is False and the gate NEVER pauses, so the
robot keeps working. The feature therefore activates on its own once a package-capable model is trained
and dropped in at the YoloGate weights path — no code change needed.

The detector is a YoloGate filtered to the package class; pass model=<shared> to reuse the barcode
gate's already-loaded model (one model in memory). Integration is a single call-in at the top of
InferenceController._run_auto_inference (see _maybe_pause_for_package there).
"""

import logging

from real_world.no_flip_place import YoloGate, SCAN_CAMERAS, DEFAULT_WEIGHTS

log = logging.getLogger(__name__)

PACKAGE_CLASS = "package"      # model class name that means "a package is present" (case-insensitive)


class PackagePresenceGate:
    def __init__(self, cameras=SCAN_CAMERAS, *, weights=DEFAULT_WEIGHTS, conf=0.5,
                 package_class=PACKAGE_CLASS, absent_to_pause=10, present_to_resume=3,
                 detector=None, model=None, debug=False):
        self.package_class = str(package_class)
        # Detection port: a YoloGate restricted to the package class (case-insensitive). Reuses a shared
        # preloaded model when one is passed, so the barcode + package gates hold one model in memory.
        self.detector = detector if detector is not None else YoloGate(
            cameras=cameras, weights=weights, conf=conf, classes={self.package_class}, model=model)
        if hasattr(self.detector, "debug"):
            self.detector.debug = bool(debug)
        self.absent_to_pause = max(1, int(absent_to_pause))
        self.present_to_resume = max(1, int(present_to_resume))
        self.enabled = True           # live on/off: False -> the gate never pauses (see _maybe_pause_*)
        # Debounced presence belief + counters. Start "present" so a run never pre-emptively pauses
        # before the first few scans have had a chance to look.
        self.present = True
        self.paused = False           # currently pausing inference (arm homed, waiting for a package)
        self._absent = 0              # consecutive scans the package has been missed (toward pausing)
        self._present = 0             # consecutive scans the package has been seen (toward resuming)
        log.info("PackagePresenceGate ready: class=%r absent_to_pause=%d present_to_resume=%d capable=%s",
                 self.package_class, self.absent_to_pause, self.present_to_resume, self.capable)

    @property
    def capable(self) -> bool:
        """True ONLY if we can actually detect the package class — the sole condition under which the
        gate is allowed to pause. Missing model or a model without the package class => fail-open."""
        det = self.detector
        return bool(getattr(det, "available", False) and hasattr(det, "has_class")
                    and det.has_class(self.package_class))

    def scan(self, env) -> bool:
        """Run one package detection and update the debounced `present` belief. Returns `present`.
        Does not move the robot, so it is safe to call outside the auto loop (e.g. a GUI indicator)."""
        seen = bool(self.detector(env, None))
        if seen:
            self._present += 1
            self._absent = 0
            if not self.present and self._present >= self.present_to_resume:
                self.present = True
        else:
            self._absent += 1
            self._present = 0
            if self.present and self._absent >= self.absent_to_pause:
                self.present = False
        return self.present

    def on_busy(self):
        """Call when the robot is holding an item: treat as present, reset counters, drop any pause.
        (We never pause mid-task even though the arm may occlude the package.)"""
        self._absent = 0
        self._present = 0
        self.present = True
        self.paused = False

    def status(self, hold_s=1.5):
        """Live snapshot for a GUI indicator. Returns a dict:
            enabled      — armed?
            capable      — can actually detect the package class (else fail-open, never pauses)
            available    — the detector backend is usable (YOLO model loaded); None if unknown
            present      — debounced belief that a package is in view
            paused       — inference is currently paused (arm homed, waiting for a package)
            package_seen — a package was detected within hold_s (visible right now)
            last_text    — label+confidence of the most recent hit (or None)
            absent       — consecutive scans the package has been missed
            present_count— consecutive scans the package has been seen
        """
        det = self.detector
        return {
            "enabled": self.enabled,
            "capable": self.capable,
            "available": getattr(det, "available", None),
            "present": self.present,
            "paused": self.paused,
            "package_seen": bool(hasattr(det, "seen_within") and det.seen_within(hold_s)),
            "last_text": getattr(det, "last_text", None),
            "absent": self._absent,
            "present_count": self._present,
        }

    def reset(self):
        """Drop the pause + counters (e.g. on auto start / stop / E-stop) so a restart begins fresh
        assuming a package is present (it is re-decided live within the first few scans)."""
        self.present = True
        self.paused = False
        self._absent = 0
        self._present = 0
