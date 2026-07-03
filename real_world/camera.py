"""Dynamic camera subscription hub for HumanoidEnv.

Extracted from HumanoidEnv so the env stays a thin coordinator. The hub owns every
camera subscription and its own lock, and manages one SDK camera object per camera
*dynamically*: a camera is SUBSCRIBED (costs DDS bandwidth) only while a consumer has
``request``ed it recently or it is pinned; it is evicted (object closed) after
``idle_timeout``. Using one object per camera means toggling one never disturbs the
others' streams.

Threading model (unchanged from the in-env version):
  * ``capture_tick`` runs on the env's collect thread and is the SINGLE mutator of the
    live-camera set and the frame buffers.
  * ``request`` / ``get_frame`` / ``active_cameras`` / ``get_intrinsics`` / ``snapshot``
    run on other threads and take the hub lock.

The SDK camera class is INJECTED (``camera_cls``) so this module never imports
``a2d_sdk`` — it stays importable on a sim-only machine and unit-testable with a fake.
"""

import copy
import threading
import time


# A camera auto-switches OFF if no consumer has requested it within this window.
CAMERA_IDLE_TIMEOUT = 5.0

# Every camera name the SDK knows about. A camera is only SUBSCRIBED (i.e. costs DDS
# bandwidth) while a live camera object is held for it. This list is just the set of
# names a consumer is allowed to request. (More angles exist; limited to these for simplicity.)
KNOWN_CAMERAS = ["head", "head_depth", "hand_left", "hand_right", "head_center_fisheye"]

# Fallback intrinsics when the SDK can't supply them (ported from
# gui/camera_panel.py:get_default_camera_intrinsics so the hub owns intrinsics).
_DEFAULT_INTRINSICS = {
    "head": {'width': 1280, 'height': 800, 'fx': 900.0, 'fy': 900.0,
             'cx': 640.0, 'cy': 400.0, 'distortion': [0.0, 0.0, 0.0, 0.0, 0.0]},
    "head_depth": {'width': 1280, 'height': 800, 'fx': 900.0, 'fy': 900.0,
                   'cx': 640.0, 'cy': 400.0, 'distortion': [0.0, 0.0, 0.0, 0.0, 0.0]},
    "hand_left": {'width': 320, 'height': 240, 'fx': 300.0, 'fy': 300.0,
                  'cx': 160.0, 'cy': 120.0, 'distortion': [0.0, 0.0, 0.0, 0.0, 0.0]},
    "hand_right": {'width': 320, 'height': 240, 'fx': 300.0, 'fy': 300.0,
                   'cx': 160.0, 'cy': 120.0, 'distortion': [0.0, 0.0, 0.0, 0.0, 0.0]},
    "head_center_fisheye": {'width': 640, 'height': 480, 'fx': 400.0, 'fy': 400.0,
                            'cx': 320.0, 'cy': 240.0, 'distortion': [0.1, -0.1, 0.0, 0.0, 0.0]},
}


def default_camera_intrinsics(name):
    """Default intrinsics for a camera (falls back to 'head' for unknown names)."""
    return _DEFAULT_INTRINSICS.get(name, _DEFAULT_INTRINSICS["head"])


class CameraHub:
    """Owns dynamic per-camera SDK subscriptions + the latest-frame buffers."""

    def __init__(self, camera_cls, allowed=KNOWN_CAMERAS, pinned=(),
                 idle_timeout=CAMERA_IDLE_TIMEOUT):
        self._Camera = camera_cls           # SDK camera class (or None on a sim-only machine)
        # Names a consumer is allowed to request (validation only — not subscribed).
        self.allowed = list(allowed)
        # "Pinned" cameras stay ON for the hub's whole life (never idle-evicted), e.g.
        # data-collection cameras. Empty for the GUI -> nothing on at launch.
        self._pinned = set(pinned)
        self.idle_timeout = idle_timeout

        self._lock = threading.Lock()
        self._active = set()                 # cameras requested recently
        self._last_requested = {}            # name -> time.monotonic()
        self._cams = {}                      # name -> live camera object (the DDS subscription)
        self._frames = {}                    # name -> rolling [prev, cur]
        self._intrinsics = {}                # name -> intrinsics dict (lazy)
        self._unknown_warned = set()         # bad camera names already warned (once each)

    # ===================== consumer-facing camera switch =====================
    def request(self, name):
        """Mark a camera as wanted this cycle and switch it ON.

        Idempotent and cheap — consumers call this every loop tick to keep a camera
        alive. Names outside the allowed set are ignored. Logs ONLY on the OFF->ON
        transition (this is a per-tick hot path).
        """
        if name not in self.allowed:
            if name not in self._unknown_warned:      # warn once per bad name, never per tick
                self._unknown_warned.add(name)
                print(f"[CameraHub] camera name not recognized: {name}")
            return
        with self._lock:
            self._last_requested[name] = time.monotonic()
            if name not in self._active:              # OFF->ON transition only
                self._active.add(name)
                print(f"[CameraHub] camera ON: {name} -> {sorted(self._active)}")

    def get_frame(self, name):
        """request(name) + return the latest single frame (copy), or None until the
        collect loop has fetched the camera at least once (just switched on / warming up)."""
        self.request(name)
        with self._lock:
            pair = self._frames.get(name)
            return copy.deepcopy(pair[-1]) if pair else None

    def active_cameras(self):
        """Names currently SUBSCRIBED (a live camera object exists -> streaming). This is
        what the GUI's live indicator shows."""
        with self._lock:
            return sorted(self._cams.keys())

    def get_intrinsics(self, name):
        """Camera intrinsics, fetched once from the live SDK object then cached. Falls back
        to defaults when the camera isn't currently subscribed (intrinsics are static)."""
        with self._lock:
            cached = self._intrinsics.get(name)
            cam = self._cams.get(name)
        if cached is not None:
            return cached
        info = None
        if cam is not None:                           # only query a live subscription
            try:
                if hasattr(cam, 'get_camera_info'):
                    info = cam.get_camera_info(name)
                elif hasattr(cam, 'get_intrinsics'):
                    info = cam.get_intrinsics(name)
            except Exception as e:
                print(f"[CameraHub] get_intrinsics {name}: {e}")
                info = None
        if not info:
            # don't cache the default — let a real value replace it once the camera is on
            return default_camera_intrinsics(name)
        with self._lock:
            self._intrinsics[name] = info
        return info

    def snapshot(self, names):
        """{name: deepcopy of the rolling [prev, cur] pair, or None if not captured yet}.
        The env's get_obs uses this to build one inference observation."""
        with self._lock:
            return {n: (copy.deepcopy(self._frames[n]) if self._frames.get(n) else None)
                    for n in names}

    def have_frames(self, names):
        """True once every requested camera has produced at least one frame pair."""
        with self._lock:
            return all(self._frames.get(n) is not None for n in names)

    # ===================== pinning (used by the recording path) =====================
    def pin(self, names):
        """Keep `names` subscribed regardless of request recency (never idle-evicted)."""
        with self._lock:
            self._pinned |= set(names)

    def unpin(self, names):
        """Drop `names` from the pinned set (they go back to idle-eviction rules)."""
        with self._lock:
            self._pinned -= set(names)

    # ===================== collect-thread driver =====================
    def capture_tick(self):
        """One producer tick (collect thread only): evict idle cameras, reconcile the live
        subscription set to (recently-requested | pinned), read the latest frame per live
        camera, publish the rolling pairs, and RETURN this tick's fresh frames dict
        ({name: [prev, cur]}) for the caller's freshness check + recording."""
        now_mono = time.monotonic()
        with self._lock:
            stale = [n for n, t in self._last_requested.items()
                     if now_mono - t > self.idle_timeout and n not in self._pinned]
            for n in stale:
                self._active.discard(n)
                self._last_requested.pop(n, None)
            desired = set(self._active) | self._pinned

        # Open/close camera objects to match desired (controls bandwidth).
        self._reconcile(desired)

        fresh = self._read_frames(desired)
        with self._lock:
            for name, pair in fresh.items():
                self._frames[name] = pair
        return fresh

    def _reconcile(self, desired):
        """Make the set of live camera objects match `desired`. Opens a dedicated camera
        object per newly-wanted camera (the DDS subscription that costs bandwidth) and closes
        the object for any camera no longer wanted. Collect-thread only, so self._cams has a
        single mutator; reads elsewhere take the lock."""
        current = set(self._cams.keys())
        for name in current - desired:
            cam = self._cams.pop(name)
            try:
                cam.close()
            except Exception as e:
                print(f"[CameraHub] camera.close({name}) failed: {e}")
            with self._lock:
                self._frames.pop(name, None)          # no stale frame on re-activation
                self._intrinsics.pop(name, None)
            print(f"[CameraHub] camera unsubscribed (OFF): {name}")
        for name in desired - current:
            if self._Camera is None:                  # sim-only machine: no SDK to subscribe with
                continue
            try:
                cam = self._Camera([name])            # <-- the per-camera DDS subscription
            except Exception as e:
                print(f"[CameraHub] camera open({name}) failed: {e}")
                continue
            with self._lock:
                self._cams[name] = cam
            print(f"[CameraHub] camera subscribed (ON): {name}")

    def _read_frames(self, active):
        """Latest frame per live camera as a rolling [prev, cur] pair. get_latest_image
        returns (image, timestamp) and the first frame is None (SDK note 9.6), so we skip
        until a real frame arrives. Runs on the collect thread (sole writer of _frames)."""
        out = {}
        for name in active:
            cam = self._cams.get(name)
            if cam is None:                           # just requested; object opens next tick
                continue
            image, _ = cam.get_latest_image(name)
            if image is None:
                continue
            prev = self._frames.get(name)
            out[name] = [prev[-1], image] if prev else [image, image]
        return out

    # ===================== lifecycle =====================
    def close_all(self):
        """Close every live camera subscription (the hub owns all of them)."""
        for name, cam in list(self._cams.items()):
            try:
                cam.close()
            except Exception as e:
                print(f"[CameraHub] camera.close({name}) failed: {e}")
        self._cams.clear()
