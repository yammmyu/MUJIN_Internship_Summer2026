"""Hardware-free demo backend for the Humanoid Control Console.

Everything here is a *stand-in* for the real robot stack (``a2d_sdk`` SDK, ROS
``rclpy``, the policy inference server, the PyBullet sim). It exists so the exact
same GUI can run on any laptop — no robot, no SDK, no network — for recorded
demos, screenshots, and UI iteration.

Design rules:
  * These objects expose ONLY the method/attribute surface the GUI calls
    (see the grep in the panels). A permissive ``__getattr__`` returns a safe
    no-op for anything unforeseen, so the demo can never crash the UI on a call
    a real object would have.
  * Data is synthetic but *alive*: joints breathe, cameras stream a rendered
    scene, the substep monitor scrolls, and the evaluation dashboard fills in
    over time — so the console looks like it is really driving a robot.
  * NOTHING here touches hardware. It is only constructed when the app is
    launched with ``--demo``; the real launch path is untouched.
"""

import math
import pathlib
import threading
import time
from collections import deque

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
#  Shared synthetic clock / motion model                                       #
# --------------------------------------------------------------------------- #
_T0 = time.monotonic()


def _clock():
    return time.monotonic() - _T0


def _joints14(phase):
    """A smooth, plausible 14-joint pose [left7, right7] as a function of phase."""
    q = np.zeros(14, dtype=np.float64)
    for j in range(7):
        amp = 0.35 - 0.03 * j
        q[j] = amp * math.sin(phase * 0.6 + j * 0.7) + (0.2 if j in (1, 3) else 0.0)
        q[j + 7] = amp * math.sin(phase * 0.6 + j * 0.7 + math.pi) - (0.2 if j in (1, 3) else 0.0)
    return q


def _grip_at(phase):
    """Binary [left, right] grip that closes/opens on a slow cycle."""
    gl = 1 if (phase % 8.0) > 4.0 else 0
    gr = 1 if (phase % 12.0) > 7.0 else 0
    return [gl, gr]


# --------------------------------------------------------------------------- #
#  Synthetic camera feed                                                       #
# --------------------------------------------------------------------------- #
class _CameraFeed:
    """Renders a believable live camera image per named view with cv2 drawing."""

    _SCENES = {
        "head":        dict(bg=(38, 44, 58), tint=(70, 90, 130), label="HEAD"),
        "hand_left":   dict(bg=(30, 40, 36), tint=(60, 110, 90), label="L-WRIST"),
        "hand_right":  dict(bg=(40, 34, 44), tint=(110, 70, 110), label="R-WRIST"),
        "head_depth":  dict(bg=(20, 20, 28), tint=(40, 60, 120), label="DEPTH"),
        "head_center_fisheye": dict(bg=(34, 40, 50), tint=(80, 90, 110), label="FISHEYE"),
    }

    def frame(self, name, w=640, h=480):
        scene = self._SCENES.get(name, self._SCENES["head"])
        t = _clock()
        img = np.zeros((h, w, 3), dtype=np.uint8)
        # Vertical gradient background.
        top = np.array(scene["bg"], dtype=np.float32)
        bot = top * 0.55
        for y in range(h):
            img[y, :] = top + (bot - top) * (y / h)
        # A "table" band + a moving target block the arm is working over.
        cv2.rectangle(img, (0, int(h * 0.68)), (w, h), tuple(int(c) for c in bot * 0.8), -1)
        cx = int(w * 0.5 + math.sin(t * 0.5) * w * 0.18)
        cy = int(h * 0.6 + math.cos(t * 0.7) * h * 0.04)
        cv2.rectangle(img, (cx - 34, cy - 26), (cx + 34, cy + 26),
                      tuple(int(c) for c in scene["tint"]), -1)
        cv2.rectangle(img, (cx - 34, cy - 26), (cx + 34, cy + 26), (230, 230, 235), 1)
        # Reticle.
        cv2.line(img, (w // 2 - 16, h // 2), (w // 2 + 16, h // 2), (200, 210, 225), 1)
        cv2.line(img, (w // 2, h // 2 - 16), (w // 2, h // 2 + 16), (200, 210, 225), 1)
        # Light sensor noise so it reads as a live feed.
        noise = np.random.randint(0, 10, (h, w, 1), dtype=np.uint8)
        img = cv2.add(img, np.repeat(noise, 3, axis=2))
        # HUD overlay.
        cv2.rectangle(img, (0, 0), (w, 26), (18, 22, 30), -1)
        cv2.putText(img, f"{scene['label']}  live", (10, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 200, 255), 1, cv2.LINE_AA)
        cv2.putText(img, time.strftime("%H:%M:%S"), (w - 92, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 210, 225), 1, cv2.LINE_AA)
        cv2.putText(img, "DEMO", (w - 92, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 150, 200), 1, cv2.LINE_AA)
        return img


# --------------------------------------------------------------------------- #
#  No-op safety net                                                            #
# --------------------------------------------------------------------------- #
class _Stub:
    """Base that returns a harmless no-op for any attribute the GUI might call
    but this demo hasn't implemented, so a missed call never raises."""

    def __getattr__(self, name):
        def _noop(*a, **k):
            return None
        return _noop


# --------------------------------------------------------------------------- #
#  Robot / controller stand-ins                                                #
# --------------------------------------------------------------------------- #
class DemoRobot(_Stub):
    def __init__(self):
        self._grip = [0.0, 0.0]

    def arm_joint_states(self):
        return _joints14(_clock()), time.time()

    def gripper_states(self):
        return np.array(self._grip, dtype=np.float64), time.time()

    def move_gripper(self, lr):
        self._grip = [float(lr[0]), float(lr[1])]

    def waist_joint_states(self):
        return np.array([0.05 * math.sin(_clock() * 0.3)]), time.time()

    def head_joint_states(self):
        return np.array([0.0, 0.0]), time.time()

    def move_head(self, *a, **k):
        return None

    def shutdown(self):
        return None


def _ee_frame(base_x, base_y, phase, sign):
    px = base_x + 0.05 * math.sin(phase * 0.5)
    py = base_y + 0.04 * math.cos(phase * 0.6) * sign
    pz = 0.22 + 0.03 * math.sin(phase * 0.7)
    return {
        "position": {"x": round(px, 4), "y": round(py, 4), "z": round(pz, 4)},
        "orientation": {"quaternion": {"x": 0.0, "y": 0.707, "z": 0.0, "w": 0.707}},
    }


class DemoRobotController(_Stub):
    def get_motion_status(self):
        ph = _clock()
        return {
            "error": {"has_error": False, "message": ""},
            "collisions": [],
            "frames": {
                "arm_left_link7": _ee_frame(0.28, 0.20, ph, +1),
                "arm_right_link7": _ee_frame(0.28, -0.20, ph, -1),
            },
        }

    def trajectory_tracking_control(self, *a, **k):
        return None


class DemoSlam(_Stub):
    pass


class DemoWheelController(_Stub):
    def destroy_node(self):
        return None


class DemoDummyServer(_Stub):
    def start(self):
        return None

    def stop(self):
        return None

    def set_image(self, *a, **k):
        return None


# --------------------------------------------------------------------------- #
#  Environment stand-in                                                        #
# --------------------------------------------------------------------------- #
class DemoEnv(_Stub):
    """Mirrors the slice of ``HumanoidEnv`` the GUI touches, with a rolling
    substep queue and on-demand synthetic camera frames."""

    ALLOWED = ["head", "head_depth", "hand_left", "hand_right", "head_center_fisheye"]

    def __init__(self, output_dir="recordings"):
        self.output_dir = pathlib.Path(output_dir)
        self.sim = None
        self.speed_scale = 1.0
        self.append_ahead_rows = 4
        self._feed = _CameraFeed()
        self._requested = set()
        self._staged = deque()
        self._estopped = False
        self._auto = False
        self._recording = False

    # ---- cameras ----
    @property
    def cameras(self):
        return list(self.ALLOWED)

    def request(self, name):
        self._requested.add(name)

    def get_frame(self, name):
        self._requested.add(name)
        return self._feed.frame(name)

    def active_cameras(self):
        return list(self._requested)

    def get_intrinsics(self, name):
        return None

    # ---- lifecycle ----
    def start(self, *a, **k):
        return None

    def stop(self, *a, **k):
        return None

    def set_seed(self, *a, **k):
        return None

    # ---- execution knobs ----
    def set_speed_scale(self, s):
        self.speed_scale = max(1e-3, float(s))
        sub = max(1, round(10.0 / self.speed_scale))
        return self.speed_scale, sub

    # ---- substep monitor sources ----
    def _rows(self, n, base_phase):
        rows = []
        for i in range(n):
            ph = base_phase + i * 0.25
            rows.append((_joints14(ph), _grip_at(ph)))
        return rows

    def robot_q_preview(self, n=10):
        # Auto path: a continuously advancing stream keyed on wall time.
        if not self._auto or self._estopped:
            return []
        return self._rows(n, _clock() * 4.0)

    def staged_preview(self, n=10):
        return list(self._staged)[:n]

    @property
    def staged_substeps(self):
        return len(self._staged)

    def _stage(self, n):
        base = _clock() * 4.0 + len(self._staged) * 0.25
        for i in range(n):
            ph = base + i * 0.25
            self._staged.append((_joints14(ph), _grip_at(ph)))

    def release_n_substeps(self, n):
        if self._estopped:
            return 0
        k = min(n, len(self._staged))
        for _ in range(k):
            self._staged.popleft()
        return k

    def release_remaining_substeps(self):
        if self._estopped:
            return 0
        k = len(self._staged)
        self._staged.clear()
        return k

    # ---- safety ----
    def lock_robot(self):
        self._estopped = True
        self._auto = False
        self._staged.clear()

    def reset_estop(self):
        self._estopped = False

    # ---- recording ----
    def start_recording(self, episode_name=None):
        self._recording = True

    def stop_recording(self):
        self._recording = False


# --------------------------------------------------------------------------- #
#  Inference controller stand-in                                               #
# --------------------------------------------------------------------------- #
class DemoInference(_Stub):
    """Simulates the policy loop. Drives the DemoEnv so the manual step-through
    and auto-run both produce visible motion, and grows a live eval session."""

    def __init__(self, env, eval_writer=None):
        self.env = env
        self._eval = eval_writer
        self.te_radius = 3
        self.te_sigma = 1.5
        self.te_m = 0.15
        self.te_buffer_len = 4
        self.is_auto_inference = False
        self._steps = 0
        # No-flip barcode macro stand-in: a non-None marker keeps the GUI checkbox enabled, and
        # no_flip_place_status() below reports a barcode as "seen" for part of each cycle so the live
        # indicator pill visibly blinks green in the demo.
        self.no_flip_place = object()
        self._no_flip_enabled = True

    @property
    def no_flip_place_enabled(self):
        return self._no_flip_enabled

    @no_flip_place_enabled.setter
    def no_flip_place_enabled(self, v):
        self._no_flip_enabled = bool(v)

    def no_flip_place_status(self):
        # Synthetic 14 s cycle so the pill visibly walks: watching -> counting -> LOCKED (persists a
        # few seconds after the label leaves view) -> reset.
        commit_count = 20
        t = _clock() % 14.0
        seen = self._no_flip_enabled and t < 8.0        # label visible for the first 8 s
        committed = self._no_flip_enabled and 6.0 <= t < 12.0   # latches at 6 s, holds past 8 s, resets at 12 s
        seen_count = int(min(t, 6.0) / 6.0 * commit_count) if seen else 0
        return {
            "enabled": self._no_flip_enabled,
            "has_detector": True,
            "available": True,
            "barcode_seen": seen,
            "barcode_text": "DEMO-LABEL 220x150" if (seen or committed) else None,
            "committed": committed,
            "commit_progress": seen_count / commit_count,
            "seen_count": seen_count,
            "commit_count": commit_count,
            "fired": False,
        }

    def start(self):
        return None

    def stop(self):
        self.is_auto_inference = False
        if self._eval:
            self._eval.stop()

    # ---- tuning ----
    def set_smoothing(self, radius=None, sigma=None, m=None):
        if radius is not None:
            self.te_radius = int(radius)
        if sigma is not None:
            self.te_sigma = float(sigma)
        if m is not None:
            self.te_m = float(m)
        return self.te_radius, self.te_sigma, self.te_m

    def set_buffer_len(self, n):
        if n is not None:
            self.te_buffer_len = int(n)
        return self.te_buffer_len

    # ---- manual step-through ----
    def inference_once(self):
        self._steps = 16
        return True

    def steps_remaining(self):
        return self._steps

    def execute_inference_result(self, once=False):
        if self._steps <= 0:
            return False, "no prediction — run inference first"
        take = 1 if once else self._steps
        take = min(take, self._steps)
        self.env._stage(take)
        self._steps -= take
        return True, "sim-validated (demo)"

    # ---- auto run ----
    def auto_inference(self, stop=False):
        if stop:
            self.is_auto_inference = False
            self.env._auto = False
            if self._eval:
                self._eval.stop()
            return
        self.is_auto_inference = True
        self.env._auto = True
        self.env._estopped = False
        if self._eval:
            self._eval.start()


# --------------------------------------------------------------------------- #
#  Live evaluation-session generator (feeds the eval dashboard)                #
# --------------------------------------------------------------------------- #
class DemoEvalWriter:
    """Appends realistic trial outcomes to an eval JSONL over time, so the
    evaluation dashboard visibly fills in during an auto run — exactly what a
    live/recorded demo wants to show."""

    def __init__(self, path, period_s=3.0, seed_trials=6):
        import json
        self._json = json
        self.path = pathlib.Path(path)
        self.period = period_s
        self._stop = threading.Event()
        self._thread = None
        self._running = False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Seed with a short history so the dashboard is never empty on first paint.
        self.path.write_text("")
        for _ in range(seed_trials):
            self._append_trial()

    _CATS = ["success", "grasp-miss", "dropped", "misplaced", "collision", "other"]
    _WEIGHTS = [0.72, 0.10, 0.06, 0.05, 0.04, 0.03]

    def _append_trial(self):
        import datetime as dt
        outcome = np.random.choice(self._CATS, p=self._WEIGHTS)
        success = outcome == "success"
        rec = {
            "task": "flip_place",
            "outcome": str(outcome),
            "duration_s": round(float(np.random.normal(11.5, 2.2)), 2),
            "grasp_misses": int(np.random.choice([0, 0, 0, 1, 2], p=[.62, .12, .08, .12, .06])),
            "grasp_src": "log",
            "ts": dt.datetime.now().isoformat(timespec="seconds"),
        }
        with self.path.open("a") as f:
            f.write(self._json.dumps(rec) + "\n")

    def start(self):
        if self._running:
            return
        self._running = True
        self._stop.clear()

        def _loop():
            while not self._stop.wait(self.period):
                self._append_trial()

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        self._stop.set()
