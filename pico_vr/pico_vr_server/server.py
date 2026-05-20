"""
Dummy server side of the Pico VR example.

Two listening sockets:

* **Downstream** (``port``): for each accepted client, pushes an RGB image
  (as JPEG) and a synthetic :class:`JointState` at a configurable rate. The
  image is loaded once from ``example.png`` (next to this file by default)
  and the same JPEG bytes are reused every tick. Both message types are
  multiplexed onto the same connection using the shared framing in
  :mod:`xr_examples.pico_vr_common.protocol`.
* **Upstream** (``port2``): receives joint-state messages from a client and
  hands them to a user-supplied callback (the default just logs them).
"""

import logging
import math
import os
import socket
import threading
import time
from io import BytesIO
from typing import Callable, List, Optional, Tuple

from PIL import Image


def _quat_to_yaw_pitch_roll_deg(qx: float, qy: float, qz: float, qw: float
                                ) -> Tuple[float, float, float]:
    """Convert an OpenXR quaternion to intrinsic Y-X-Z Tait-Bryan angles.

    Returns ``(yaw, pitch, roll)`` in degrees, where (with OpenXR's Y-up
    convention) yaw is rotation about the vertical Y axis, pitch about the
    lateral X axis (positive = looking up), and roll about the forward Z
    axis (positive = tilting right).
    """
    sin_p = 2.0 * (qw * qx - qz * qy)
    if abs(sin_p) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sin_p)
    else:
        pitch = math.asin(sin_p)
    yaw = math.atan2(
        2.0 * (qw * qy + qx * qz),
        1.0 - 2.0 * (qx * qx + qy * qy),
    )
    roll = math.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qz * qz + qx * qx),
    )
    return math.degrees(yaw), math.degrees(pitch), math.degrees(roll)


def _format_pose(label: str, slab: List[float]) -> str:
    px, py, pz, qx, qy, qz, qw = slab
    yaw, pitch, roll = _quat_to_yaw_pitch_roll_deg(qx, qy, qz, qw)
    return (
        f"{label} pos=({px:+.3f},{py:+.3f},{pz:+.3f}) "
        f"ypr=({yaw:+6.1f}°,{pitch:+6.1f}°,{roll:+6.1f}°)"
    )


def _fmt_trigger(value: Optional[bool]) -> str:
    if value is None:
        return "n/a"
    return "ON " if value else "off"


# Trigger-list index → display label. Matches the wire order documented in
# xr_examples.pico_vr_common.protocol.JointState.
_TRIGGER_LABELS = (
    "L_grab", "R_grab",
    "L_X", "L_Y", "L_stk",
    "R_A", "R_B", "R_stk",
)


def _fmt_buttons(triggers: List[bool]) -> str:
    """Format face-button + thumbstick-click bits (triggers[2:8]). Empty if none set."""
    pressed = [
        _TRIGGER_LABELS[i]
        for i in range(2, min(len(triggers), len(_TRIGGER_LABELS)))
        if triggers[i]
    ]
    return ",".join(pressed) if pressed else "-"


def _fmt_stick(axes: List[float], offset: int) -> str:
    """Format one (x, y) thumbstick reading from axes[offset:offset+2]."""
    if len(axes) < offset + 2:
        return "n/a"
    return f"({axes[offset]:+.2f},{axes[offset + 1]:+.2f})"

from pico_vr.pico_vr_common.protocol import (
    JointState,
    MSG_IMAGE,
    MSG_JOINTS,
    monotonic_timestamp,
    recv_message,
    send_message,
)

logger = logging.getLogger("xr_examples.pico_vr_server.server")

_DEFAULT_IMAGE_PATH = os.path.join(os.path.dirname(__file__), "example.png")

_DEFAULT_JOINT_NAMES: List[str] = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def _load_image_as_jpeg(path: str, size: Optional[Tuple[int, int]],
                        quality: int) -> Tuple[bytes, Tuple[int, int]]:
    """Load ``path``, optionally resize to ``size`` (w, h), and JPEG-encode once."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        if size is not None and size != im.size:
            im = im.resize(size, Image.LANCZOS)
        buf = BytesIO()
        im.save(buf, format="JPEG", quality=quality)
        return buf.getvalue(), im.size


def _make_dummy_joints(t: float, names: List[str]) -> JointState:
    positions = [
        math.sin(2.0 * math.pi * 0.2 * t + 0.3 * i) for i, _ in enumerate(names)
    ]
    velocities = [
        2.0 * math.pi * 0.2 * math.cos(2.0 * math.pi * 0.2 * t + 0.3 * i)
        for i, _ in enumerate(names)
    ]
    return JointState(
        timestamp=t,
        names=list(names),
        positions=positions,
        velocities=velocities,
        efforts=[0.0] * len(names),
    )


class _DownstreamPublisher:
    """One thread per accepted client; pushes images and joints at ``rate`` Hz."""

    def __init__(self, host: str, port: int, rate_hz: float,
                 image_jpeg: Optional[bytes], joint_names: List[str],
                 stop_event: threading.Event):
        self.host = host
        self.port = port
        self.period = 1.0 / max(1.0, rate_hz)
        self._image_lock = threading.Lock()
        self._image_jpeg: Optional[bytes] = image_jpeg
        self.joint_names = joint_names
        self.stop_event = stop_event
        self._listener: Optional[socket.socket] = None
        self._accept_thread: Optional[threading.Thread] = None
        self._client_threads: List[threading.Thread] = []

    def set_image_jpeg(self, image_jpeg: bytes) -> None:
        with self._image_lock:
            self._image_jpeg = image_jpeg

    def get_image_jpeg(self) -> Optional[bytes]:
        with self._image_lock:
            return self._image_jpeg

    def start(self) -> None:
        self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._listener.bind((self.host, self.port))
        self._listener.listen(4)
        self._listener.settimeout(0.5)
        logger.info(f"Downstream listening on {self.host}:{self.port}")
        self._accept_thread = threading.Thread(
            target=self._accept_loop, name="pico-vr-server-accept-downstream", daemon=True,
        )
        self._accept_thread.start()

    def stop(self) -> None:
        if self._listener is not None:
            try:
                self._listener.close()
            except OSError:
                pass
            self._listener = None
        if self._accept_thread is not None:
            self._accept_thread.join(timeout=2.0)
            self._accept_thread = None
        for t in self._client_threads:
            t.join(timeout=2.0)
        self._client_threads.clear()

    def _accept_loop(self) -> None:
        while not self.stop_event.is_set() and self._listener is not None:
            try:
                sock, addr = self._listener.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            logger.info(f"Downstream client connected from {addr}")
            t = threading.Thread(
                target=self._serve_client, args=(sock, addr),
                name=f"pico-vr-server-down-{addr[1]}", daemon=True,
            )
            t.start()
            self._client_threads.append(t)

    def _serve_client(self, sock: socket.socket, addr) -> None:
        try:
            next_tick = time.monotonic()
            while not self.stop_event.is_set():
                t = monotonic_timestamp()
                image_jpeg = self.get_image_jpeg()
                if image_jpeg is not None:
                    send_message(sock, MSG_IMAGE, image_jpeg)
                joints = _make_dummy_joints(t, self.joint_names)
                send_message(sock, MSG_JOINTS, joints.to_json_bytes())
                next_tick += self.period
                sleep_for = next_tick - time.monotonic()
                if sleep_for > 0:
                    time.sleep(sleep_for)
                else:
                    next_tick = time.monotonic()
        except (OSError, BrokenPipeError) as exc:
            logger.info(f"Downstream client {addr} disconnected: {exc!r}")
        finally:
            try:
                sock.close()
            except OSError:
                pass


class _UpstreamReceiver:
    """One thread per accepted client; decodes incoming joint messages."""

    def __init__(self, host: str, port: int,
                 on_joints: Callable[[JointState], None],
                 stop_event: threading.Event):
        self.host = host
        self.port = port
        self.on_joints = on_joints
        self.stop_event = stop_event
        self._listener: Optional[socket.socket] = None
        self._accept_thread: Optional[threading.Thread] = None
        self._client_threads: List[threading.Thread] = []

    def start(self) -> None:
        self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._listener.bind((self.host, self.port))
        self._listener.listen(4)
        self._listener.settimeout(0.5)
        logger.info(f"Upstream listening on {self.host}:{self.port}")
        self._accept_thread = threading.Thread(
            target=self._accept_loop, name="pico-vr-server-accept-upstream", daemon=True,
        )
        self._accept_thread.start()

    def stop(self) -> None:
        if self._listener is not None:
            try:
                self._listener.close()
            except OSError:
                pass
            self._listener = None
        if self._accept_thread is not None:
            self._accept_thread.join(timeout=2.0)
            self._accept_thread = None
        for t in self._client_threads:
            t.join(timeout=2.0)
        self._client_threads.clear()

    def _accept_loop(self) -> None:
        while not self.stop_event.is_set() and self._listener is not None:
            try:
                sock, addr = self._listener.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            logger.info(f"Upstream client connected from {addr}")
            sock.settimeout(1.0)
            t = threading.Thread(
                target=self._serve_client, args=(sock, addr),
                name=f"pico-vr-server-up-{addr[1]}", daemon=True,
            )
            t.start()
            self._client_threads.append(t)

    def _serve_client(self, sock: socket.socket, addr) -> None:
        try:
            while not self.stop_event.is_set():
                msg = recv_message(sock, self.stop_event)
                if msg is None:
                    break
                msg_type, payload = msg
                if msg_type != MSG_JOINTS:
                    logger.debug(f"Ignoring upstream msg type {msg_type:#x} from {addr}")
                    continue
                try:
                    state = JointState.from_json_bytes(payload)
                except (ValueError, KeyError) as exc:
                    logger.warning(f"Bad joint payload from {addr}: {exc!r}")
                    continue
                self.on_joints(state)
        except (OSError, ValueError) as exc:
            logger.info(f"Upstream client {addr} disconnected: {exc!r}")
        finally:
            try:
                sock.close()
            except OSError:
                pass


class DummyServer:
    """Tie the downstream publisher and upstream receiver together."""

    def __init__(self, host: str = "0.0.0.0", port: int = 5555, port2: int = 5556,
                 image_path: Optional[str] = None,
                 image_size: Optional[Tuple[int, int]] = None,
                 rate_hz: float = 30.0,
                 jpeg_quality: int = 80,
                 joint_names: Optional[List[str]] = None,
                 on_joints: Optional[Callable[[JointState], None]] = None,
                 joint_log_interval: float = 0.5,
                 use_default_image: bool = True) -> None:
        self.host = host
        self.port = port
        self.port2 = port2
        self.jpeg_quality = jpeg_quality
        self._image_size = image_size
        self.stop_event = threading.Event()
        self._joint_log_interval = joint_log_interval
        self._last_joint_log = 0.0
        self._joint_log_lock = threading.Lock()
        if on_joints is None:
            on_joints = self._log_joints
        image_jpeg: Optional[bytes] = None
        if image_path is not None or use_default_image:
            resolved_path = image_path or _DEFAULT_IMAGE_PATH
            image_jpeg, actual_size = _load_image_as_jpeg(
                resolved_path, image_size, jpeg_quality,
            )
            logger.info(
                f"Loaded image {resolved_path} ({actual_size[0]}x{actual_size[1]}, "
                f"{len(image_jpeg)} bytes JPEG q={jpeg_quality})"
            )
        self.downstream = _DownstreamPublisher(
            host=host, port=port, rate_hz=rate_hz,
            image_jpeg=image_jpeg,
            joint_names=joint_names or _DEFAULT_JOINT_NAMES,
            stop_event=self.stop_event,
        )
        self.upstream = _UpstreamReceiver(
            host=host, port=port2, on_joints=on_joints, stop_event=self.stop_event,
        )

    def _log_joints(self, state: JointState) -> None:
        # Rate-limit: at 60 Hz upstream we'd otherwise flood the log.
        now = time.monotonic()
        with self._joint_log_lock:
            if now - self._last_joint_log < self._joint_log_interval:
                return
            self._last_joint_log = now
        positions = state.positions
        triggers = state.triggers
        axes = state.axes
        left_grab = triggers[0] if len(triggers) > 0 else None
        right_grab = triggers[1] if len(triggers) > 1 else None
        buttons = _fmt_buttons(triggers)
        left_stick = _fmt_stick(axes, 0)
        right_stick = _fmt_stick(axes, 2)
        if len(positions) == 21:
            head = positions[0:7]
            left = positions[7:14]
            right = positions[14:21]
            logger.info(
                f"xr pose t={state.timestamp:.3f}  "
                f"{_format_pose('H', head)}  |  "
                f"{_format_pose('L', left)} grab={_fmt_trigger(left_grab)} stk={left_stick}  |  "
                f"{_format_pose('R', right)} grab={_fmt_trigger(right_grab)} stk={right_stick}  "
                f"btn={buttons}"
            )
        elif len(positions) == 14:
            left = positions[0:7]
            right = positions[7:14]
            logger.info(
                f"controller pose t={state.timestamp:.3f}  "
                f"{_format_pose('L', left)} grab={_fmt_trigger(left_grab)} stk={left_stick}  |  "
                f"{_format_pose('R', right)} grab={_fmt_trigger(right_grab)} stk={right_stick}  "
                f"btn={buttons}"
            )
        else:
            preview = ", ".join(f"{p:+.3f}" for p in positions[:6])
            logger.info(
                f"upstream joints t={state.timestamp:.3f} n={len(positions)} "
                f"positions=[{preview}] triggers={triggers} axes={axes}"
            )

    def set_image(self, image, color_order: str = "RGB") -> None:
        """Update the streamed frame from an HxWx3 uint8 ndarray.

        ``color_order`` is "RGB" (default) or "BGR". The frame is JPEG-encoded
        once here and stored; all connected clients see it on the next tick.
        """
        arr = image
        if color_order.upper() == "BGR":
            arr = arr[:, :, ::-1]
        pil_im = Image.fromarray(arr, mode="RGB")
        if self._image_size is not None and self._image_size != pil_im.size:
            pil_im = pil_im.resize(self._image_size, Image.LANCZOS)
        buf = BytesIO()
        pil_im.save(buf, format="JPEG", quality=self.jpeg_quality)
        self.downstream.set_image_jpeg(buf.getvalue())

    def start(self) -> None:
        self.stop_event.clear()
        self.downstream.start()
        self.upstream.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.upstream.stop()
        self.downstream.stop()

    def wait(self) -> None:
        """Block until :meth:`stop` is called (or the process is signalled)."""
        try:
            while not self.stop_event.is_set():
                self.stop_event.wait(0.5)
        except KeyboardInterrupt:
            pass
