"""
TCP transport for the Pico VR client example.

Two independent connections to the server:

* **Downstream** (``host:port``): receives a multiplexed stream of JPEG image
  frames and joint-state messages. The most recent decoded image is stored
  for the renderer; the most recent joint state is written into a shared
  :class:`JointStateBuffer`.
* **Upstream** (``host:port2``): a separate thread polls a shared
  :class:`JointStateBuffer` and pushes every new revision back to the server.

Each thread reconnects with a fixed delay on failure.
"""

import logging
import socket
import threading
from io import BytesIO
from typing import Optional, Tuple

import numpy as np
from PIL import Image

from pico_vr.pico_vr_common.protocol import (
    JointState,
    JointStateBuffer,
    MSG_IMAGE,
    MSG_JOINTS,
    recv_message,
    send_message,
)

logger = logging.getLogger("xr_examples.pico_vr_client.tcp_stream")


class DownstreamReceiver:
    """Receives images and joint states from the server on a single TCP connection."""

    def __init__(self, host: str, port: int, incoming_joints: JointStateBuffer,
                 reconnect_delay: float = 1.0):
        self.host = host
        self.port = port
        self.incoming_joints = incoming_joints
        self.reconnect_delay = reconnect_delay
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._frame_id = 0
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="pico-vr-downstream", daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def latest_frame(self) -> Tuple[Optional[np.ndarray], int]:
        with self._lock:
            return self._frame, self._frame_id

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                with socket.create_connection((self.host, self.port), timeout=5.0) as sock:
                    sock.settimeout(5.0)
                    logger.info(f"Downstream connected to {self.host}:{self.port}")
                    self._receive_loop(sock)
            except (OSError, socket.timeout, ValueError) as exc:
                logger.warning(f"Downstream error: {exc!r}; retry in {self.reconnect_delay}s")
                self._stop_event.wait(self.reconnect_delay)

    def _receive_loop(self, sock: socket.socket) -> None:
        while not self._stop_event.is_set():
            msg = recv_message(sock, self._stop_event)
            if msg is None:
                logger.info("Downstream closed by remote")
                return
            msg_type, payload = msg
            if msg_type == MSG_IMAGE:
                frame = _decode_jpeg(payload)
                if frame is not None:
                    with self._lock:
                        self._frame = frame
                        self._frame_id += 1
                        if self._frame_id == 1:
                            logger.info(
                                f"First image frame received: "
                                f"{frame.shape[1]}x{frame.shape[0]} ({len(payload)} bytes)"
                            )
            elif msg_type == MSG_JOINTS:
                try:
                    state = JointState.from_json_bytes(payload)
                except (ValueError, KeyError) as exc:
                    logger.warning(f"Bad joint message: {exc!r}")
                    continue
                self.incoming_joints.set(state)
            else:
                logger.debug(f"Ignoring unknown message type {msg_type:#x}")


class UpstreamSender:
    """Polls a :class:`JointStateBuffer` and sends any new joint state to the server."""

    def __init__(self, host: str, port: int, outgoing_joints: JointStateBuffer,
                 poll_interval: float = 1.0 / 60.0, reconnect_delay: float = 1.0):
        self.host = host
        self.port = port
        self.outgoing_joints = outgoing_joints
        self.poll_interval = poll_interval
        self.reconnect_delay = reconnect_delay
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_sent_version = 0

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._last_sent_version = 0
        self._thread = threading.Thread(
            target=self._run, name="pico-vr-upstream", daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                with socket.create_connection((self.host, self.port), timeout=5.0) as sock:
                    sock.settimeout(5.0)
                    logger.info(f"Upstream connected to {self.host}:{self.port}")
                    self._send_loop(sock)
            except (OSError, socket.timeout) as exc:
                logger.warning(f"Upstream error: {exc!r}; retry in {self.reconnect_delay}s")
                self._stop_event.wait(self.reconnect_delay)

    def _send_loop(self, sock: socket.socket) -> None:
        while not self._stop_event.is_set():
            state, version = self.outgoing_joints.get()
            if state is not None and version != self._last_sent_version:
                send_message(sock, MSG_JOINTS, state.to_json_bytes())
                self._last_sent_version = version
            self._stop_event.wait(self.poll_interval)


def _decode_jpeg(payload: bytes) -> Optional[np.ndarray]:
    try:
        with Image.open(BytesIO(payload)) as img:
            return np.asarray(img.convert("RGB"), dtype=np.uint8)
    except Exception as exc:
        logger.warning(f"Failed to decode JPEG ({len(payload)} bytes): {exc!r}")
        return None
