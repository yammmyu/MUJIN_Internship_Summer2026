"""
Wire protocol shared between :mod:`xr_examples.pico_vr_client` and
:mod:`xr_examples.pico_vr_server`.

Each message on the wire is::

    [uint32 big-endian total_len] [uint8 msg_type] [total_len-1 bytes payload]

``msg_type`` is one of :data:`MSG_IMAGE` (payload is JPEG-encoded RGB) or
:data:`MSG_JOINTS` (payload is UTF-8 JSON encoding a :class:`JointState`).
The same framing is used in both directions; the downstream channel
(server → client) carries both message types, while the upstream channel
(client → server) carries only joint messages.
"""

import json
import socket
import struct
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

MSG_IMAGE = 0x01
MSG_JOINTS = 0x02

_HEADER = struct.Struct("!I")
_MAX_PAYLOAD = 64 * 1024 * 1024  # 64 MiB safety cap


@dataclass
class JointState:
    """Snapshot of a robot's joint values, modelled loosely on ROS sensor_msgs/JointState.

    The optional ``triggers`` and ``axes`` lists carry VR controller input that
    is not part of the ROS message but is convenient to ship in the same
    envelope. Both are omitted from the JSON when empty so older receivers
    parse the rest unchanged.

    Wire order used by the Pico VR client when sourcing from the XR controllers:

    ``triggers`` (booleans)::

        [0] left  trigger (grab)
        [1] right trigger (grab)
        [2] left  X button
        [3] left  Y button
        [4] left  thumbstick click
        [5] right A button
        [6] right B button
        [7] right thumbstick click

    ``axes`` (floats in [-1, 1])::

        [0] left  thumbstick x
        [1] left  thumbstick y
        [2] right thumbstick x
        [3] right thumbstick y
    """

    timestamp: float = 0.0
    names: List[str] = field(default_factory=list)
    positions: List[float] = field(default_factory=list)
    velocities: List[float] = field(default_factory=list)
    efforts: List[float] = field(default_factory=list)
    triggers: List[bool] = field(default_factory=list)
    axes: List[float] = field(default_factory=list)

    def to_json_bytes(self) -> bytes:
        payload = {
            "timestamp": self.timestamp,
            "names": self.names,
            "positions": list(self.positions),
            "velocities": list(self.velocities),
            "efforts": list(self.efforts),
        }
        if self.triggers:
            payload["triggers"] = [bool(v) for v in self.triggers]
        if self.axes:
            payload["axes"] = [float(v) for v in self.axes]
        return json.dumps(payload).encode("utf-8")

    @classmethod
    def from_json_bytes(cls, payload: bytes) -> "JointState":
        d = json.loads(payload.decode("utf-8"))
        return cls(
            timestamp=float(d.get("timestamp", 0.0)),
            names=list(d.get("names", [])),
            positions=[float(v) for v in d.get("positions", [])],
            velocities=[float(v) for v in d.get("velocities", [])],
            efforts=[float(v) for v in d.get("efforts", [])],
            triggers=[bool(v) for v in d.get("triggers", [])],
            axes=[float(v) for v in d.get("axes", [])],
        )


class JointStateBuffer:
    """Thread-safe latest-value container for :class:`JointState`."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state: Optional[JointState] = None
        self._version = 0

    def set(self, state: JointState) -> None:
        with self._lock:
            self._state = state
            self._version += 1

    def get(self) -> Tuple[Optional[JointState], int]:
        with self._lock:
            return self._state, self._version


def send_message(sock: socket.socket, msg_type: int, payload: bytes) -> None:
    """Send a single framed message. Thread-unsafe; callers must serialise sends per socket."""
    total_len = len(payload) + 1
    if total_len - 1 > _MAX_PAYLOAD:
        raise ValueError(f"Payload too large: {total_len - 1} bytes")
    header = _HEADER.pack(total_len) + bytes([msg_type & 0xFF])
    sock.sendall(header + payload)


def recv_message(sock: socket.socket, stop_event: Optional[threading.Event] = None
                 ) -> Optional[Tuple[int, bytes]]:
    """Receive one framed message. Returns ``(msg_type, payload)`` or ``None`` on EOF/stop."""
    header = _recv_exact(sock, 4, stop_event)
    if header is None:
        return None
    (total_len,) = _HEADER.unpack(header)
    if total_len < 1 or total_len - 1 > _MAX_PAYLOAD:
        raise ValueError(f"Invalid message length {total_len}")
    body = _recv_exact(sock, total_len, stop_event)
    if body is None:
        return None
    return body[0], body[1:]


def _recv_exact(sock: socket.socket, n: int,
                stop_event: Optional[threading.Event]) -> Optional[bytes]:
    chunks: List[bytes] = []
    remaining = n
    while remaining > 0:
        if stop_event is not None and stop_event.is_set():
            return None
        try:
            chunk = sock.recv(remaining)
        except socket.timeout:
            continue
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def monotonic_timestamp() -> float:
    return time.monotonic()
