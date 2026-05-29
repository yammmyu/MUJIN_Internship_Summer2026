"""TCP server for the GUI motion-planning client.

One listening socket, one active client at a time (the GUI). Frame format:
4 bytes big-endian length + UTF-8 JSON payload, matching the client side in
`gui/motion_planning.py`.

This module is intentionally protocol-only: it dispatches inbound JSON to
callbacks on a controller object and exposes ``send_json`` for outbound.
The PlanningServer wires it up.
"""
from __future__ import annotations

import json
import logging
import socket
import struct
import threading
from typing import Any, Callable, Optional

log = logging.getLogger(__name__)

_MAX_FRAME_BYTES = 50 * 1024 * 1024  # 50 MB safety cap (matches client)


class TcpServer:
    """Single-client length-prefixed JSON server.

    The controller is any object exposing optional callbacks:
        on_state(msg), on_detection_image(msg),
        on_trajectory_ack(msg), on_trajectory_result(msg),
        on_unknown(msg), on_connect(addr), on_disconnect(addr)
    Missing callbacks are simply skipped.
    """

    def __init__(self, controller: Any, host: str = "0.0.0.0", port: int = 9100):
        self._controller = controller
        self._host = host
        self._port = port

        self._listener: Optional[socket.socket] = None
        self._client_sock: Optional[socket.socket] = None
        self._client_addr = None

        self._send_lock = threading.Lock()
        self._stop = threading.Event()
        self._accept_thread: Optional[threading.Thread] = None
        self._reader_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def start(self) -> None:
        self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._listener.bind((self._host, self._port))
        self._listener.listen(1)
        self._listener.settimeout(1.0)
        log.warning("[TCP] listening on %s:%d", self._host, self._port)
        self._accept_thread = threading.Thread(
            target=self._accept_loop, daemon=True, name="planning-tcp-accept")
        self._accept_thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._close_client()
        try:
            if self._listener is not None:
                self._listener.close()
        except Exception:
            pass
        self._listener = None

    # ------------------------------------------------------------------ #
    # Outbound
    # ------------------------------------------------------------------ #
    def send_json(self, obj: dict) -> bool:
        sock = self._client_sock
        if sock is None:
            return False
        payload = json.dumps(obj).encode("utf-8")
        try:
            with self._send_lock:
                sock.sendall(struct.pack(">I", len(payload)) + payload)
            return True
        except OSError as e:
            log.warning("[TCP] send error: %s", e)
            self._close_client()
            return False

    @property
    def connected(self) -> bool:
        return self._client_sock is not None

    # ------------------------------------------------------------------ #
    # Accept / read loops
    # ------------------------------------------------------------------ #
    def _accept_loop(self) -> None:
        while not self._stop.is_set():
            try:
                client, addr = self._listener.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            # 替换已有连接（典型场景：GUI 重启）
            if self._client_sock is not None:
                log.warning("[TCP] replacing existing client connection")
                self._close_client()
            client.settimeout(None)
            self._client_sock = client
            self._client_addr = addr
            log.warning("[TCP] client connected from %s", addr)
            self._notify("on_connect", addr)
            self._reader_thread = threading.Thread(
                target=self._reader_loop, args=(client, addr),
                daemon=True, name="planning-tcp-reader")
            self._reader_thread.start()
        log.warning("[TCP] accept loop stopped")

    def _reader_loop(self, sock: socket.socket, addr) -> None:
        try:
            while not self._stop.is_set() and self._client_sock is sock:
                payload = self._recv_frame(sock)
                if payload is None:
                    log.warning("[TCP] client %s disconnected", addr)
                    return
                try:
                    obj = json.loads(payload.decode("utf-8"))
                except Exception as e:
                    log.warning("[TCP] bad JSON from %s: %s", addr, e)
                    continue
                self._dispatch(obj)
        finally:
            if self._client_sock is sock:
                self._close_client()
            self._notify("on_disconnect", addr)

    def _dispatch(self, obj: dict) -> None:
        mtype = obj.get("type")
        # Map type -> controller method name
        handler_name = {
            "state": "on_state",
            "detection_image": "on_detection_image",
            "trajectory_ack": "on_trajectory_ack",
            "trajectory_result": "on_trajectory_result",
        }.get(mtype, "on_unknown")
        self._notify(handler_name, obj)

    def _notify(self, name: str, *args) -> None:
        fn: Optional[Callable] = getattr(self._controller, name, None)
        if fn is None:
            return
        try:
            fn(*args)
        except Exception:
            log.exception("[TCP] controller.%s raised", name)

    # ------------------------------------------------------------------ #
    # Framing
    # ------------------------------------------------------------------ #
    def _recv_frame(self, sock: socket.socket) -> Optional[bytes]:
        head = self._recv_exact(sock, 4)
        if head is None:
            return None
        (length,) = struct.unpack(">I", head)
        if length == 0 or length > _MAX_FRAME_BYTES:
            log.warning("[TCP] bogus frame length %d, dropping", length)
            return None
        return self._recv_exact(sock, length)

    def _recv_exact(self, sock: socket.socket, n: int) -> Optional[bytes]:
        buf = bytearray()
        while len(buf) < n:
            try:
                chunk = sock.recv(n - len(buf))
            except OSError:
                return None
            if not chunk:
                return None
            buf.extend(chunk)
        return bytes(buf)

    def _close_client(self) -> None:
        sock = self._client_sock
        self._client_sock = None
        self._client_addr = None
        if sock is not None:
            try:
                sock.close()
            except Exception:
                pass
