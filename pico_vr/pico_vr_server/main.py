"""
Run the dummy Pico VR server. Streams synthetic RGB frames + joint values to any
connected client and accepts joint updates back.

Defaults can be set via ``LOCAL_HOST`` / ``REMOTE_PORT`` / ``REMOTE_PORT2``
environment variables and overridden on the command line.
"""

import argparse
import logging
import os
import signal
import sys

from .server import DummyServer

logger = logging.getLogger("xr_examples.pico_vr_server.main")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host",
        default=os.environ.get("LOCAL_HOST", "0.0.0.0"),
        help="Bind address (default: $LOCAL_HOST or 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("REMOTE_PORT", "5555")),
        help="Downstream port; image+joint stream (default: $REMOTE_PORT or 5555)",
    )
    parser.add_argument(
        "--port2",
        type=int,
        default=int(os.environ.get("REMOTE_PORT2", "5556")),
        help="Upstream port; joint updates (default: $REMOTE_PORT2 or 5556)",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Path to the source image. Defaults to example.png next to this module.",
    )
    parser.add_argument("--width", type=int, default=None,
                        help="Resize width (default: native image width)")
    parser.add_argument("--height", type=int, default=None,
                        help="Resize height (default: native image height)")
    parser.add_argument("--rate", type=float, default=30.0, help="Frame rate (Hz)")
    parser.add_argument("--jpeg-quality", type=int, default=80)
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        format="%(asctime)s.%(msecs)03d %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
        level=logging.DEBUG if args.verbose else logging.INFO,
    )
    size = None
    if args.width is not None and args.height is not None:
        size = (args.width, args.height)
    server = DummyServer(
        host=args.host,
        port=args.port,
        port2=args.port2,
        image_path=args.image,
        image_size=size,
        rate_hz=args.rate,
        jpeg_quality=args.jpeg_quality,
    )

    def _handle_signal(_signum, _frame):
        logger.info("Signal received, stopping server")
        server.stop_event.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    server.start()
    try:
        server.wait()
    finally:
        server.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
