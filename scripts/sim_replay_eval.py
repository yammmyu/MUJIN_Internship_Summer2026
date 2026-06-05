"""Watch our IK drive the G1 left arm in PyBullet from recorded EE trajectories.

Simple, local, CPU-only (no robot, no GPU, no policy server). Purpose: see how our
Pinocchio IK handles joints + motion when fed a real EE-pose trajectory — before any
hardware command.

The PyBullet environment is launched ONCE (`serve`) and stays up; you stream recordings
into it from another terminal (`send`). One-shot `replay` is kept for quick checks.

Flow (per recording):
  1. Read its left-EE poses (left_pos, left_quat) from robot_states.npz.
  2. Place the model's base_offset using the calibrated firmware->base_link offset
     (fk_calibration.json, produced by fk_consistency_check.py — PASS at ~5 mm / 0.8 deg
     residual on live-joint recordings), so the recorded SDK EE poses live in the same
     frame as the URDF FK. Seed the arm from the recording's first live arm_joints.
     (--anchor-first re-anchors base_offset to that recording's first EE pose instead,
     for recordings whose absolute frame can't be trusted.)
  3. For each recorded EE pose: solve IK (warm-started from the previous solution — "the
     last config becomes the current"), command the PyBullet arm, step, repeat.

Usage:
    # Terminal 1 — launch the sim once (GUI stays open, waits for recordings):
    .venv/bin/python scripts/sim_replay_eval.py serve \
        --recordings /home/chenyanyu/Downloads/recordings

    # Terminal 2 — send recordings into the running sim:
    .venv/bin/python scripts/sim_replay_eval.py send recording001
    .venv/bin/python scripts/sim_replay_eval.py send recording018 --max-frames 200

    # One-shot (launch, replay once, exit) — old behavior:
    .venv/bin/python scripts/sim_replay_eval.py replay --recording recording001 --direct
"""

import argparse
import json
import os
import queue
import socket
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from real_world.ik import DEFAULT_URDF, DEFAULT_CALIBRATION  # noqa: E402
from real_world.sim_backend import SimEnv  # noqa: E402

HUMANOID = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_RECORDINGS = os.path.join(HUMANOID, "MDM_data_collection", "recordings")

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8753


def _sim_from_args(args):
    """Build a SimEnv from the shared CLI args (serve/replay)."""
    return SimEnv(
        urdf=args.urdf, ee_frame=args.ee_frame, calibration=args.calibration,
        sim_hz=args.sim_hz, direct=args.direct, anchor_first=args.anchor_first,
        fixed_rate=args.fixed_rate, settle_tol_deg=args.settle_tol_deg,
        settle_timeout_s=args.settle_timeout_s, playback_hz=args.playback_hz,
    )


# --------------------------------------------------------------------------- #
#  serve: launch the env once, accept recording requests over a socket         #
# --------------------------------------------------------------------------- #
def _accept_loop(srv, jobs):
    """Daemon thread: accept connections, read one JSON request line, enqueue (req, conn).

    All PyBullet work stays on the main thread; this thread only does socket I/O.
    """
    while True:
        try:
            conn, _ = srv.accept()
        except OSError:
            return
        try:
            line = conn.makefile().readline()
            req = json.loads(line) if line.strip() else {}
            jobs.put((req, conn))
        except (OSError, json.JSONDecodeError) as e:
            try:
                conn.sendall(f"ERROR: bad request: {e}\n".encode())
                conn.close()
            except OSError:
                pass


def serve(args):
    env = _sim_from_args(args)
    jobs = queue.Queue()

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.host, args.port))
    srv.listen(8)
    threading.Thread(target=_accept_loop, args=(srv, jobs), daemon=True).start()

    print(f"Sim ready. Listening on {args.host}:{args.port} "
          f"(recordings dir: {args.recordings})")
    print("Send recordings from another terminal:")
    print(f"  .venv/bin/python scripts/sim_replay_eval.py send recording001 "
          f"--port {args.port}")
    print("Close the PyBullet window (or Ctrl-C) to exit.\n")

    try:
        while env.connected():
            try:
                req, conn = jobs.get_nowait()
            except queue.Empty:
                env.idle_step()
                continue

            recording = req.get("recording")
            max_frames = int(req.get("max_frames", 0) or 0)

            def emit(line, _conn=conn):
                print(line)
                try:
                    _conn.sendall((line + "\n").encode())
                except OSError:
                    pass

            if not recording:
                emit("ERROR: request missing 'recording'")
            else:
                print(f"\n=== replaying {recording} ===")
                try:
                    env.replay(recording, args.recordings, max_frames, emit)
                except Exception as e:    # don't let one bad request kill the server
                    emit(f"ERROR: {type(e).__name__}: {e}")
            try:
                conn.close()
            except OSError:
                pass
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        srv.close()
        env.disconnect()


# --------------------------------------------------------------------------- #
#  send: client — push one recording into a running server, stream its log      #
# --------------------------------------------------------------------------- #
def send(args):
    req = {"recording": args.recording, "max_frames": args.max_frames}
    try:
        s = socket.create_connection((args.host, args.port), timeout=5)
    except OSError as e:
        print(f"Could not reach sim server at {args.host}:{args.port} ({e}).\n"
              f"Is it running? Start it with:  "
              f".venv/bin/python scripts/sim_replay_eval.py serve")
        sys.exit(1)
    with s:
        s.sendall((json.dumps(req) + "\n").encode())
        s.shutdown(socket.SHUT_WR)
        for line in s.makefile():       # stream the server's per-frame log back here
            print(line, end="")


# --------------------------------------------------------------------------- #
#  replay: one-shot (launch, replay once, exit)                                 #
# --------------------------------------------------------------------------- #
def replay_once(args):
    env = _sim_from_args(args)
    try:
        env.replay(args.recording, args.recordings, args.max_frames, print)
    finally:
        if not args.direct and env.connected():
            print("\nClose the PyBullet window to exit.")
            while env.connected():
                time.sleep(0.1)
        env.disconnect()


# --------------------------------------------------------------------------- #
#  CLI                                                                         #
# --------------------------------------------------------------------------- #
def _add_sim_args(ap):
    """Args shared by `serve` and `replay` (the env-building side)."""
    ap.add_argument("--recordings", default=DEFAULT_RECORDINGS)
    ap.add_argument("--urdf", default=DEFAULT_URDF)
    ap.add_argument("--calibration", default=DEFAULT_CALIBRATION,
                    help="fk_calibration.json (firmware->base_link offset + ee_frame)")
    ap.add_argument("--ee-frame", default=None, help="override the calibration's ee_frame")
    ap.add_argument("--anchor-first", action="store_true",
                    help="ignore the calibrated offset; re-anchor base_offset to each "
                         "recording's first EE pose (for untrusted absolute frames)")
    ap.add_argument("--sim-hz", type=float, default=240.0)
    ap.add_argument("--direct", action="store_true", help="headless (no GUI), for testing")
    ap.add_argument("--settle-tol-deg", type=float, default=1.0,
                    help="a pose is reached when every joint is within this of the command")
    ap.add_argument("--settle-timeout-s", type=float, default=1.5,
                    help="give up settling after this long (flags the pose as unreached)")
    ap.add_argument("--fixed-rate", action="store_true",
                    help="fixed time budget per pose instead of settling")
    ap.add_argument("--playback-hz", type=float, default=20.0,
                    help="poses/sec in --fixed-rate mode")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("serve", help="launch the PyBullet env once and wait for recordings")
    _add_sim_args(ps)
    ps.add_argument("--host", default=DEFAULT_HOST)
    ps.add_argument("--port", type=int, default=DEFAULT_PORT)
    ps.set_defaults(func=serve)

    pc = sub.add_parser("send", help="push a recording into a running sim server")
    pc.add_argument("recording", help="recording name, e.g. recording001")
    pc.add_argument("--max-frames", type=int, default=0, help="0 = all")
    pc.add_argument("--host", default=DEFAULT_HOST)
    pc.add_argument("--port", type=int, default=DEFAULT_PORT)
    pc.set_defaults(func=send)

    pr = sub.add_parser("replay", help="one-shot: launch, replay one recording, exit")
    pr.add_argument("--recording", default="recording001")
    pr.add_argument("--max-frames", type=int, default=0, help="0 = all")
    _add_sim_args(pr)
    pr.set_defaults(func=replay_once)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
