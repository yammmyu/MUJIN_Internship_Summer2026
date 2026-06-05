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
import tempfile
import threading
import time

import numpy as np
import pybullet as p

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from real_world.ik import (  # noqa: E402
    PinocchioArmModel, PinocchioDLSIKSolver, IKQuery,
    quat_xyzw_to_se3, LEFT_ARM_JOINTS, DEFAULT_URDF, DEFAULT_CALIBRATION,
    load_calibration,
)

HUMANOID = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_RECORDINGS = os.path.join(HUMANOID, "MDM_data_collection", "recordings")
LEFT_GRIPPER_JOINTS = ([f"left_narrow{i}_joint" for i in (1, 2, 3, 4)] +
                       [f"left_wide{i}_joint" for i in (1, 2, 3, 4)])
GRIPPER_CLOSE_RAD = 0.6   # rough scalar grip[0,1] -> joint angle (uncalibrated)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8753


def patched_urdf(urdf_path):
    """PyBullet can't resolve package://; rewrite to relative and load from the URDF dir."""
    with open(urdf_path) as f:
        text = f.read().replace("package://", "")
    d = os.path.dirname(urdf_path)
    fd, tmp = tempfile.mkstemp(suffix=".urdf", dir=d)
    with os.fdopen(fd, "w") as f:
        f.write(text)
    return tmp, d


def load_trajectory(recordings_dir, recording, max_frames):
    """Read a recording's left-EE poses, grip, and live arm joints from robot_states.npz."""
    path = os.path.join(recordings_dir, recording, "robot_states.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"no recording at {path}")
    npz = np.load(path)
    ee_pos = npz["left_pos"].astype(np.float64)
    ee_quat = npz["left_quat"].astype(np.float64)
    grip = npz["gripper"].astype(np.float64)[:, 0]
    arm_joints = npz["arm_joints"].astype(np.float64)   # (T, 14)
    n = len(ee_pos) if not max_frames else min(len(ee_pos), max_frames)
    return ee_pos, ee_quat, grip, arm_joints, n


# --------------------------------------------------------------------------- #
#  The persistent PyBullet sim environment                                    #
# --------------------------------------------------------------------------- #
class SimEnv:
    """Holds the PyBullet world + IK model/solver. Launched once; replays many recordings.

    All PyBullet calls happen on the thread that built this object (the main thread).
    """

    def __init__(self, args):
        self.direct = args.direct
        self.anchor_first = args.anchor_first
        self.fixed_rate = args.fixed_rate

        # --- IK model + solver (calibrated, constant base_offset) ---
        self.cal = load_calibration(args.calibration)
        ee_frame = args.ee_frame or self.cal.ee_frame
        self.model = PinocchioArmModel(urdf_path=args.urdf, ee_frame=ee_frame)
        self.left_slice = slice(0, 7) if self.cal.arm_joint_slice == "left_first" else slice(7, 14)
        if not self.anchor_first:
            self.model.base_offset = self.cal.base_offset_se3()
        self.solver = PinocchioDLSIKSolver(self.model, damping=1e-2, max_iters=100, tol=1e-4)

        # --- PyBullet ---
        p.connect(p.DIRECT if self.direct else p.GUI)
        tmp_urdf, search_dir = patched_urdf(args.urdf)
        p.setAdditionalSearchPath(search_dir)
        p.setGravity(0, 0, 0)                          # kinematic view: arm holds commanded pose
        try:
            self.body = p.loadURDF(tmp_urdf, useFixedBase=True)
        finally:
            os.remove(tmp_urdf)
        jmap = {p.getJointInfo(self.body, i)[1].decode(): i
                for i in range(p.getNumJoints(self.body))}
        self.arm_idx = [jmap[j] for j in LEFT_ARM_JOINTS]
        self.grip_idx = [jmap[j] for j in LEFT_GRIPPER_JOINTS if j in jmap]

        # --- stepping / settling ---
        self.sim_dt = 1.0 / args.sim_hz
        p.setTimeStep(self.sim_dt)
        self.settle_tol = np.radians(args.settle_tol_deg)
        self.settle_tol_deg = args.settle_tol_deg
        self.settle_max_steps = max(1, int(args.settle_timeout_s * args.sim_hz))
        self.fixed_steps = max(1, int(args.sim_hz / args.playback_hz))

    def connected(self):
        return p.isConnected()

    def idle_step(self):
        """Keep the GUI responsive while waiting for the next recording."""
        if self.direct:
            time.sleep(0.02)
        else:
            p.stepSimulation()
            time.sleep(self.sim_dt)

    def _cur_arm_q(self):
        return np.array([p.getJointState(self.body, k)[0] for k in self.arm_idx])

    def replay(self, recording, recordings_dir, max_frames, emit):
        """Replay one recording. `emit(str)` receives each log line (stdout + client)."""
        ee_pos, ee_quat, grip, arm_joints, n = load_trajectory(
            recordings_dir, recording, max_frames)
        emit(f"Recording {recording}: {n} EE poses")

        # Seed from the recording's first live left-arm joints, clamped to URDF limits.
        q_seed = np.clip(arm_joints[0, self.left_slice], self.model.lower, self.model.upper)
        if self.anchor_first:
            # Re-anchor base_offset so the EE starts exactly at the seed's FK.
            E0 = self.model.fk_local(q_seed)
            S0 = quat_xyzw_to_se3(ee_pos[0], ee_quat[0])
            self.model.base_offset = S0 * E0.inverse()
            emit("base_offset: anchored to this recording's first EE pose (--anchor-first)")
        else:
            self.model.base_offset = self.cal.base_offset_se3()
            emit(f"base_offset: calibrated; residual "
                 f"{self.cal.pos_residual_m * 1000:.1f} mm / {self.cal.rot_residual_deg:.2f} deg")

        # Reset the arm to the seed so each recording starts clean.
        for k, qi in zip(self.arm_idx, q_seed):
            p.resetJointState(self.body, k, float(qi))

        idx = np.linspace(0, n - 1, n).astype(int)
        q = q_seed.copy()
        ik_err, steps, n_timeout, n_skipped = [], [], 0, 0
        mode = "fixed-rate" if self.fixed_rate else f"settle(tol={self.settle_tol_deg}deg)"
        emit(f"mode: {mode}")
        emit(f"{'frame':>6} | {'IK pos err(mm)':>14} {'|dq| step(deg)':>14} "
             f"{'settle steps':>12} {'near-lim':>8}")

        for i in idx:
            # warm-start: last config is current
            q_des = self.solver.solve(IKQuery(target_pos=ee_pos[i], target_quat=ee_quat[i],
                                              current_joints=q))

            # Unreachable target: skip it entirely. Don't command the arm and DON'T advance
            # the seed — the next target is planned from the last reachable config, so a skip
            # can't poison the warm-start.
            if not self.solver.last_reachable:
                n_skipped += 1
                emit(f"{i:>6} | {self.solver.last_pos_err * 1000:>14.2f} "
                     f"{'SKIP (unreachable)':>14} {'-':>12} {'-':>8}")
                continue

            p.setJointMotorControlArray(self.body, self.arm_idx, p.POSITION_CONTROL,
                                        targetPositions=q_des.tolist(),
                                        forces=[200.0] * len(self.arm_idx))
            if self.grip_idx:
                g = float(np.clip(grip[i], 0, 1)) * GRIPPER_CLOSE_RAD
                p.setJointMotorControlArray(self.body, self.grip_idx, p.POSITION_CONTROL,
                                            targetPositions=[g] * len(self.grip_idx),
                                            forces=[20.0] * len(self.grip_idx))

            # Default: step until the arm settles at q_des (or times out). --fixed-rate: just
            # give a fixed budget per pose.
            n_steps = 0
            timed_out = False
            while True:
                p.stepSimulation()
                if not self.direct:
                    time.sleep(self.sim_dt)
                n_steps += 1
                if self.fixed_rate:
                    if n_steps >= self.fixed_steps:
                        break
                else:
                    if np.max(np.abs(self._cur_arm_q() - q_des)) < self.settle_tol:
                        break
                    if n_steps >= self.settle_max_steps:
                        timed_out = True
                        break

            res = np.linalg.norm(self.model.fk(q_des)[0] - ee_pos[i])    # IK's own reach error
            step = np.degrees(np.max(np.abs(q_des - q)))
            near = bool(np.any((q_des - self.model.lower < 0.05) |
                               (self.model.upper - q_des < 0.05)))
            ik_err.append(res)
            steps.append(step)
            if timed_out:
                n_timeout += 1
            emit(f"{i:>6} | {res * 1000:>14.2f} {step:>14.2f} {n_steps:>12}"
                 f"{' TO' if timed_out else '':>0} {str(near):>8}")
            q = q_des                       # the last solution becomes the current seed

        ik_err, steps = np.array(ik_err), np.array(steps)
        if len(ik_err):
            emit(f"\nSummary: IK pos-err median {np.median(ik_err) * 1000:.1f} mm "
                 f"(max {ik_err.max() * 1000:.1f}) | step median {np.median(steps):.1f} deg "
                 f"(max {steps.max():.1f}) | reached {len(ik_err)}/{len(idx)} | "
                 f"skipped(unreachable) {n_skipped}/{len(idx)} | "
                 f"settle-timeouts {n_timeout}/{len(idx)}")
        else:
            emit(f"\nSummary: every target was unreachable — skipped {n_skipped}/{len(idx)} "
                 f"(check base_offset anchoring / workspace).")


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
    env = SimEnv(args)
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
        if env.connected():
            p.disconnect()


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
    env = SimEnv(args)
    try:
        env.replay(args.recording, args.recordings, args.max_frames, print)
    finally:
        if not args.direct and env.connected():
            print("\nClose the PyBullet window to exit.")
            while env.connected():
                time.sleep(0.1)
        if env.connected():
            p.disconnect()


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
