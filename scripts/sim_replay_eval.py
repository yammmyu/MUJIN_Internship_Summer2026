"""Watch our IK drive the G1 left arm in PyBullet from a recorded EE trajectory.

Simple, local, CPU-only (no robot, no GPU, no policy server). Purpose: see how our
Pinocchio IK handles joints + motion when fed a real EE-pose trajectory — before any
hardware command.

Flow:
  1. Pick a recording; read its left-EE poses (left_pos, left_quat) from robot_states.npz.
  2. Anchor the model's base_offset to the FIRST recorded EE pose, so the arm starts at a
     chosen seed config and the targets stay in the workspace. (FK calibration vs firmware
     is skipped for now — the recordings have frozen arm_joints; see fk_consistency_check.py.
     This anchoring makes the motion/joint behavior meaningful, not the absolute placement.)
  3. For each recorded EE pose: solve IK (warm-started from the previous solution — "the last
     config becomes the current"), command the PyBullet arm, step, repeat.

Usage:
    .venv/bin/python scripts/sim_replay_eval.py --recording recording006
    .venv/bin/python scripts/sim_replay_eval.py --recording recording006 --direct   # headless
"""

import argparse
import os
import sys
import tempfile
import time

import numpy as np
import pybullet as p

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from real_world.ik import (  # noqa: E402
    PinocchioArmModel, PinocchioDLSIKSolver, IKQuery,
    quat_xyzw_to_se3, LEFT_ARM_JOINTS, DEFAULT_URDF,
)

HUMANOID = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_RECORDINGS = os.path.join(HUMANOID, "MDM_data_collection", "recordings")
LEFT_GRIPPER_JOINTS = ([f"left_narrow{i}_joint" for i in (1, 2, 3, 4)] +
                       [f"left_wide{i}_joint" for i in (1, 2, 3, 4)])
GRIPPER_CLOSE_RAD = 0.6   # rough scalar grip[0,1] -> joint angle (uncalibrated)


def patched_urdf(urdf_path):
    """PyBullet can't resolve package://; rewrite to relative and load from the URDF dir."""
    with open(urdf_path) as f:
        text = f.read().replace("package://", "")
    d = os.path.dirname(urdf_path)
    fd, tmp = tempfile.mkstemp(suffix=".urdf", dir=d)
    with os.fdopen(fd, "w") as f:
        f.write(text)
    return tmp, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recording", default="recording006")
    ap.add_argument("--recordings", default=DEFAULT_RECORDINGS)
    ap.add_argument("--urdf", default=DEFAULT_URDF)
    ap.add_argument("--ee-frame", default="Link7_l")
    ap.add_argument("--sim-hz", type=float, default=240.0)
    ap.add_argument("--direct", action="store_true", help="headless (no GUI), for testing")
    ap.add_argument("--max-frames", type=int, default=0, help="0 = all")
    # Default: wait for the arm to settle at each pose before sending the next.
    ap.add_argument("--settle-tol-deg", type=float, default=1.0,
                    help="consider a pose reached when every joint is within this of the command")
    ap.add_argument("--settle-timeout-s", type=float, default=1.5,
                    help="give up settling after this long (flags the pose as unreached)")
    ap.add_argument("--fixed-rate", action="store_true",
                    help="old behavior: a fixed time budget per pose instead of settling")
    ap.add_argument("--playback-hz", type=float, default=20.0,
                    help="poses/sec in --fixed-rate mode")
    args = ap.parse_args()

    # --- recorded EE trajectory ---
    npz = np.load(os.path.join(args.recordings, args.recording, "robot_states.npz"))
    ee_pos = npz["left_pos"].astype(np.float64)
    ee_quat = npz["left_quat"].astype(np.float64)
    grip = npz["gripper"].astype(np.float64)[:, 0]
    n = len(ee_pos) if not args.max_frames else min(len(ee_pos), args.max_frames)
    print(f"Recording {args.recording}: {n} EE poses")

    # --- IK model, base_offset anchored to the first recorded pose ---
    model = PinocchioArmModel(urdf_path=args.urdf, ee_frame=args.ee_frame)
    q_seed = np.clip(np.array([0.0, -0.3, 0.0, -0.6, 0.0, 0.3, 0.0]), model.lower, model.upper)
    E0 = model.fk_local(q_seed)
    S0 = quat_xyzw_to_se3(ee_pos[0], ee_quat[0])
    model.base_offset = S0 * E0.inverse()        # base_offset^-1 * S0 == E0 (start at seed)
    solver = PinocchioDLSIKSolver(model, damping=1e-2, max_iters=100, tol=1e-4)

    # --- PyBullet ---
    p.connect(p.DIRECT if args.direct else p.GUI)
    tmp_urdf, search_dir = patched_urdf(args.urdf)
    p.setAdditionalSearchPath(search_dir)
    p.setGravity(0, 0, 0)                          # kinematic view: arm holds commanded pose
    try:
        body = p.loadURDF(tmp_urdf, useFixedBase=True)
    finally:
        os.remove(tmp_urdf)
    jmap = {p.getJointInfo(body, i)[1].decode(): i for i in range(p.getNumJoints(body))}
    arm_idx = [jmap[j] for j in LEFT_ARM_JOINTS]
    grip_idx = [jmap[j] for j in LEFT_GRIPPER_JOINTS if j in jmap]
    for k, qi in zip(arm_idx, q_seed):
        p.resetJointState(body, k, float(qi))

    # --- replay: recorded EE -> IK -> sim, warm-started from the last solution ---
    sim_dt = 1.0 / args.sim_hz
    p.setTimeStep(sim_dt)
    idx = np.linspace(0, n - 1, n).astype(int)

    def cur_arm_q():
        return np.array([p.getJointState(body, k)[0] for k in arm_idx])

    settle_tol = np.radians(args.settle_tol_deg)
    settle_max_steps = max(1, int(args.settle_timeout_s * args.sim_hz))
    fixed_steps = max(1, int(args.sim_hz / args.playback_hz))   # for --fixed-rate

    q = q_seed.copy()
    ik_err, steps, n_unreached, n_timeout, n_skipped = [], [], 0, 0, 0
    mode = "fixed-rate" if args.fixed_rate else f"settle(tol={args.settle_tol_deg}deg)"
    print(f"mode: {mode}")
    print(f"{'frame':>6} | {'IK pos err(mm)':>14} {'|dq| step(deg)':>14} {'settle steps':>12} {'near-lim':>8}")
    for i in idx:
        q_des = solver.solve(IKQuery(target_pos=ee_pos[i], target_quat=ee_quat[i],
                                     current_joints=q))   # warm-start: last config is current

        # Unreachable target (outside the workspace / no IK solution within tol): skip it
        # entirely. Don't command the arm and DON'T advance the seed — the next target is
        # planned from the last reachable config, so a skip can't poison the warm-start.
        if not solver.last_reachable:
            n_skipped += 1
            print(f"{i:>6} | {solver.last_pos_err*1000:>14.2f} {'SKIP (unreachable)':>14} "
                  f"{'-':>12} {'-':>8}")
            continue

        p.setJointMotorControlArray(body, arm_idx, p.POSITION_CONTROL,
                                    targetPositions=q_des.tolist(), forces=[200.0] * len(arm_idx))
        if grip_idx:
            g = float(np.clip(grip[i], 0, 1)) * GRIPPER_CLOSE_RAD
            p.setJointMotorControlArray(body, grip_idx, p.POSITION_CONTROL,
                                        targetPositions=[g] * len(grip_idx), forces=[20.0] * len(grip_idx))

        # Default: step until the arm settles at q_des (or times out). --fixed-rate:
        # just give a fixed budget per pose (old behavior).
        n_steps = 0
        timed_out = False
        while True:
            p.stepSimulation()
            if not args.direct:
                time.sleep(sim_dt)
            n_steps += 1
            if args.fixed_rate:
                if n_steps >= fixed_steps:
                    break
            else:
                if np.max(np.abs(cur_arm_q() - q_des)) < settle_tol:
                    break
                if n_steps >= settle_max_steps:
                    timed_out = True
                    break

        res = np.linalg.norm(model.fk(q_des)[0] - ee_pos[i])    # IK's own reach error
        step = np.degrees(np.max(np.abs(q_des - q)))
        near = bool(np.any((q_des - model.lower < 0.05) | (model.upper - q_des < 0.05)))
        ik_err.append(res); steps.append(step)
        if res > 0.02:
            n_unreached += 1
        if timed_out:
            n_timeout += 1
        if i % 20 == 0 or res > 0.02 or step > 45 or timed_out:
            print(f"{i:>6} | {res*1000:>14.2f} {step:>14.2f} {n_steps:>12}{' TO' if timed_out else '':>0} {str(near):>8}")
        q = q_des                       # the last solution becomes the current seed

    ik_err, steps = np.array(ik_err), np.array(steps)
    if len(ik_err):
        print(f"\nSummary: IK pos-err median {np.median(ik_err)*1000:.1f} mm (max {ik_err.max()*1000:.1f}) | "
              f"step median {np.median(steps):.1f} deg (max {steps.max():.1f}) | "
              f"reached {len(ik_err)}/{len(idx)} | skipped(unreachable) {n_skipped}/{len(idx)} | "
              f"settle-timeouts {n_timeout}/{len(idx)}")
    else:
        print(f"\nSummary: every target was unreachable — skipped {n_skipped}/{len(idx)} "
              f"(check base_offset anchoring / workspace).")
    if not args.direct:
        print("Close the PyBullet window to exit.")
        while p.isConnected():
            time.sleep(0.1)
    p.disconnect()


if __name__ == "__main__":
    main()
