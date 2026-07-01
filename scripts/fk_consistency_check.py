"""Offline FK-consistency probe: does our URDF FK match the SDK's firmware FK?

The SDK's EE pose (`get_motion_status()['frames']['arm_left_link7']`) is computed by
firmware, NOT from the URDF, and the firmware frame names don't exist in the URDF. So
before trusting URDF-based IK we must confirm `my_FK(q) == SDK_FK(q)` on real data and
resolve the unknowns: which URDF link is the EE, which `arm_joints` columns are the left
arm, the quaternion convention, and the constant base frame offset.

This runs fully offline against recordings produced by MDM_data_collection/
robot_data_collect.py — each `robot_states.npz` pairs raw `arm_joints` with the SDK EE
pose (`left_pos`, `left_quat`). No robot, no GPU.

Method
------
For each candidate (ee_frame, joint-slice, quat-convention):
  A_i = base_link^M_ee  = our reduced-model FK at the recorded left-arm joints
  B_i = fbase^M_ee      = recorded SDK EE pose
If a single constant X (= fbase^M_baselink) explains all frames, then B_i = X * A_i, so
X_i = B_i * A_i^-1 should be constant. We take X = SE3-mean of {X_i} and measure the
residual ||X*A_i - B_i|| over all frames. The config with the lowest residual wins; the
fitted X is the `base_offset` written to fk_calibration.json.

Base-frame verdict: a low residual with a *constant* X (+ the 7 arm joints alone) means
the firmware frame is effectively arm/torso-relative -> offline calibration is valid.
An irreducibly high residual for ALL configs means the EE pose depends on unrecorded
waist/lift DOF -> must re-collect with waist logged (the npz has no waist columns).

Usage:
    .venv/bin/python scripts/fk_consistency_check.py \
        [--recordings .../MDM_data_collection/recordings] \
        [--num-recordings 5] [--per-recording 60] [--write]
"""

import argparse
import dataclasses
import glob
import json
import os
import sys

import numpy as np
import pinocchio as pin

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from real_world.ik import (  # noqa: E402
    PinocchioArmModel, FKCalibration, DEFAULT_URDF, DEFAULT_CALIBRATION,
    DEFAULT_CALIBRATION_RIGHT, LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS,
    se3_to_pos_quat_xyzw,
)

DEFAULT_RECORDINGS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "MDM_data_collection", "recordings",
)

# Per-side candidates. The joint slice is fixed per side (left arm = arm_joints[0:7], right =
# [7:14]); the model uses that side's URDF joints and its own EE-frame candidates.
SIDE_CFG = {
    "left":  {"arm_joints": LEFT_ARM_JOINTS,  "slice": slice(0, 7),  "slice_name": "left_first",
              "ee_frames": ["Link7_l", "left_base_link", "gripper_center"],
              "pos_key": "left_pos",  "quat_key": "left_quat",  "out": DEFAULT_CALIBRATION},
    "right": {"arm_joints": RIGHT_ARM_JOINTS, "slice": slice(7, 14), "slice_name": "right_first",
              "ee_frames": ["Link7_r", "right_base_link", "gripper_center"],
              "pos_key": "right_pos", "quat_key": "right_quat", "out": DEFAULT_CALIBRATION_RIGHT},
}
QUAT_CONVENTIONS = ["xyzw", "wxyz"]


def load_samples(recordings_dir, num_recordings, per_recording, cfg):
    """Gather (q_arm7, sdk_pos, sdk_quat_raw) for the chosen side from recordings."""
    npzs = sorted(glob.glob(os.path.join(recordings_dir, "*", "robot_states.npz")))
    if not npzs:
        raise FileNotFoundError(f"no robot_states.npz under {recordings_dir}")
    npzs = npzs[:num_recordings]
    q, pos, quat = [], [], []
    within_joint_std = []   # per-recording mean joint std (to detect frozen joints)
    for path in npzs:
        d = np.load(path)
        n = len(d["timestamps"])
        idx = np.linspace(0, n - 1, min(per_recording, n)).astype(int)
        aj = d["arm_joints"][idx]
        q.append(aj[:, cfg["slice"]])
        pos.append(d[cfg["pos_key"]][idx])
        quat.append(d[cfg["quat_key"]][idx])
        # freeze-detect on THIS side's joint columns only (a frozen arm defeats calibration).
        within_joint_std.append(d["arm_joints"][:, cfg["slice"]].astype(float).std(axis=0).mean())
    return (np.concatenate(q), np.concatenate(pos), np.concatenate(quat),
            len(npzs), float(np.max(within_joint_std)))


def sdk_se3(pos, quat_raw, convention):
    """Recorded SDK pose -> SE3. quat_raw is stored as [a,b,c,d]; interpret per convention."""
    a, b, c, d = quat_raw
    if convention == "xyzw":
        xyzw = np.array([a, b, c, d], dtype=np.float64)
    else:  # stored as wxyz
        xyzw = np.array([b, c, d, a], dtype=np.float64)
    xyzw /= (np.linalg.norm(xyzw) + 1e-12)
    return pin.SE3(pin.Quaternion(xyzw).matrix(), np.asarray(pos, dtype=np.float64))


def se3_mean(Xs, iters=20):
    """Fréchet mean of a list of SE3 via tangent-space iteration."""
    M = Xs[0]
    for _ in range(iters):
        tang = np.mean([pin.log6(M.actInv(X)).vector for X in Xs], axis=0)
        M = M * pin.exp6(tang)
        if np.linalg.norm(tang) < 1e-12:
            break
    return M


def residuals(X, As, Bs):
    """Per-frame position (m) and rotation (deg) error of B_i vs X*A_i."""
    pos_err, rot_err = [], []
    for A, B in zip(As, Bs):
        P = X * A
        pos_err.append(np.linalg.norm(P.translation - B.translation))
        rel = P.rotation.T @ B.rotation
        rot_err.append(np.linalg.norm(pin.log3(rel)))
    return np.array(pos_err), np.degrees(np.array(rot_err))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recordings", default=DEFAULT_RECORDINGS)
    ap.add_argument("--urdf", default=DEFAULT_URDF)
    ap.add_argument("--side", choices=["left", "right"], default="left",
                    help="which arm to calibrate (right uses right_pos/right_quat + arm_joints[7:14])")
    ap.add_argument("--num-recordings", type=int, default=6)
    ap.add_argument("--per-recording", type=int, default=60)
    ap.add_argument("--write", action="store_true",
                    help="write the winning config to the side's fk_calibration file")
    ap.add_argument("--out", default=None, help="override output path (default per --side)")
    args = ap.parse_args()

    cfg = SIDE_CFG[args.side]
    out_path = args.out or cfg["out"]
    print(f"Calibrating the {args.side.upper()} arm "
          f"(joints[{cfg['slice'].start}:{cfg['slice'].stop}], "
          f"EE={cfg['pos_key']}/{cfg['quat_key']})\n")

    qA, pos, quat, n_rec, within_qstd = load_samples(
        args.recordings, args.num_recordings, args.per_recording, cfg)
    n = len(pos)
    print(f"Loaded {n} samples from {n_rec} recordings under {args.recordings}\n")

    # Data sanity: joints must vary WITHIN a recording, or FK can't be calibrated. We've
    # seen recordings where arm_joints is frozen (stale arm_joint_states()) while the
    # EE pose moves -> calibration is impossible. (Pooled std hides this because the
    # frozen value differs between recordings, so check the max within-recording std.)
    posstd = pos.std(axis=0).mean()
    print(f"sanity: max within-recording arm-joint std = {within_qstd:.5f} rad | "
          f"pooled left_pos std = {posstd:.5f} m")
    if within_qstd < 1e-3:
        print("\nVerdict: BLOCKED — arm_joints is effectively FROZEN (≈0 variance) while the "
              "EE pose moves. The recordings contain EE poses but no live joint angles "
              "(stale arm_joint_states() at collection time). FK calibration is impossible "
              "from this data. Fix data collection to log live arm_joints and re-record, OR "
              "calibrate live on hardware (read joints + get_motion_status together while "
              "moving the arm).")
        return

    results = []
    for ee_frame in cfg["ee_frames"]:
        try:
            model = PinocchioArmModel(urdf_path=args.urdf, ee_frame=ee_frame,
                                      arm_joints=cfg["arm_joints"])
        except ValueError:
            print(f"  (skip ee_frame {ee_frame!r}: not in URDF)")
            continue
        As = [model.fk_local(q) for q in qA]
        for conv in QUAT_CONVENTIONS:
            Bs = [sdk_se3(pos[i], quat[i], conv) for i in range(n)]
            Xs = [B * A.inverse() for A, B in zip(As, Bs)]
            X = se3_mean(Xs)
            pe, re = residuals(X, As, Bs)
            results.append(dict(
                ee_frame=ee_frame, slice=cfg["slice_name"], quat=conv, X=X,
                pos_mean=pe.mean(), pos_med=np.median(pe), pos_max=pe.max(),
                rot_mean=re.mean(), rot_med=np.median(re), rot_max=re.max(),
            ))

    results.sort(key=lambda r: (r["rot_med"] + 50 * r["pos_med"]))
    print(f"{'ee_frame':16} {'slice':12} {'quat':5} | "
          f"{'pos_med(m)':>10} {'pos_max':>8} | {'rot_med(deg)':>12} {'rot_max':>8}")
    print("-" * 86)
    for r in results:
        print(f"{r['ee_frame']:16} {r['slice']:12} {r['quat']:5} | "
              f"{r['pos_med']:10.4f} {r['pos_max']:8.4f} | "
              f"{r['rot_med']:12.3f} {r['rot_max']:8.3f}")

    best = results[0]
    print("\n=== Best config ===")
    print(f"  ee_frame={best['ee_frame']}  slice={best['slice']}  quat={best['quat']}")
    print(f"  residual: pos median {best['pos_med']*1000:.1f} mm (max {best['pos_max']*1000:.1f}), "
          f"rot median {best['rot_med']:.2f}deg (max {best['rot_max']:.2f})")
    bp, bq = se3_to_pos_quat_xyzw(best["X"])
    print(f"  base_offset pos={np.round(bp,4).tolist()} quat_xyzw={np.round(bq,4).tolist()}")

    # Verdict
    if best["pos_med"] < 0.01 and best["rot_med"] < 2.0:
        verdict = ("PASS: a constant base_offset + the 7 arm joints explain the SDK EE "
                   "pose. Firmware frame is arm/torso-relative; offline calibration is valid.")
    elif best["pos_med"] < 0.03 and best["rot_med"] < 5.0:
        verdict = ("MARGINAL: residual is small but above target. Likely usable; consider "
                   "more recordings / check joint order. Inspect before trusting on hardware.")
    else:
        verdict = ("FAIL: no config gets a low residual. The EE pose likely depends on "
                   "unrecorded waist/lift DOF (npz has no waist columns) or the joint order "
                   "differs. Re-collect with waist logged, or pin the waist pose.")
    print(f"\nVerdict: {verdict}")

    if args.write:
        cal = FKCalibration(
            ee_frame=best["ee_frame"],
            quat_convention=best["quat"],
            arm_joint_slice=best["slice"],
            base_offset_pos=[float(v) for v in bp],
            base_offset_quat_xyzw=[float(v) for v in bq],
            pos_residual_m=float(best["pos_med"]),
            rot_residual_deg=float(best["rot_med"]),
        )
        with open(out_path, "w") as f:
            json.dump(dataclasses.asdict(cal), f, indent=2)
        print(f"\nWrote calibration -> {out_path}")
    else:
        print(f"\n(dry run; pass --write to save {out_path})")


if __name__ == "__main__":
    main()
