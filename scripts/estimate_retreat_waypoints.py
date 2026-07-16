#!/usr/bin/env python3
"""Estimate the 5 retreat waypoints from recorded episodes.

The missed-grasp recovery used to drive both arms all the way back to a single fixed home pose.
Instead we retreat only as far as needed: this script derives 5 average joint-space waypoints
(both arms, 14-vec each) that trace the arm's APPROACH BEFORE the grasp, evenly spread in phase.
At run time the retreat picks the nearest waypoint that isn't ahead of the arm's current approach
phase (see real_world/retreat.py), so a near-object miss retreats a little and an early miss
retreats more — never an unconditional snap to the start.

Method, per recording:
  * grasp frame = first frame the RIGHT gripper reads closed (>= GRIPPER_CLOSE_THRESH). The right
    gripper is the grasp (the left stays open); this is the same signal the recovery monitor uses.
  * pre-grip window = frames [0, round(PRE_GRIP_END_FRAC * grasp_frame)]. Ending at 85% (default)
    keeps the LAST waypoint a real distance before the grasp — retreating to the frame right
    before the close would be pointless.
  * sample arm_joints (14-vec = [left7, right7]) at 5 phases evenly spaced over that window
    (fractions 0, .25, .5, .75, 1).
Then average each phase's 14-vec across all recordings. Waypoint 1 is the average START pose (also
the pose the top of each auto run homes to); waypoint 5 is the average pre-grasp pose.

Writes real_world/config/retreat_waypoints.json and prints a per-phase spread (std) sanity check.

Usage:
    .venv/bin/python scripts/estimate_retreat_waypoints.py [RECORDINGS_DIR] \
        [--end-frac 0.85] [--n 5] [--out real_world/config/retreat_waypoints.json]
"""

import argparse
import glob
import json
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
DEFAULT_DIR = os.path.join(_ROOT, "MDM_data_collection", "recordings")
DEFAULT_OUT = os.path.join(_ROOT, "real_world", "config", "retreat_waypoints.json")
GRIPPER_CLOSE_THRESH = 50.0        # mirror real_world.postprocess (avoid importing torch-adjacent deps)


def _grasp_frame(gripper):
    """First frame the RIGHT gripper reads closed, or None if it never closes."""
    closed = np.asarray(gripper, float)[:, 1] >= GRIPPER_CLOSE_THRESH
    return int(np.argmax(closed)) if closed.any() else None


def estimate_waypoints(recordings_dir, end_frac, n):
    """(n, 14) average pre-grasp waypoints + diagnostics dict."""
    paths = sorted(glob.glob(os.path.join(recordings_dir, "*", "robot_states.npz")))
    if not paths:
        raise SystemExit(f"no robot_states.npz under {recordings_dir}")
    fracs = np.linspace(0.0, 1.0, n)             # phase samples incl. endpoints
    per_phase = [[] for _ in range(n)]           # per_phase[j] = list of 14-vecs across recordings
    used = 0
    skipped = 0
    for p in paths:
        d = np.load(p, allow_pickle=True)
        q = np.asarray(d["arm_joints"], float)   # (N,14) = [left7, right7]
        g = _grasp_frame(d["gripper"])
        if g is None or g < 2:                   # no grasp / grasps immediately -> unusable approach
            skipped += 1
            continue
        end = int(round(end_frac * g))           # last sampled frame (before the grasp)
        end = max(1, min(end, len(q) - 1))
        for j, f in enumerate(fracs):
            per_phase[j].append(q[int(round(f * end))])
        used += 1
    waypoints = np.stack([np.mean(per_phase[j], axis=0) for j in range(n)])   # (n,14)
    stds = np.stack([np.std(per_phase[j], axis=0) for j in range(n)])         # (n,14)
    return waypoints, dict(n_recordings=len(paths), used=used, skipped=skipped,
                           fracs=fracs.tolist(), stds=stds)


def main():
    ap = argparse.ArgumentParser(description="Estimate the retreat waypoints from recordings.")
    ap.add_argument("recordings_dir", nargs="?", default=DEFAULT_DIR)
    ap.add_argument("--end-frac", type=float, default=0.85,
                    help="fraction of each episode's grasp frame the LAST waypoint sits at (default 0.85)")
    ap.add_argument("--n", type=int, default=5, help="number of waypoints (default 5)")
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    W, diag = estimate_waypoints(args.recordings_dir, args.end_frac, args.n)
    payload = {
        "_note": ("Average pre-grasp APPROACH waypoints (both arms, 14 = [left7,right7], rad), evenly "
                  "spread in phase, from scripts/estimate_retreat_waypoints.py. Waypoint 0 = average "
                  "start pose (auto-run homes here); the last = ~end_frac of the way to the grasp. The "
                  "recovery retreat moves to the nearest waypoint not ahead of the current approach "
                  "phase. Re-run the script if the task/recordings change."),
        "end_frac": args.end_frac,
        "n_recordings_used": diag["used"],
        "waypoints": W.tolist(),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"# {diag['used']} recordings used ({diag['skipped']} skipped), end_frac={args.end_frac}, "
          f"n={args.n}")
    print(f"# wrote {args.out}")
    np.set_printoptions(precision=3, suppress=True)
    for j in range(args.n):
        print(f"WP{j+1} (phase {diag['fracs'][j]:.2f})  max joint std across recordings = "
              f"{diag['stds'][j].max():.3f} rad")
        print(f"    {np.array(W[j])}")


if __name__ == "__main__":
    main()
