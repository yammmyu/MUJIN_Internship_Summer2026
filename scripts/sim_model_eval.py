"""Watch the trained DUAL-ARM policy drive BOTH G1 arms in PyBullet from a recorded episode.

Purpose: SEE how the model behaves — especially the right arm — on real recorded perception,
in simulation, WITHOUT the robot. This is the model-in-the-loop counterpart to
sim_replay_eval.py (which replays recorded EE poses through IK); here the arm motion comes from
the POLICY's predictions on the recording's video + proprioception, not from the recording.

Flow (per inference step):
  1. Pull one obs from the recording: head + hand_left + hand_right video frames + the dual EE /
     gripper state (RecordedObsSource — the same obs shape the live env feeds the policy).
  2. POST it to the policy server -> a 20-col action chunk L[pos3+rot6d6+grip1] ++ R[...].
  3. Decode each row, solve per-arm IK (left + right solvers), and drive BOTH arms in the sim,
     ramping to each row so the motion is smooth and velocity-bounded (like the real release).
  4. Print diagnostics so you can VALIDATE the model:
       - IK reachability + orientation error per arm (is the predicted pose attainable / sane?)
       - max per-substep joint velocity vs the safety cap (does the model demand over-cap motion?)
       - predicted-vs-recorded EE deviation per arm (does the policy track the demo, or diverge?)

Sim-only: no a2d_sdk, no hardware. The only network hop is the (remote) policy server.

Usage:
    .venv/bin/python scripts/sim_model_eval.py --recording recording005 \
        --recordings /home/chenyanyu/Documents/Humanoid/humanoid/MDM_data_collection/recordings \
        --host 10.12.11.144 --port 9001

    # headless (no PyBullet window, diagnostics only):
    .venv/bin/python scripts/sim_model_eval.py --recording recording005 --direct
"""

import argparse
import os
import sys
import time

import numpy as np
import pybullet as p

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from real_world.ik import build_solver, IKQuery, rot6d_to_quat  # noqa: E402
from real_world.sim_backend import SimEnv  # noqa: E402
from real_world.recorded_obs import RecordedObsSource  # noqa: E402
from real_world.build_data import build_predict_request  # noqa: E402
from real_world.inference_controller import post_predict, PC4080_HOST, PC4080_PORT  # noqa: E402
from real_world.timing import MAX_JOINT_STEP, RECORD_HZ  # noqa: E402

HUMANOID = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_RECORDINGS = os.path.join(HUMANOID, "MDM_data_collection", "recordings")
GRIPPER_CLOSE_THRESH = 10.0   # raw gripper >= this -> closed=1 (matches humanoid_env / inference)


def _quat_angle_deg(qa, qb):
    """Angle (deg) between two xyzw quaternions."""
    qa = np.asarray(qa, float) / (np.linalg.norm(qa) + 1e-12)
    qb = np.asarray(qb, float) / (np.linalg.norm(qb) + 1e-12)
    return float(np.degrees(2.0 * np.arccos(min(1.0, abs(float(np.dot(qa, qb)))))))


def _decode_dual(row):
    """20-col action row -> (Lpos, Lquat, Lgrip, Rpos, Rquat, Rgrip). xyzw quats."""
    a = np.asarray(row, dtype=np.float64)
    return (a[0:3], rot6d_to_quat(a[3:9]), float(a[9]),
            a[10:13], rot6d_to_quat(a[13:19]), float(a[19]))


def _drive_both(sim, q_from, q_to, render, cap=MAX_JOINT_STEP):
    """Ramp BOTH arms from q_from(14) to q_to(14) in velocity-bounded steps, rendering each.
    Returns the max per-substep joint delta actually applied (to flag over-cap demands)."""
    q_from = np.asarray(q_from, float); q_to = np.asarray(q_to, float)
    span = float(np.max(np.abs(q_to - q_from)))
    n = max(1, int(np.ceil(span / cap))) if cap > 0 else 1
    for i in range(1, n + 1):
        sim.reset_arms(q_from + (q_to - q_from) * (i / n))
        render()
    return span / n if n else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recording", default="recording005")
    ap.add_argument("--recordings", default=DEFAULT_RECORDINGS)
    ap.add_argument("--host", default=PC4080_HOST, help="policy server host")
    ap.add_argument("--port", type=int, default=PC4080_PORT, help="policy server port")
    ap.add_argument("--inference-hz", type=float, default=5.0,
                    help="obs sampling / re-prediction rate (Hz); lower = fewer, wider steps")
    ap.add_argument("--exec-rows", type=int, default=0,
                    help="policy-chunk rows to execute per inference (0 = auto = the obs stride, "
                         "i.e. receding-horizon so the sim stays aligned with the recording)")
    ap.add_argument("--max-steps", type=int, default=0, help="stop after N inferences (0 = all)")
    ap.add_argument("--seed-from-sim", action="store_true",
                    help="chain IK from the sim's CURRENT pose (closed-loop; shows accumulated drift). "
                         "Default seeds each inference from the recorded joints (isolates the model).")
    ap.add_argument("--mock-recorded", action="store_true",
                    help="NO server: use the recorded EE as the 'prediction' (ground-truth baseline). "
                         "If both arms look clean here but bad with the real server, it's the MODEL.")
    ap.add_argument("--direct", action="store_true", help="headless (no PyBullet window)")
    ap.add_argument("--slow", type=float, default=0.0, help="extra seconds to sleep per rendered substep")
    args = ap.parse_args()

    src = RecordedObsSource(args.recording, args.recordings,
                            record_hz=RECORD_HZ, inference_hz=args.inference_hz)
    stride = src.stride
    exec_rows = args.exec_rows or stride

    solL, solR = build_solver(side="left"), build_solver(side="right")
    sim = SimEnv(direct=args.direct)
    seed14 = src.arm_joints[0].copy()               # recording's first 14 joints
    sim.reset_arms(seed14)

    def render():
        p.stepSimulation()
        if not args.direct:
            time.sleep(1.0 / 240.0 + args.slow)

    _sink = "RECORDED EE (mock, ground truth)" if args.mock_recorded else f"policy {args.host}:{args.port}"
    print(f"=== model-in-sim eval: {args.recording} -> {_sink} ===")
    print(f"    obs @ {args.inference_hz} Hz (stride {stride}), exec {exec_rows} rows/inference, "
          f"seed={'sim' if args.seed_from_sim else 'recorded'}\n")
    print(f"{'infer':>5} {'row':>5} | {'L_reach':>7} {'R_reach':>7} | {'L_oErr°':>7} {'R_oErr°':>7} | "
          f"{'maxV/cap':>8} | {'L_track':>7} {'R_track':>7}")
    print("-" * 78)

    tot = dict(steps=0, l_unreach=0, r_unreach=0, overcap=0)
    n_infer = 0
    try:
        while sim.connected():
            obs = src.get_obs()
            if obs is None:
                break
            sid = obs['step_id']
            if args.mock_recorded:
                # GROUND-TRUTH baseline (no server): the "prediction" is the recorded dual EE for a
                # short horizon from here. Drives both arms from the demo itself — if THIS looks
                # distorted/over-cap, it's an IK/calibration issue, not the model.
                horizon = np.clip(np.arange(sid, sid + 12), 0, src.n - 1)
                action = np.array([[*src.robotl_eef[i], src.grips[i][0],
                                    *src.robotr_eef[i], src.grips[i][1]] for i in horizon],
                                  dtype=np.float64)
            else:
                try:
                    resp = post_predict(args.host, args.port, build_predict_request(obs), timeout=15)
                except Exception as e:
                    print(f"\n  [could not reach policy server {args.host}:{args.port}] "
                          f"{type(e).__name__}: {e}")
                    print("   Is the policy server running and reachable? Check --host/--port, the "
                          "server\n   process, and network/firewall. To test the sim path WITHOUT a "
                          "server, add --mock-recorded.")
                    break
                if 'error' in resp:
                    print(f"  [server error] {resp['error']}")
                    break
                action = np.asarray(resp['action'], dtype=np.float64)
            if action.ndim != 2 or action.shape[1] < 20:
                print(f"  [skip] unexpected action shape {action.shape} (need (T,>=20) dual-arm)")
                break
            for gc in (9, 19):
                action[:, gc] = (action[:, gc] >= GRIPPER_CLOSE_THRESH).astype(float)

            # IK seed: sim's current pose (closed-loop) or the recorded joints (isolate the model).
            if args.seed_from_sim:
                seedL, seedR = sim._cur_arms_q()[:7], sim._cur_arms_q()[7:]
            else:
                si = min(sid, len(src.arm_joints) - 1)
                seedL, seedR = src.arm_joints[si, :7].copy(), src.arm_joints[si, 7:].copy()

            nrows = min(exec_rows, action.shape[0])
            l_un = r_un = 0; l_oe = r_oe = 0.0; maxv = 0.0; l_trk = r_trk = 0.0
            for k in range(nrows):
                Lpos, Lquat, Lgrip, Rpos, Rquat, Rgrip = _decode_dual(action[k])
                qL = solL.solve(IKQuery(target_pos=Lpos, target_quat=Lquat, current_joints=seedL))
                lr, le = solL.last_reachable, _quat_angle_deg(solL.m.fk(qL)[1], Lquat)
                qR = solR.solve(IKQuery(target_pos=Rpos, target_quat=Rquat, current_joints=seedR))
                rr, re = solR.last_reachable, _quat_angle_deg(solR.m.fk(qR)[1], Rquat)
                l_un += (not lr); r_un += (not rr)
                l_oe = max(l_oe, le); r_oe = max(r_oe, re)
                # drive both arms (ramped) from the sim's current pose to this predicted config
                v = _drive_both(sim, sim._cur_arms_q(), np.concatenate([qL, qR]), render)
                maxv = max(maxv, v)
                # predicted-vs-recorded EE deviation (does the policy track the demo?)
                ri = min(sid + k, len(src.robotl_eef) - 1)
                l_trk = max(l_trk, float(np.linalg.norm(Lpos - np.asarray(src.robotl_eef[ri])[:3])))
                r_trk = max(r_trk, float(np.linalg.norm(Rpos - np.asarray(src.robotr_eef[ri])[:3])))
                seedL, seedR = qL, qR

            over = maxv / MAX_JOINT_STEP
            flag = "  <-- OVER-CAP" if over > 1.0 else ""
            print(f"{n_infer:5d} {sid:5d} | {nrows - l_un:3d}/{nrows:<3d} {nrows - r_un:3d}/{nrows:<3d} | "
                  f"{l_oe:7.2f} {r_oe:7.2f} | {over:7.2f}x | {l_trk*100:6.1f} {r_trk*100:6.1f}{flag}")
            tot['steps'] += nrows; tot['l_unreach'] += l_un; tot['r_unreach'] += r_un
            tot['overcap'] += int(over > 1.0)
            n_infer += 1
            if args.max_steps and n_infer >= args.max_steps:
                break
    except KeyboardInterrupt:
        print("\ninterrupted.")
    finally:
        src.close()

    print("\n=== summary ===")
    print(f"  inferences: {n_infer}   rows executed: {tot['steps']}")
    print(f"  LEFT  unreachable rows: {tot['l_unreach']}   orientation issues surface as L_oErr")
    print(f"  RIGHT unreachable rows: {tot['r_unreach']}   (high R_unreach / R_oErr / R_track = a right-arm model problem)")
    print(f"  inferences demanding OVER-CAP joint velocity: {tot['overcap']}")
    print(f"  track columns are predicted-vs-recorded EE deviation in CM (large = policy diverges from the demo)")
    if not args.direct and sim.connected():
        print("\n  close the PyBullet window to exit.")
        while sim.connected():
            time.sleep(0.1)
    sim.disconnect()


if __name__ == "__main__":
    main()
