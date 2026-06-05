"""Drive the policy in PyBullet from a recorded episode — inference -> our IK -> sim.

Offline test of the full deploy path WITHOUT moving the robot: a recorded episode's head +
hand_left video and EE state are fed to the policy server; each predicted EE pose is turned
into 7 left-arm joint angles by our URDF IK (real_world/ik.py) and driven into an in-process
PyBullet sim. No socket — IK is a method call and the sim is in-process; the only network
hop is the (remote) policy server.

`HumanoidEnv` owns the execution path: its `_exec_loop` runs the IK and, with `real=False`,
hands joints to the sim only. (On the robot, real=True adds a release pipeline where a
sim-executed trajectory is replayed to hardware via the GUI "释放到真机" button — never reached
here: real=False plus an injected _NoRobot mean nothing can touch the robot.)

Threading: the main thread steps the sim; HumanoidEnv's exec thread runs IK and feeds it;
(policy mode) InferenceController's thread fetches recorded obs and calls the policy.

Usage:
    # Validate IK + exec + sim WITHOUT the policy server (feeds recorded EE poses directly):
    .venv/bin/python scripts/sim_infer_eval.py --source replay \
        --recording recording001 --recordings /home/chenyanyu/Documents/Humanoid/humanoid/MDM_data_collection/recordings

    # Full inference path (needs the policy server reachable):
    .venv/bin/python scripts/sim_infer_eval.py --source policy \
        --recording recording001 --recordings /home/chenyanyu/Downloads/recordings \
        --host 10.12.11.144 --port 9001
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from real_world.ik import quat_xyzw_to_se3, DEFAULT_URDF, DEFAULT_CALIBRATION  # noqa: E402
from real_world.sim_backend import SimEnv, load_trajectory  # noqa: E402
from real_world.recorded_obs import RecordedObsSource  # noqa: E402
# These import a2d_sdk (only present on the robot machine), so keep them after the SDK-free ones.
from real_world.humanoid_env import HumanoidEnv, RECORD_HZ  # noqa: E402
from real_world.inference_controller import (  # noqa: E402
    InferenceController, PC4080_HOST, PC4080_PORT, INFERENCE_HZ,
)

HUMANOID = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_RECORDINGS = os.path.join(HUMANOID, "MDM_data_collection", "recordings")


class _NoRobot:
    """Hard safety stand-in for the SDK in sim-only runs.

    Injected as both robot and robot_controller so HumanoidEnv never constructs a real
    Robot()/RobotController() and holds NO live DDS handle. Any attempted SDK call (a read
    or, critically, move_arm/move_gripper) raises instead of reaching hardware — so it is
    provably impossible for this runner to move the robot, regardless of flags."""

    def __getattr__(self, name):
        def _blocked(*args, **kwargs):
            raise RuntimeError(
                f"_NoRobot: SDK call '{name}' blocked — sim_infer_eval is sim-only and must "
                f"never touch the robot.")
        return _blocked


def pose_to_action_row(pos, quat_xyzw, grip):
    """Recorded EE pose -> a policy-format action row [eef_pos(3), 6D_rot(6), gripper(1)].

    The 6D rotation is the first two columns of the rotation matrix (the same convention
    humanoid_env.rot6d_to_quat decodes), so a replayed row flows through the identical exec
    path as a real prediction — exercising IK without the policy server.
    """
    R = quat_xyzw_to_se3(pos, quat_xyzw).rotation
    rot6d = [*R[:, 0], *R[:, 1]]
    return [float(pos[0]), float(pos[1]), float(pos[2]), *map(float, rot6d), float(grip)]


def run_replay(env, args):
    """Feed the recording's own EE poses straight through the exec path (no policy)."""
    ee_pos, ee_quat, grip, _arm, n = load_trajectory(
        args.recordings, args.recording, args.max_frames)
    rows = [pose_to_action_row(ee_pos[i], ee_quat[i], grip[i]) for i in range(n)]
    print(f"[sim_infer_eval] replay: {n} recorded EE poses -> IK -> sim")
    env.submit_actions(rows)
    # Done when the exec thread has drained every row into the sim.
    return lambda: env.queue_empty()


def run_policy(env, obs, args):
    """Run the policy on recorded obs; predictions flow through the exec path to the sim."""
    inf = InferenceController(env, robot_info=None, obs_source=obs,
                              host=args.host, port=args.port,
                              inference_hz=args.inference_hz)
    inf.start()
    inf.auto_inference()
    print(f"[sim_infer_eval] policy: streaming recorded obs to {args.host}:{args.port}")
    # Done once the recording is exhausted and the last chunk has drained.
    return lambda: obs.done and env.queue_empty(), inf


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recording", default="recording001")
    ap.add_argument("--recordings", default=DEFAULT_RECORDINGS)
    ap.add_argument("--source", choices=["policy", "replay"], default="policy",
                    help="policy: run the policy server on recorded obs; "
                         "replay: feed recorded EE poses directly (no server)")
    ap.add_argument("--max-frames", type=int, default=0, help="0 = all (replay only)")
    ap.add_argument("--urdf", default=DEFAULT_URDF)
    ap.add_argument("--calibration", default=DEFAULT_CALIBRATION)
    ap.add_argument("--sim-hz", type=float, default=240.0)
    ap.add_argument("--inference-hz", type=float, default=INFERENCE_HZ)
    ap.add_argument("--direct", action="store_true", help="headless (no GUI)")
    ap.add_argument("--host", default=PC4080_HOST, help="policy server host")
    ap.add_argument("--port", type=int, default=PC4080_PORT, help="policy server port")
    args = ap.parse_args()

    # Seed the IK/sim from the recording's first live left-arm joints.
    _ee_pos, _ee_quat, _grip, arm_joints, _n = load_trajectory(args.recordings, args.recording, 0)
    q0 = arm_joints[0, :7]

    # In-process sim (main thread owns it).
    sim = SimEnv(urdf=args.urdf, calibration=args.calibration,
                 sim_hz=args.sim_hz, direct=args.direct)
    sim.reset_arm(q0)

    # Execution path: IK -> sim only. real=False AND a tripwire _NoRobot in place of the SDK
    # handles => no live DDS connection exists and any SDK call raises. Provably no hardware
    # motion. (Also means no robot/DDS needs to be running to use this runner.)
    env = HumanoidEnv(robot=_NoRobot(), robot_controller=_NoRobot(),
                      sim=sim, real=False, seed_q=q0, frequency=RECORD_HZ)
    env.start(run_collect=False, run_exec=True)

    obs = None
    inf = None
    try:
        if args.source == "replay":
            stop_fn = run_replay(env, args)
        else:
            obs = RecordedObsSource(args.recording, args.recordings,
                                    record_hz=RECORD_HZ, inference_hz=args.inference_hz)
            stop_fn, inf = run_policy(env, obs, args)
        sim.run_until(stop_fn)          # blocks on the main thread until drained / window closed
    finally:
        if inf is not None:
            inf.stop()
        env.stop()
        if obs is not None:
            obs.close()
        sim.disconnect()
    print("[sim_infer_eval] done.")


if __name__ == "__main__":
    main()
