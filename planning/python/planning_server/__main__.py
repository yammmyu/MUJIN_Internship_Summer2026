"""CLI entry: start the planning server, optionally run a vision pass, drop into IPython.

Run from the planning/ directory:
    PYTHONPATH=$PYTHONPATH:$(pwd)/python python3 -m planning_server \
        --robotFile A2D_Omnipicker/A2D.kinbody.xml \
        --rgbPath python/detection/temp/head_20260527_134928.jpg \
        --depthPath python/detection/temp/depth.png \
        --fx 600 --fy 600 --cx 320 --cy 240 \
        --depthScale 0.001

Without --rgbPath/--depthPath the server starts empty and you can populate
the scene from the IPython prompt via `process(...)`.
"""
from __future__ import annotations

import argparse
import logging

import openravepy

from a2d_reachability.__main__ import LEFT_MANIP, RIGHT_MANIP

from .server import PlanningServer


@openravepy.with_destroy
def Main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robotFile", default="A2D_Omnipicker/A2D.kinbody.xml")
    parser.add_argument("--logLevel", default="WARNING")

    # Vision input (optional at startup; can also be called from IPython).
    parser.add_argument("--rgbPath", help="RGB image (jpg/png) for the initial detection pass.")
    parser.add_argument("--depthPath", help="Depth image (16-bit PNG or .npy) aligned to RGB.")
    parser.add_argument("--fx", type=float, help="Pinhole intrinsic fx (px).")
    parser.add_argument("--fy", type=float, help="Pinhole intrinsic fy (px).")
    parser.add_argument("--cx", type=float, help="Pinhole intrinsic cx (px).")
    parser.add_argument("--cy", type=float, help="Pinhole intrinsic cy (px).")
    parser.add_argument("--depthScale", type=float, default=0.001,
                        help="Multiply raw depth by this to get meters (default 0.001 = uint16 mm).")
    parser.add_argument("--camYaw", type=float, default=0.0)
    parser.add_argument("--camPitch", type=float, default=0.0)
    parser.add_argument("--camRoll", type=float, default=0.0)
    parser.add_argument("--conf", type=float, default=0.5)

    # Optional autonomous grasp of the first detection.
    parser.add_argument("--grab", action="store_true",
                        help="After detection, grab the first detected box.")
    parser.add_argument("--manip", default=LEFT_MANIP,
                        choices=(LEFT_MANIP, RIGHT_MANIP),
                        help="Manipulator to use for --grab.")
    parser.add_argument("--skipIkPrep", action="store_true",
                        help="Skip up-front IK model load (faster startup, slower first Grab).")

    # TCP server (motion-planning GUI client)
    parser.add_argument("--tcpHost", default="0.0.0.0",
                        help="TCP listen host for the GUI client (default 0.0.0.0).")
    parser.add_argument("--tcpPort", type=int, default=9100,
                        help="TCP listen port for the GUI client (default 9100).")
    parser.add_argument("--noTcp", action="store_true",
                        help="Skip starting the TCP server.")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.logLevel.upper(), logging.WARNING))

    server = PlanningServer(
        robotFile=args.robotFile,
        prepareManipulators=not args.skipIkPrep,
        logLevel=args.logLevel,
    )

    if not args.noTcp:
        server.StartTcpServer(host=args.tcpHost, port=args.tcpPort)

    visionGiven = bool(args.rgbPath and args.depthPath)
    if visionGiven:
        missing = [n for n in ("fx", "fy", "cx", "cy") if getattr(args, n) is None]
        if missing:
            raise SystemExit("--rgbPath/--depthPath given but missing %s" % ", ".join(missing))
        added = server.ProcessVision(
            rgbPath=args.rgbPath, depthPath=args.depthPath,
            fx=args.fx, fy=args.fy, cx=args.cx, cy=args.cy,
            depthScale=args.depthScale,
            cameraYawDeg=args.camYaw, cameraPitchDeg=args.camPitch, cameraRollDeg=args.camRoll,
            minConfidenceThreshold=args.conf,
        )
        if args.grab and added:
            server.Grab(added[0]["name"], manip=args.manip)
        elif args.grab:
            print("\033[31;1m--grab requested but no detection was added.\033[0m")

    # --------------------------- IPython helpers --------------------------- #
    env = server.env
    robot = server.robot
    handles = server.handles

    def process(rgbPath, depthPath, fx, fy, cx, cy,
                depthScale=0.001,
                cameraYawDeg=0.0, cameraPitchDeg=0.0, cameraRollDeg=0.0,
                conf=0.5, clearPrevious=True,
                enablePointCloud=True,
                pointCloudVoxelSize=0.02,
                pointCloudMaskMargin=0.03,
                pointCloudBoxHalfSize=0.005,
                pointCloudMaxRange=3.0,
                pointCloudMinRange=0.10,
                pointCloudColor=(0.45, 0.45, 0.55)):
        return server.ProcessVision(
            rgbPath=rgbPath, depthPath=depthPath,
            fx=fx, fy=fy, cx=cx, cy=cy,
            depthScale=depthScale,
            cameraYawDeg=cameraYawDeg, cameraPitchDeg=cameraPitchDeg, cameraRollDeg=cameraRollDeg,
            minConfidenceThreshold=conf, clearPrevious=clearPrevious,
            enablePointCloud=enablePointCloud,
            pointCloudVoxelSize=pointCloudVoxelSize,
            pointCloudMaskMargin=pointCloudMaskMargin,
            pointCloudBoxHalfSize=pointCloudBoxHalfSize,
            pointCloudMaxRange=pointCloudMaxRange,
            pointCloudMinRange=pointCloudMinRange,
            pointCloudColor=pointCloudColor,
        )

    def grab(name, manip=LEFT_MANIP, **kw):
        return server.Grab(name, manip=manip, **kw)

    def gen_box_ik(name, manip=LEFT_MANIP, **kw):
        return server.GenerateBoxIK(name, manip=manip, **kw)

    def release(name=None):
        return server.Release(name)

    def add_box(name="box0", halfExtents=(0.06, 0.04, 0.10), pose=None,
                color=(0.85, 0.35, 0.10)):
        return server.AddBox(name, halfExtents, pose, color)

    def del_box(name="box0"):
        return server.DeleteBox(name)

    def clear_cloud():
        """Remove the scene point cloud KinBody."""
        return server.ClearPointCloud()

    # ---- Motion planning TCP helpers (require connected GUI client) ----
    def trigger_detection(**kw):
        """Ask the GUI for latest head RGB+Depth and run detection. kwargs
        accepted: fx, fy, cx, cy, depthScale, cameraYawDeg, cameraPitchDeg,
        cameraRollDeg, minConfidenceThreshold, clearPrevious, timeout."""
        return server.TriggerDetection(**kw)

    def plan_grab(name, manip=LEFT_MANIP, **kw):
        """Plan & visualize a grasp; pause live-state during planning."""
        return server.PlanGrab(name, manip=manip, **kw)

    def send_trajectory(**kw):
        """Dispatch the pending PlanGrab() trajectory to the GUI client."""
        return server.SendTrajectory(**kw)

    def discard_trajectory():
        return server.DiscardTrajectory()

    def pause_live():
        return server.PauseLiveState()

    def resume_live():
        return server.ResumeLiveState()

    print("\nPlanningServer IPython namespace:")
    print("  server                              # PlanningServer instance")
    print("  env, robot, handles                 # already bound")
    print("  process(rgbPath, depthPath, fx, fy, cx, cy, depthScale=0.001,")
    print("          cameraYawDeg=0, cameraPitchDeg=0, cameraRollDeg=0,")
    print("          enablePointCloud=True, pointCloudVoxelSize=0.02,")
    print("          pointCloudMaskMargin=0.03, pointCloudBoxHalfSize=0.005,")
    print("          pointCloudMaxRange=3.0, pointCloudMinRange=0.10)")
    print("  clear_cloud()                       # remove scene point cloud body")
    print("  add_box(name, halfExtents, pose)    # manual scene authoring")
    print("  del_box(name)")
    print("  gen_box_ik(name, manip=...)         # visualize grasp IK candidates")
    print("  grab(name, manip=...)               # plan + replay grasp (no TCP send)")
    print("  release(name=None)                  # release a held object")
    print("  server.objects                      # name -> {'body','detection'}")
    print("  manips: LEFT=%r, RIGHT=%r" % (LEFT_MANIP, RIGHT_MANIP))
    print("  --- motion planning TCP (GUI client) ---")
    print("  server.tcp_connected                # True when GUI is connected")
    print("  trigger_detection(fx=..., fy=..., cx=..., cy=...)")
    print("                                      # request RGB+Depth, run detector")
    print("  plan_grab(name, manip=...)          # plan grasp; cache pending_trajectory")
    print("  send_trajectory(wait_ack=True)      # dispatch pending traj to client")
    print("  discard_trajectory()                # drop pending traj & resume live")
    print("  pause_live() / resume_live()        # manual live-state freeze toggle")
    print("  server.pending_trajectory           # last planned (until sent/discarded)")
    print("  server.last_client_state            # most recent state msg from client")

    from IPython import embed
    embed()


if __name__ == "__main__":
    Main()
