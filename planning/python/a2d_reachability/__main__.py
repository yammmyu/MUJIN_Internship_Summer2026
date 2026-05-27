"""
A2D humanoid dual-arm motion planning and IK visualization.

Migrated from the reversed_anyware reachability test. Loads the A2D
kinbody/robot, computes IK for the left and right manipulators against
target poses, plans trajectories with BiRRT, and replays them in the
OpenRAVE viewer.

Entry point (env should provide openravepy):
    PYTHONPATH=$PYTHONPATH:$(pwd)/python python3 -m a2d_reachability \
        --robotFile A2D_Omnipicker/A2D.kinbody.xml \
        --leftTarget 0.45 0.55 1.10 0 0 0 \
        --rightTarget 0.45 -0.55 1.10 0 0 0
"""
import argparse
import logging
import os
import sys
import time

import numpy as np
import openravepy
from openravepy import (
    DebugLevel,
    Environment,
    IkParameterization,
    IkParameterizationType,
    PlanningError,
    RaveCreateKinBody,
    RaveGetDefaultViewerType,
    RaveSetDebugLevel,
    interfaces,
    misc,
    quatFromRotationMatrix,
)
from openravepy.databases import inversekinematics, linkstatistics

from .reach import FindIkSolutions, IkFailureJointLimits
from .spatial import Pose as P
from .spatial import Quaternion as Q
from .spatial import Translation as T
from .spatial import Vec3

log = logging.getLogger(__name__)

X = Vec3([1, 0, 0])
Y = Vec3([0, 1, 0])
Z = Vec3([0, 0, 1])

LEFT_MANIP = "gripper_center"
RIGHT_MANIP = "right_gripper_center"

p0=[ 1.53137720e+00, -2.86843091e-01, -1.54240549e+00, -1.29224229e+00, 1.50418999e-02,  1.51589894e+00, -2.76407991e-02,  0.00000000e+00, 1.77954229e-16, -2.11344958e-16,  0.00000000e+00,  0.00000000e+00, 0.00000000e+00,  0.00000000e+00]
m0=[-0.69901767, -0.71467714, -0.0219021 , -0.01145323,  0.43401373, 0.09348374,  0.95005244]


def _ComputeJointValueScore(val, low, high):
    import math

    mu = (low + high) / 2.0
    var = (high - low) ** 2 / 4.0 / math.log(2)
    return math.exp(-((val - mu) ** 2) / var)


def ComputeJointValuesScore(jointValues, joints):
    return min(
        _ComputeJointValueScore(jointValue, lowLimits[0], highLimits[0])
        for jointValue, (lowLimits, highLimits) in zip(
            jointValues, (joint.GetLimits() for joint in joints)
        )
    )


def InitEnvironment(robotFile):
    # type: (str) -> Environment
    extraData = [
        os.path.dirname(os.path.realpath(robotFile)) or ".",
        os.path.realpath("."),
    ]
    os.environ["OPENRAVE_DATA"] = ":".join(
        [p for p in extraData if os.path.isdir(p)]
        + os.environ.get("OPENRAVE_DATA", "").split(":")
    )

    env = Environment(0)
    env.SetViewer(RaveGetDefaultViewerType(), True)

    robot = env.ReadRobotURI(robotFile)
    if robot is None:
        raise RuntimeError("Failed to read robot from %r" % robotFile)
    env.Add(robot)

    env.GetViewer().SetCamera(
        P.I + Z.RotDeg(-90) + X.RotDeg(-90) + [0, -1, -3.5], 3.5
    )
    env.GetViewer().SendCommand("SetProjectionMode perspective")
    RaveSetDebugLevel(DebugLevel.Error)
    return env


def GetRobot(env):
    robots = env.GetRobots()
    assert len(robots) >= 1, "No robot loaded"
    return robots[0]


def PrepareManipulator(robot, manipName):
    robot.SetActiveManipulator(manipName)
    manip = robot.GetManipulator(manipName)
    robot.SetActiveDOFs(manip.GetArmIndices())

    env = robot.GetEnv()
    with env:
        ikmodel = inversekinematics.InverseKinematicsModel(
            robot, iktype=IkParameterizationType.Transform6D, freejoints=['Joint5_l' if manipName == LEFT_MANIP else 'Joint5_r']
        )
        if not ikmodel.load():
            log.warning("Generating IK model for %s (one-time, may take a while)", manipName)
            ikmodel.generate(iktype=IkParameterizationType.Transform6D, forceikbuild=False, ikfastmaxcasedepth=3, precision=6)
            ikmodel.save()
            # ikmodel.autogenerate()

        lmodel = linkstatistics.LinkStatisticsModel(robot)
        if not lmodel.load():
            lmodel.autogenerate()
        lmodel.setRobotWeights()
        lmodel.setRobotResolutions(xyzdelta=0.02)
    return manip


def BuildTargetPose(args):
    """Build a target Pose from CLI/script arguments.

    3 values:  (x, y, z)                       — position only, identity rotation.
    6 values:  (x, y, z, rx, ry, rz)           — position + intrinsic XYZ Euler
               rotations in radians. The body is rotated first by rx around its
               own X axis, then by ry around the resulting body Y axis, then by
               rz around the resulting body Z axis. Equivalent to ROS RPY
               (extrinsic ZYX). For "X 45° then Y 90°" use (..., pi/4, pi/2, 0).
    7 values:  (qw, qx, qy, qz, x, y, z)       — position + explicit quaternion,
               matching the underlying Pose layout in spatial.py.
    """
    if len(args) == 3:
        return P(P.I + T(list(args)))
    if len(args) == 6:
        x, y, z, rx, ry, rz = args
        # Build per-axis quaternions individually so the magnitudes are exactly
        # the per-axis angles (NOT a rotation vector whose magnitude is a mixed
        # angle). Composition order: Q = qx * qy * qz == intrinsic XYZ.
        qx = Vec3([rx, 0, 0]).q if rx else Q.I
        qy = Vec3([0, ry, 0]).q if ry else Q.I
        qz = Vec3([0, 0, rz]).q if rz else Q.I
        return P(P.I + T([x, y, z]) + qx + qy + qz)
    if len(args) == 7:
        return P(list(args))
    raise ValueError(
        "Target pose must have 3 (xyz), 6 (xyz + Euler XYZ radians) "
        "or 7 (qw qx qy qz x y z) values, got %s" % args
    )


def SolveAndScoreIk(env, robot, manip, targetPose, target=None):
    """Find best IK solution at targetPose. Returns (best_solution, all_solutions).

    target: optional KinBody (e.g. the box being grasped) whose collisions with
    the gripper are ignored, so a grasp pose touching the object stays feasible.
    """
    RaveSetDebugLevel(DebugLevel.Error)
    with env:
        robot.Enable(True)
        solutions = FindIkSolutions(manip, targetPose, target=target)

    bestScore = -1.0
    bestSolution = None
    armJoints = [robot.GetJointFromDOFIndex(i) for i in manip.GetArmIndices()]
    for solution, constraints in sorted(solutions, key=lambda p: not bool(p[1])):
        if constraints:
            continue
        score = ComputeJointValuesScore(solution, armJoints)
        if score > bestScore:
            bestScore = score
            bestSolution = solution
    return bestSolution, solutions


def PlanTrajectory(env, robot, manip, startDof, goalPose, maxvelmult=0.2):
    """Plan a trajectory from startDof to goalPose using BiRRT."""
    robot.SetActiveManipulator(manip)
    robot.SetActiveDOFs(manip.GetArmIndices())
    # robot.SetDOFValues(startDof, manip.GetArmIndices())

    basemanip = interfaces.BaseManipulation(robot, maxvelmult=maxvelmult)
    try:
        # Let MoveToHandPosition apply its default parabolic smoother so the
        # returned trajectory is time-parametrized and densified. Passing
        # postprocessing=None returns the raw BiRRT vertex chain whose
        # adjacent waypoints can differ by tens of degrees per joint.
        traj = basemanip.MoveToHandPosition(
            matrices=[goalPose],
            maxiter=10000,
            maxtries=1,
            seedik=40,
            jitter=0,
            execute=False,
            outputtrajobj=True,
        )
        return traj
    except PlanningError as e:
        log.warning("BiRRT failed for manip %s: %s", manip.GetName(), e)
        return None


def ReplayTrajectory(robot, traj, dt=0.01):
    """Sample the (time-parametrized) trajectory at dt intervals and replay.

    parabolicsmoother2 stores a smoothed path as a handful of parabolic ramps
    (often 3-5 waypoints total); Sample() interpolates inside the ramps, so
    the smoothness comes from how finely we sample, not from waypoint count.
    """
    if traj is None:
        return
    spec = traj.GetConfigurationSpecification()
    jointValuesGroup = spec.FindCompatibleGroup("joint_values", False)
    _, _, dofStrs = jointValuesGroup.name.split(" ", 2)
    dofIndices = [int(i) for i in dofStrs.split(" ")]
    startIdx = jointValuesGroup.offset
    endIdx = startIdx + jointValuesGroup.dof
    duration = traj.GetDuration()
    numWaypoints = traj.GetNumWaypoints()

    if duration <= 0:
        log.warning("Replay: no timing, stepping %d raw waypoints", numWaypoints)
        for i in range(numWaypoints):
            dofValues = traj.GetWaypoint(i, jointValuesGroup)
            robot.SetDOFValues(list(dofValues), dofIndices)
            time.sleep(0.05)
        return

    steps = max(2, int(duration / dt))
    log.warning("Replay: duration=%.3fs, waypoints=%d, samples=%d (dt=%.3fs)",
                duration, numWaypoints, steps + 1, dt)

    prev = None
    maxDelta = 0.0
    for i in range(steps + 1):
        t = duration * i / steps
        dofValues = list(traj.Sample(t, spec)[startIdx:endIdx])
        if prev is not None:
            delta = max(abs(a - b) for a, b in zip(dofValues, prev))
            maxDelta = max(maxDelta, delta)
        prev = dofValues
        robot.SetDOFValues(dofValues, dofIndices)
        time.sleep(dt)
    log.warning("Replay: max per-sample joint delta = %.4f rad (%.2f deg)",
                maxDelta, np.rad2deg(maxDelta))


def DrawTarget(env, pose, handles):
    handles.append(misc.DrawAxes(env, pose, dist=0.2, linewidth=4))


# --------------------------------------------------------------------------- #
# Box management
# --------------------------------------------------------------------------- #
def AddBox(env, name="box0", halfExtents=(0.10, 0.06, 0.18), pose=None,
           color=(0.85, 0.35, 0.10)):
    """Add (or replace) a box KinBody in the environment.

    halfExtents: (hx, hy, hz) in meters.
    pose:        7-element [qw, qx, qy, qz, x, y, z]; default places the box in
                 front of the robot at a graspable height.
    Returns the KinBody. Call from the IPython session to populate the scene.
    """
    old = env.GetKinBody(name)
    if old is not None:
        env.Remove(old)

    body = RaveCreateKinBody(env, "")
    body.InitFromBoxes(
        np.array([[0.0, 0.0, 0.0, halfExtents[0], halfExtents[1], halfExtents[2]]]),
        True,
    )
    body.SetName(name)
    env.Add(body)

    if pose is None:
        pose = [1.0, 0.0, 0.0, 0.0, 0.45, 0.0, 1.0]
    body.SetTransform(np.array(pose, dtype=float))

    for link in body.GetLinks():
        for geom in link.GetGeometries():
            geom.SetDiffuseColor(np.array(color, dtype=float))

    log.warning("AddBox %r: halfExtents=%s, pose=%s", name, list(halfExtents), list(pose))
    return body


def DeleteBox(env, name="box0"):
    """Remove a box (or any KinBody) by name. Returns True if something was removed."""
    body = env.GetKinBody(name)
    if body is None:
        log.warning("DeleteBox: no body named %r", name)
        return False
    env.Remove(body)
    log.warning("DeleteBox: removed %r", name)
    return True


# --------------------------------------------------------------------------- #
# Grasp IK generation
# --------------------------------------------------------------------------- #
def _BoxFaceGrasps(box, gripperOffset=0.0):
    """Build one candidate grasp per box face.

    For each of the 6 faces: approach is along the face's inward normal (tool +Z
    points into the box), and the tool X axis is aligned with the shorter of the
    two in-plane edges (the direction a parallel gripper would close on).

    Returns a list of dicts with keys:
        axisIdx, sign, center (3,), normal (3, outward), edgeLens [e1, e2],
        graspWidth, pose (spatial.P, [qw qx qy qz x y z]).
    """
    T = np.array(box.GetTransform())
    R = T[:3, :3]
    c = T[:3, 3]
    geom = box.GetLinks()[0].GetGeometries()[0]
    he = np.array(geom.GetBoxExtents(), dtype=float)  # (hx, hy, hz)
    axes = [R[:, 0], R[:, 1], R[:, 2]]

    grasps = []
    for axisIdx in range(3):
        inplaneIdx = [i for i in range(3) if i != axisIdx]
        for sign in (+1.0, -1.0):
            outward = sign * axes[axisIdx]
            faceCenter = c + outward * he[axisIdx]
            approach = -outward  # tool +Z points into the box

            # in-plane edges (full lengths) and axes
            edges = [(i, axes[i], 2.0 * he[i]) for i in inplaneIdx]
            edges.sort(key=lambda e: e[2])  # shorter edge first
            shortAxis = edges[0][1]

            tz = approach / np.linalg.norm(approach)
            tx = shortAxis - tz * np.dot(shortAxis, tz)  # orthogonalize (already ⊥)
            tx = tx / np.linalg.norm(tx)
            ty = np.cross(tz, tx)
            Rtool = np.column_stack([tx, ty, tz])
            quat = quatFromRotationMatrix(Rtool)

            target = faceCenter + outward * gripperOffset
            pose = P([float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]),
                      float(target[0]), float(target[1]), float(target[2])])

            grasps.append({
                "axisIdx": axisIdx,
                "sign": sign,
                "center": faceCenter,
                "normal": outward,
                "edgeLens": [edges[0][2], edges[1][2]],
                "graspWidth": edges[0][2],
                "pose": pose,
            })
    return grasps


def GenerateBoxIK(env, robot, box, manipName, gripperOffset=0.0,
                  visualize=True, handles=None):
    """Compute and visualize grasp IK candidates for a box.

    Faces are prioritized: the short-side faces (perpendicular to the box's
    longest axis) first, then by proximity to the arm base. Each candidate gets
    an IK solve (gripper-vs-box collisions ignored). Feasible grasps are drawn
    with green axes, infeasible with red. Returns the candidate list sorted by
    priority, each annotated with a "solution" key (None if unreachable).
    """
    manip = PrepareManipulator(robot, manipName)
    grasps = _BoxFaceGrasps(box, gripperOffset)

    geom = box.GetLinks()[0].GetGeometries()[0]
    he = np.array(geom.GetBoxExtents(), dtype=float)
    longestAxis = int(np.argmax(he))
    armBase = np.array(manip.GetBase().GetTransform()[:3, 3])

    for g in grasps:
        g["isShortFace"] = (g["axisIdx"] == longestAxis)
        g["distToArm"] = float(np.linalg.norm(g["center"] - armBase))
    # short-side face first, then nearest to the arm base
    grasps.sort(key=lambda g: (not g["isShortFace"], g["distToArm"]))

    for g in grasps:
        bestSolution, _ = SolveAndScoreIk(env, robot, manip, g["pose"], target=box)
        g["solution"] = bestSolution
        feasible = bestSolution is not None
        if visualize and handles is not None:
            colormode = "rgb" if feasible else None
            h = misc.DrawAxes(env, g["pose"], dist=0.08, linewidth=3)
            handles.append(h)
        log.warning("  face axis=%d sign=%+d shortFace=%s dist=%.3f width=%.3f -> %s",
                    g["axisIdx"], int(g["sign"]), g["isShortFace"], g["distToArm"],
                    g["graspWidth"], "OK" if feasible else "unreachable")

    feasible = [g for g in grasps if g["solution"] is not None]
    log.warning("GenerateBoxIK: %d/%d feasible grasp(s) for %s on %r",
                len(feasible), len(grasps), manipName, box.GetName())
    return grasps


# --------------------------------------------------------------------------- #
# Grasp trajectory
# --------------------------------------------------------------------------- #
def Grab(env, robot, manipName, box, gripperOffset=0.0,
         maxvelmult=0.2, replayDt=0.01, handles=None, attach=True):
    """Select a hand + box, plan and visualize a grasp trajectory.

    Picks the highest-priority reachable grasp from GenerateBoxIK, plans a path
    from the arm's current configuration to that grasp (box collisions disabled
    during planning so the TCP can coincide with the box surface), replays it,
    and optionally Grab()s the box so it follows the gripper afterwards.
    """
    if handles is None:
        handles = []
    manip = PrepareManipulator(robot, manipName)
    grasps = GenerateBoxIK(env, robot, box, manipName, gripperOffset=gripperOffset,
                           visualize=True, handles=handles)
    feasible = [g for g in grasps if g["solution"] is not None]
    if not feasible:
        log.error("Grab: no feasible grasp for %r with %s", box.GetName(), manipName)
        return False

    best = feasible[0]
    log.warning("Grab: chosen face axis=%d sign=%+d center=%s",
                best["axisIdx"], int(best["sign"]), list(best["center"]))
    DrawTarget(env, best["pose"], handles)

    boxWasEnabled = box.IsEnabled()
    box.Enable(False)  # let the TCP reach the box surface during planning
    try:
        startDof = robot.GetDOFValues(manip.GetArmIndices())
        traj = PlanTrajectory(env, robot, manip, startDof, best["pose"],
                              maxvelmult=maxvelmult)
    finally:
        box.Enable(boxWasEnabled)

    if traj is None:
        log.error("Grab: trajectory planning failed")
        return False

    log.warning("Grab: trajectory duration=%.3fs", traj.GetDuration())
    ReplayTrajectory(robot, traj, dt=replayDt)

    if attach:
        robot.SetActiveManipulator(manipName)
        robot.Grab(box)
        log.warning("Grab: %s now holding %r", manipName, box.GetName())
    return True


def ReleaseBox(env, robot, box=None):
    """Release a grabbed box so it stops following the gripper.

    box: a KinBody to release. If None, releases everything the robot holds.
    Returns the list of released body names.
    """
    if box is None:
        released = [b.GetName() for b in robot.GetGrabbed()]
        robot.ReleaseAllGrabbed()
        log.warning("ReleaseBox: released all grabbed %s", released)
        return released

    if box not in robot.GetGrabbed():
        log.warning("ReleaseBox: robot is not holding %r", box.GetName())
        return []
    robot.Release(box)
    log.warning("ReleaseBox: released %r", box.GetName())
    return [box.GetName()]


def RunArm(env, robot, manipName, targetPose, label, handles,
           maxvelmult=0.2, replayDt=0.01):
    log.warning("=== %s arm: target = %s ===", label, list(targetPose))
    DrawTarget(env, targetPose, handles)
    manip = PrepareManipulator(robot, manipName)
    bestSolution, solutions = SolveAndScoreIk(env, robot, manip, targetPose)
    if bestSolution is None:
        log.error("%s arm: no collision-free IK found at %s", label, list(targetPose))
        for sol, constraints in solutions:
            hasJointLimits = any(isinstance(c, IkFailureJointLimits) for c in constraints)
            tag = "  [JL]" if hasJointLimits else ""
            log.warning("  candidate dof=%s constraints=%s%s",
                        ["%7.4f" % v for v in sol], constraints, tag)
        return False

    log.warning("%s arm IK solution: %s", label, ["%7.4f" % v for v in bestSolution])
    # show the IK pose briefly
    # with env:
    #     robot.SetDOFValues(bestSolution, manip.GetArmIndices())
    time.sleep(0.5)

    # Plan from zero pose
    # startDof = [0.0] * len(manip.GetArmIndices())
    # startDof = robot.GetDOFValues()
    startDof = p0 if manipName == LEFT_MANIP else [0.0] * len(manip.GetArmIndices())
    traj = PlanTrajectory(env, robot, manip, startDof, targetPose, maxvelmult=maxvelmult)
    if traj is None:
        log.error("%s arm: BiRRT trajectory planning failed", label)
        return False
    log.warning("%s arm trajectory duration: %.3fs", label, traj.GetDuration())
    ReplayTrajectory(robot, traj, dt=replayDt)
    return True


@openravepy.with_destroy
def Main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--robotFile",
        default="A2D_Omnipicker/A2D.kinbody.xml",
        help="path to A2D robot XML",
    )
    parser.add_argument(
        "--leftTarget",
        nargs="+",
        type=float,
        # default=[0.40, 0.55, 1.05, 0, 0, 0],
        default=m0,
        help="left-arm target: 3 (xyz), 6 (xyz + Euler XYZ radians, intrinsic) or 7 (qw qx qy qz x y z)",
    )
    parser.add_argument(
        "--rightTarget",
        nargs="+",
        type=float,
        default=[0.40, -0.55, 1.05, 0, 0, 0],
        help="right-arm target: same format as --leftTarget",
    )
    parser.add_argument("--logLevel", default="WARNING")
    parser.add_argument(
        "--maxvelmult", type=float, default=0.2,
        help="velocity multiplier passed to BaseManipulation; lower → slower, more visible motion. "
             "A2D's kinbody declares 572.958 deg/s per joint (10 rad/s), so the default 0.2 keeps the "
             "smoothed trajectory long enough to actually see.",
    )
    parser.add_argument(
        "--replayDt", type=float, default=0.01,
        help="replay sample interval in seconds; the smoothed trajectory only has a few waypoints, "
             "so smoothness comes from how densely we Sample() inside the ramps.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.logLevel.upper()))
    RaveSetDebugLevel(DebugLevel.Error)

    env = InitEnvironment(args.robotFile)
    robot = GetRobot(env)
    log.warning("Loaded robot %s with %d manipulators", robot.GetName(), len(robot.GetManipulators()))
    for m in robot.GetManipulators():
        log.warning("  manip: %s (armIndices=%s)", m.GetName(), list(m.GetArmIndices()))

    leftPose = BuildTargetPose(args.leftTarget)
    rightPose = BuildTargetPose(args.rightTarget)

    handles = []
    leftOk = RunArm(env, robot, LEFT_MANIP, leftPose, "left", handles,
                    maxvelmult=args.maxvelmult, replayDt=args.replayDt)
    rightOk = RunArm(env, robot, RIGHT_MANIP, rightPose, "right", handles,
                     maxvelmult=args.maxvelmult, replayDt=args.replayDt)

    if leftOk and rightOk:
        print("\033[32;1mBoth arms reached their targets.\033[0m")
    else:
        print("\033[31;1mOne or more arms failed (see logs above).\033[0m")

    # IPython helpers bound to this env/robot so the user can build a scene and
    # try grasps interactively without re-passing env/robot every time.
    def add_box(name="box0", halfExtents=(0.10, 0.06, 0.18), pose=None,
                color=(0.85, 0.35, 0.10)):
        return AddBox(env, name, halfExtents, pose, color)

    def del_box(name="box0"):
        return DeleteBox(env, name)

    def gen_box_ik(name="box0", manip=LEFT_MANIP, gripperOffset=0.0):
        body = env.GetKinBody(name)
        if body is None:
            print("No box named %r; call add_box(%r) first" % (name, name))
            return None
        return GenerateBoxIK(env, robot, body, manip, gripperOffset=gripperOffset,
                             handles=handles)

    def grab(name="box0", manip=LEFT_MANIP, gripperOffset=0.0):
        body = env.GetKinBody(name)
        if body is None:
            print("No box named %r; call add_box(%r) first" % (name, name))
            return False
        return Grab(env, robot, manip, body, gripperOffset=gripperOffset,
                    maxvelmult=args.maxvelmult, replayDt=args.replayDt, handles=handles)

    def release(name=None):
        # name=None releases everything the robot holds.
        return ReleaseBox(env, robot, env.GetKinBody(name) if name else None)

    print("\nIPython helpers (env/robot already bound):")
    print("  add_box(name='box0', halfExtents=(hx,hy,hz), pose=[qw,qx,qy,qz,x,y,z])")
    print("  del_box(name='box0')")
    print("  gen_box_ik(name='box0', manip='gripper_center')   # visualize grasp IK")
    print("  grab(name='box0', manip='gripper_center')         # plan+replay grasp")
    print("  release(name='box0')   # release a held box; release() releases all")
    print("  manips: LEFT='%s', RIGHT='%s'" % (LEFT_MANIP, RIGHT_MANIP))

    from IPython import embed
    embed()

    try:
        input("Press Enter to exit and close the viewer...")
    except (EOFError, KeyboardInterrupt):
        pass


if __name__ == "__main__":
    Main()
