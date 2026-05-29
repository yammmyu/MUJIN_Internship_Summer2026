"""PlanningServer: glues a2d_reachability (OpenRAVE + IK + Grab) and detection (RGB-D).

Responsibilities:
  1. Own a single OpenRAVE Environment loaded with the A2D humanoid, with
     IK models prepared for both arms (delegated to a2d_reachability).
  2. Accept a single RGB + Depth pair (local files for now), run the
     `detection.Detector`, and spawn each detected object as a KinBody at
     the correct world pose. The camera is anchored to the A2D's head via
     the `head_camera` geom in `A2D.kinbody.xml`.
  3. Expose the existing `Grab` / `GenerateBoxIK` / `ReleaseBox` flow so a
     detected object can be picked up and the trajectory visualized.

The Detector's `orientedBox3D` is reported in its "output frame" (camera at
origin, +X depth/forward, +Y image-left, +Z up). We compose that with the
head-camera world transform to get the box's world pose, then hand it to
`a2d_reachability.AddBox`.
"""
from __future__ import annotations

import base64
import contextlib
import logging
import os
import sys
import threading
import time
import uuid
from typing import Optional

import numpy as np

from a2d_reachability.__main__ import (
    AddBox,
    DeleteBox,
    DrawTarget,
    GenerateBoxIK,
    GetRobot,
    Grab,
    InitEnvironment,
    LEFT_MANIP,
    PlanTrajectory,
    PrepareManipulator,
    ReleaseBox,
    ReplayTrajectory,
    RIGHT_MANIP,
)

from .tcp_server import TcpServer

log = logging.getLogger(__name__)


def _ManipToSide(manipName: str) -> str:
    if manipName == LEFT_MANIP:
        return "left"
    if manipName == RIGHT_MANIP:
        return "right"
    raise ValueError("Unknown manipulator: %r" % manipName)


def _SideToManip(side: str) -> str:
    if side == "left":
        return LEFT_MANIP
    if side == "right":
        return RIGHT_MANIP
    raise ValueError("Invalid side %r" % side)


def _RotationMatrixToQuat(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> (w, x, y, z) unit quaternion."""
    R = np.asarray(R, dtype=float)
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=float)
    return q / np.linalg.norm(q)


def _QuatMultiply(qa: np.ndarray, qb: np.ndarray) -> np.ndarray:
    """Hamilton product of two (w,x,y,z) quaternions."""
    w1, x1, y1, z1 = qa
    w2, x2, y2, z2 = qb
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


class PlanningServer:
    """A2D planning context: env + vision + grasp.

    Typical use (CLI / IPython):
        s = PlanningServer()
        s.ProcessVision("rgb.png", "depth.png", fx=600, fy=600, cx=320, cy=240,
                        depthScale=0.001)
        s.Grab("det_0", manip="gripper_center")
    """

    # head_camera geom translation in link_pitch_head's local frame
    # (matches A2D.kinbody.xml).
    HEAD_LINK_NAME = "link_pitch_head"
    HEAD_CAMERA_LOCAL_TRANSLATION = (-0.11, 0.04, 0.0)

    def __init__(
        self,
        robotFile: str = "A2D_Omnipicker/A2D.kinbody.xml",
        prepareManipulators: bool = True,
        logLevel: str = "WARNING",
    ):
        logging.basicConfig(level=getattr(logging, logLevel.upper(), logging.WARNING))
        self.env = InitEnvironment(robotFile)
        self.robot = GetRobot(self.env)
        self.handles: list = []
        # name -> {"body": KinBody, "detection": DetectionResult}
        self.objects: dict = {}

        # ---- TCP-server side state ----
        self._tcp: Optional[TcpServer] = None
        # Toggle for `on_state` to apply DOF updates. Held False during grasp
        # planning/visualization so the OpenRAVE viewer reflects the planned
        # path, not the live robot.
        self._live_state_enabled = True
        # Cached info from the most recent state message (for IPython inspect)
        self.last_client_state: dict = {}
        # request_id -> threading.Event/payload for synchronous waits
        self._detection_waiters: dict = {}
        self._trajectory_waiters: dict = {}
        # Latest planned (but not yet sent) trajectory
        # {"side", "path"[N][7], "delta_time", "manip", "target_pose"}
        self.pending_trajectory: Optional[dict] = None

        log.warning(
            "PlanningServer ready: robot=%s, manipulators=%s",
            self.robot.GetName(),
            [m.GetName() for m in self.robot.GetManipulators()],
        )

        if prepareManipulators:
            # Trigger IK model loading for both arms up front so the first
            # Grab call doesn't pay the (possibly multi-minute) ikfast build.
            for manip in (LEFT_MANIP, RIGHT_MANIP):
                try:
                    PrepareManipulator(self.robot, manip)
                except Exception as exc:
                    log.warning("PrepareManipulator(%s) failed: %s", manip, exc)

    # ------------------------------------------------------------------ #
    # Head camera <-> world
    # ------------------------------------------------------------------ #
    def GetCameraWorldTransform(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns (position3, R3x3) for the head camera in the OpenRAVE world.

        Position is the world location of the `head_camera` geom from the XML.
        R is the rotation mapping points expressed in the Detector's output
        frame (+X depth/forward, +Y image-left, +Z up) into the world frame.

        Assumption: the camera looks along the robot's world +X axis with
        image-up = world +Z, image-left = world +Y. This matches the Detector's
        output-frame axes exactly, so R defaults to identity. Override
        `cameraYaw/Pitch/RollDeg` in `ProcessVision` to compensate a tilted
        head; the Detector itself handles the rotation before returning poses.
        """
        link = self.robot.GetLink(self.HEAD_LINK_NAME)
        if link is None:
            raise RuntimeError(
                "Robot has no link named %r; expected the A2D head"
                % self.HEAD_LINK_NAME
            )
        headT = np.array(link.GetTransform(), dtype=float)
        offset = np.array(self.HEAD_CAMERA_LOCAL_TRANSLATION, dtype=float)
        cameraWorldPos = headT[:3, :3] @ offset + headT[:3, 3]
        R_world_camera = np.eye(3)
        return cameraWorldPos, R_world_camera

    # ------------------------------------------------------------------ #
    # Vision input
    # ------------------------------------------------------------------ #
    def ProcessVision(
        self,
        rgbPath: str,
        depthPath: str,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        depthScale: float = 0.001,
        cameraYawDeg: float = 0.0,
        cameraPitchDeg: float = 0.0,
        cameraRollDeg: float = 0.0,
        minConfidenceThreshold: float = 0.5,
        namePrefix: str = "det",
        color: tuple = (0.85, 0.35, 0.10),
        clearPrevious: bool = True,
    ) -> list[dict]:
        """Detect objects from a single RGB+Depth file pair and spawn them in env.

        Args:
            rgbPath:   path to an RGB image (anything cv2.imread can read).
            depthPath: path to depth (16-bit PNG or .npy of metric depth).
            fx,fy,cx,cy: pinhole intrinsics in pixels.
            depthScale: factor to multiply raw depth values by to get meters.
            cameraYaw/Pitch/RollDeg: passed to the Detector to compensate the
                camera's mounting orientation so returned poses are in a
                gravity-leveled, robot-aligned frame (matching our world).
            minConfidenceThreshold: segmentation confidence cutoff.
            namePrefix: KinBody naming for spawned boxes ("det_0", "det_1", ...).
            color:     RGB color for spawned boxes.
            clearPrevious: if True, removes any previously-spawned detections
                before processing the new image.

        Returns: list of {"name", "body", "detection"} dicts.
        """
        import cv2  # type: ignore[import]

        bgr = cv2.imread(rgbPath, cv2.IMREAD_COLOR)
        if bgr is None:
            raise IOError("Failed to read RGB image: %r" % rgbPath)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        if depthPath.endswith(".npy"):
            depth = np.load(depthPath)
        else:
            depth = cv2.imread(depthPath, cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise IOError("Failed to read depth image: %r" % depthPath)

        return self.ProcessVisionFromArrays(
            rgb=rgb, depth=depth, fx=fx, fy=fy, cx=cx, cy=cy,
            depthScale=depthScale,
            cameraYawDeg=cameraYawDeg, cameraPitchDeg=cameraPitchDeg, cameraRollDeg=cameraRollDeg,
            minConfidenceThreshold=minConfidenceThreshold,
            namePrefix=namePrefix, color=color, clearPrevious=clearPrevious,
        )

    def ProcessVisionFromArrays(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        depthScale: float = 0.001,
        cameraYawDeg: float = 0.0,
        cameraPitchDeg: float = 0.0,
        cameraRollDeg: float = 0.0,
        minConfidenceThreshold: float = 0.5,
        namePrefix: str = "det",
        color: tuple = (0.85, 0.35, 0.10),
        clearPrevious: bool = True,
    ) -> list[dict]:
        """Detection-and-spawn core used by both file-based ProcessVision and
        the TCP detection_trigger path."""
        self._EnsureDetectionOnPath()
        # Lazy import: needs cv2 + (in production) the mujindetection package.
        from detector import CameraIntrinsics, Detector  # type: ignore[import]

        if clearPrevious:
            for name in list(self.objects.keys()):
                DeleteBox(self.env, name)
            self.objects.clear()

        intrinsics = CameraIntrinsics(
            fx=fx, fy=fy, cx=cx, cy=cy,
            width=rgb.shape[1], height=rgb.shape[0],
            depthScale=depthScale,
        )

        detector = Detector()
        results = detector.Detect(
            rgb,
            depth,
            intrinsics,
            minConfidenceThreshold=minConfidenceThreshold,
            cameraYawDeg=cameraYawDeg,
            cameraPitchDeg=cameraPitchDeg,
            cameraRollDeg=cameraRollDeg,
        )

        cameraWorldPos, R_world_camera = self.GetCameraWorldTransform()
        log.warning(
            "PlanningServer.ProcessVision: %d detection(s); camera at world (%.3f, %.3f, %.3f)",
            len(results), *cameraWorldPos,
        )

        # Quaternion for the world<-camera rotation (used to rotate orientations).
        qWorldCamera = _RotationMatrixToQuat(R_world_camera)

        added: list[dict] = []
        for i, det in enumerate(results):
            if det.orientedBox3D is None:
                log.warning("  [%d] %s: no orientedBox3D, skipping",
                            i, det.instance.name)
                continue
            ob = det.orientedBox3D

            worldPos = R_world_camera @ np.asarray(ob.position, dtype=float) + cameraWorldPos
            worldQuat = _QuatMultiply(qWorldCamera, np.asarray(ob.quaternion, dtype=float))
            halfExtents = tuple(float(s) / 2.0 for s in ob.size)

            pose = [
                float(worldQuat[0]), float(worldQuat[1]),
                float(worldQuat[2]), float(worldQuat[3]),
                float(worldPos[0]), float(worldPos[1]), float(worldPos[2]),
            ]
            name = "%s_%d" % (namePrefix, i)
            body = AddBox(self.env, name, halfExtents, pose, color)
            log.warning(
                "  [%d] %s -> %s at world (%.3f, %.3f, %.3f), size=%s, conf=%.2f",
                i, det.instance.name, name,
                worldPos[0], worldPos[1], worldPos[2],
                [round(float(s), 3) for s in ob.size],
                float(det.instance.score),
            )
            self.objects[name] = {"body": body, "detection": det}
            added.append({"name": name, "body": body, "detection": det})

        if not added:
            log.warning("PlanningServer.ProcessVision: no 3D-locatable detections")
        return added

    @staticmethod
    def _EnsureDetectionOnPath() -> None:
        """Add ../detection to sys.path so `detector.py` can `from segmenter import ...`."""
        here = os.path.dirname(os.path.realpath(__file__))
        detectionDir = os.path.normpath(os.path.join(here, "..", "detection"))
        if os.path.isdir(detectionDir) and detectionDir not in sys.path:
            sys.path.insert(0, detectionDir)

    # ------------------------------------------------------------------ #
    # Grasp pipeline (thin wrappers; the heavy lifting lives in a2d_reachability)
    # ------------------------------------------------------------------ #
    def GenerateBoxIK(self, name: str, manip: str = LEFT_MANIP, **kwargs):
        body = self._RequireBody(name)
        return GenerateBoxIK(self.env, self.robot, body, manip,
                             handles=self.handles, **kwargs)

    def Grab(self, name: str, manip: str = LEFT_MANIP, **kwargs):
        body = self._RequireBody(name)
        return Grab(self.env, self.robot, manip, body,
                    handles=self.handles, **kwargs)

    def Release(self, name: Optional[str] = None):
        body = self.env.GetKinBody(name) if name else None
        return ReleaseBox(self.env, self.robot, body)

    def AddBox(self, name: str = "box0",
               halfExtents=(0.06, 0.04, 0.10), pose=None,
               color=(0.85, 0.35, 0.10)):
        body = AddBox(self.env, name, halfExtents, pose, color)
        self.objects[name] = {"body": body, "detection": None}
        return body

    def DeleteBox(self, name: str = "box0"):
        ok = DeleteBox(self.env, name)
        self.objects.pop(name, None)
        return ok

    def _RequireBody(self, name: str):
        body = self.env.GetKinBody(name)
        if body is None:
            raise ValueError(
                "No KinBody named %r in env. Known objects: %s"
                % (name, list(self.objects.keys()))
            )
        return body

    # ------------------------------------------------------------------ #
    # TCP server: lifecycle
    # ------------------------------------------------------------------ #
    def StartTcpServer(self, host: str = "0.0.0.0", port: int = 9100) -> None:
        """Start listening for the GUI motion-planning client."""
        if self._tcp is not None:
            log.warning("TCP server already started")
            return
        self._tcp = TcpServer(controller=self, host=host, port=port)
        self._tcp.start()

    def StopTcpServer(self) -> None:
        if self._tcp is not None:
            self._tcp.stop()
            self._tcp = None

    @property
    def tcp_connected(self) -> bool:
        return self._tcp is not None and self._tcp.connected

    # ------------------------------------------------------------------ #
    # TCP server: inbound callbacks (invoked by TcpServer reader thread)
    # ------------------------------------------------------------------ #
    def on_connect(self, addr) -> None:
        log.warning("PlanningServer: client connected from %s", addr)

    def on_disconnect(self, addr) -> None:
        log.warning("PlanningServer: client %s disconnected", addr)

    def on_state(self, msg: dict) -> None:
        """1) Reflect the client's live arm/head/waist state in env."""
        self.last_client_state = msg
        if not self._live_state_enabled:
            return
        arm = msg.get("arm_joint_values")
        head = msg.get("head_joint_values")
        waist = msg.get("waist_joint_values")
        try:
            with self.env:
                self._ApplyArmJoints(arm)
                self._ApplyHeadJoints(head)
                self._ApplyWaistJoints(waist)
        except Exception as e:
            log.warning("on_state: SetDOFValues failed: %s", e)

    def on_detection_image(self, msg: dict) -> None:
        """2b) Resolve a pending TriggerDetection() call with the payload."""
        req_id = msg.get("request_id")
        waiter = self._detection_waiters.get(req_id)
        if waiter is None:
            log.warning("on_detection_image: no waiter for request_id=%r", req_id)
            return
        waiter["payload"] = msg
        waiter["event"].set()

    def on_trajectory_ack(self, msg: dict) -> None:
        traj_id = msg.get("trajectory_id")
        log.warning("trajectory_ack id=%s status=%s reason=%s",
                    traj_id, msg.get("status"), msg.get("reason"))
        waiter = self._trajectory_waiters.get(traj_id)
        if waiter is not None:
            waiter["ack"] = msg
            waiter["ack_event"].set()

    def on_trajectory_result(self, msg: dict) -> None:
        traj_id = msg.get("trajectory_id")
        log.warning("trajectory_result id=%s status=%s reason=%s",
                    traj_id, msg.get("status"), msg.get("reason"))
        waiter = self._trajectory_waiters.get(traj_id)
        if waiter is not None:
            waiter["result"] = msg
            waiter["result_event"].set()

    def on_unknown(self, msg: dict) -> None:
        log.warning("unknown TCP message type=%r", msg.get("type"))

    # ------------------------------------------------------------------ #
    # Live-state apply helpers
    # ------------------------------------------------------------------ #
    def _ApplyArmJoints(self, arm: Optional[list]) -> None:
        if not arm or len(arm) < 14:
            return
        leftManip = self.robot.GetManipulator(LEFT_MANIP)
        rightManip = self.robot.GetManipulator(RIGHT_MANIP)
        if leftManip is not None:
            li = list(leftManip.GetArmIndices())
            if len(li) == 7:
                self.robot.SetDOFValues(arm[:7], li)
        if rightManip is not None:
            ri = list(rightManip.GetArmIndices())
            if len(ri) == 7:
                self.robot.SetDOFValues(arm[7:14], ri)

    def _ApplyHeadJoints(self, head: Optional[list]) -> None:
        if not head:
            return
        # GUI sends [yaw_rad, pitch_rad]
        indices = self._JointIndicesByName(("joint-yaw_head", "joint-pitch_head"))
        if indices:
            n = min(len(indices), len(head))
            self.robot.SetDOFValues(head[:n], indices[:n])

    def _ApplyWaistJoints(self, waist: Optional[list]) -> None:
        if not waist:
            return
        # GUI sends [pitch_rad, height_cm]; convert height cm -> m for OpenRAVE prismatic
        indices = self._JointIndicesByName(("joint-pitch", "joint-up-down"))
        if not indices:
            return
        values = [waist[0] if len(waist) >= 1 else 0.0,
                  (waist[1] * 0.01) if len(waist) >= 2 else 0.0]
        n = min(len(indices), len(values))
        self.robot.SetDOFValues(values[:n], indices[:n])

    def _JointIndicesByName(self, names) -> list:
        out = []
        for n in names:
            j = self.robot.GetJoint(n)
            if j is None:
                return []
            out.append(j.GetDOFIndex())
        return out

    # ------------------------------------------------------------------ #
    # Live-state freeze (for grasp visualization)
    # ------------------------------------------------------------------ #
    def PauseLiveState(self) -> None:
        self._live_state_enabled = False

    def ResumeLiveState(self) -> None:
        self._live_state_enabled = True

    @contextlib.contextmanager
    def SuppressLiveState(self):
        was = self._live_state_enabled
        self._live_state_enabled = False
        try:
            yield
        finally:
            self._live_state_enabled = was

    # ------------------------------------------------------------------ #
    # 2) Detection trigger -> wait for RGB+Depth -> ProcessVision
    # ------------------------------------------------------------------ #
    def TriggerDetection(
        self,
        timeout: float = 15.0,
        fx: Optional[float] = None,
        fy: Optional[float] = None,
        cx: Optional[float] = None,
        cy: Optional[float] = None,
        depthScale: Optional[float] = None,
        cameraYawDeg: float = 0.0,
        cameraPitchDeg: float = 0.0,
        cameraRollDeg: float = 0.0,
        minConfidenceThreshold: float = 0.5,
        clearPrevious: bool = True,
    ) -> list[dict]:
        """Ask the GUI for the latest head RGB+Depth, then run detection.

        Intrinsics come from the message if the GUI included them; otherwise
        from the explicit kwargs. depthScale defaults to 0.001 (uint16 mm).
        """
        if not self.tcp_connected:
            raise RuntimeError("No GUI client connected to planning server")

        req_id = str(uuid.uuid4())
        event = threading.Event()
        self._detection_waiters[req_id] = {"event": event, "payload": None}
        try:
            sent = self._tcp.send_json({
                "type": "detection_trigger",
                "request_id": req_id,
            })
            if not sent:
                raise RuntimeError("Failed to send detection_trigger")
            if not event.wait(timeout=timeout):
                raise TimeoutError("Timed out waiting for detection_image")
            payload = self._detection_waiters[req_id]["payload"]
        finally:
            self._detection_waiters.pop(req_id, None)

        import cv2  # type: ignore[import]

        rgb_b64 = payload.get("rgb_jpeg_base64")
        depth_b64 = payload.get("depth_png_base64")
        if not rgb_b64 or not depth_b64:
            raise RuntimeError("detection_image missing rgb/depth")
        rgb_buf = np.frombuffer(base64.b64decode(rgb_b64), dtype=np.uint8)
        bgr = cv2.imdecode(rgb_buf, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError("Failed to decode RGB JPEG")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        depth_buf = np.frombuffer(base64.b64decode(depth_b64), dtype=np.uint8)
        depth = cv2.imdecode(depth_buf, cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise RuntimeError("Failed to decode depth PNG")

        intr = payload.get("rgb_intrinsics") or {}
        fx = fx if fx is not None else intr.get("fx")
        fy = fy if fy is not None else intr.get("fy")
        cx = cx if cx is not None else intr.get("cx")
        cy = cy if cy is not None else intr.get("cy")
        depthScale = depthScale if depthScale is not None else payload.get("depth_scale", 0.001)
        if None in (fx, fy, cx, cy):
            raise ValueError(
                "Intrinsics missing: client did not include rgb_intrinsics and"
                " caller did not pass fx/fy/cx/cy")

        log.warning(
            "TriggerDetection: got %dx%d RGB, %dx%d depth, intr=(fx=%.1f,fy=%.1f,cx=%.1f,cy=%.1f)",
            rgb.shape[1], rgb.shape[0], depth.shape[1], depth.shape[0], fx, fy, cx, cy,
        )
        return self.ProcessVisionFromArrays(
            rgb=rgb, depth=depth, fx=fx, fy=fy, cx=cx, cy=cy, depthScale=depthScale,
            cameraYawDeg=cameraYawDeg, cameraPitchDeg=cameraPitchDeg, cameraRollDeg=cameraRollDeg,
            minConfidenceThreshold=minConfidenceThreshold,
            clearPrevious=clearPrevious,
        )

    # ------------------------------------------------------------------ #
    # 3) Plan grasp trajectory (capture for later send), with live-state freeze
    # ------------------------------------------------------------------ #
    def PlanGrab(
        self,
        name: str,
        manip: str = LEFT_MANIP,
        gripperOffset: float = 0.0,
        tiltsDeg=(0, 2, 4, 6, 8, 10),
        tiltAzimuthsDeg=(0, 2, 4, 6, 8, 10),
        maxGraspWidth: float = 0.10,
        maxvelmult: float = 0.2,
        replayDt: float = 0.01,
        samplingDt: float = 0.02,
        visualize: bool = True,
    ) -> Optional[dict]:
        """Plan a grasp trajectory and cache it for SendTrajectory.

        Mimics a2d_reachability.Grab() but extracts the OpenRAVE trajectory and
        samples it as a list of [7-joint] waypoints. During planning + replay
        live-state updates from the GUI are suppressed so the viewer reflects
        the planned motion. The cached trajectory is kept in
        `self.pending_trajectory` until SendTrajectory() or DiscardTrajectory().
        """
        body = self._RequireBody(name)
        manipObj = PrepareManipulator(self.robot, manip)
        with self.SuppressLiveState():
            grasps = GenerateBoxIK(
                self.env, self.robot, body, manip,
                gripperOffset=gripperOffset,
                tiltsDeg=tiltsDeg, tiltAzimuthsDeg=tiltAzimuthsDeg,
                maxGraspWidth=maxGraspWidth,
                visualize=True, handles=self.handles,
            )
            feasible = [g for g in grasps if g["solution"] is not None]
            if not feasible:
                log.error("PlanGrab: no feasible grasp for %r with %s", name, manip)
                return None
            best = feasible[0]
            log.warning(
                "PlanGrab: chosen face axis=%d sign=%+d zRot=%d tilt=%+d@%d",
                best["axisIdx"], int(best["sign"]),
                best["zRotDeg"], best["tiltDeg"], best["tiltAzimDeg"],
            )
            DrawTarget(self.env, best["pose"], self.handles)

            boxWasEnabled = body.IsEnabled()
            body.Enable(True)
            try:
                startDof = self.robot.GetDOFValues(manipObj.GetArmIndices())
                traj = PlanTrajectory(
                    self.env, self.robot, manipObj, startDof,
                    best["pose"], maxvelmult=maxvelmult,
                )
            finally:
                body.Enable(boxWasEnabled)

            if traj is None:
                log.error("PlanGrab: BiRRT trajectory planning failed")
                return None

            path = self._SampleTrajectoryAsArmPath(traj, manipObj, dt=samplingDt)
            log.warning(
                "PlanGrab: traj duration=%.3fs, sampled %d waypoints @ dt=%.3fs",
                traj.GetDuration(), len(path), samplingDt,
            )

            if visualize:
                ReplayTrajectory(self.robot, traj, dt=replayDt)

        self.pending_trajectory = {
            "side": _ManipToSide(manip),
            "manip": manip,
            "target_pose": list(best["pose"]),
            "path": path,
            "delta_time": samplingDt,
            "object": name,
        }
        log.warning("PlanGrab: pending_trajectory ready (side=%s, %d points). "
                    "Call SendTrajectory() to dispatch to the GUI.",
                    self.pending_trajectory["side"], len(path))
        return self.pending_trajectory

    def DiscardTrajectory(self) -> None:
        """Drop the cached trajectory and re-enable live-state updates."""
        self.pending_trajectory = None
        self.ResumeLiveState()
        log.warning("DiscardTrajectory: cleared pending trajectory")

    def _SampleTrajectoryAsArmPath(self, traj, manip, dt: float = 0.02) -> list:
        """Sample an OpenRAVE traj as a list of 7-joint waypoints for `manip`."""
        spec = traj.GetConfigurationSpecification()
        arm_indices = list(manip.GetArmIndices())
        duration = float(traj.GetDuration())
        if duration <= 0.0:
            # Single configuration: just sample at t=0
            sample = traj.Sample(0.0)
            values = spec.ExtractJointValues(sample, self.robot, arm_indices)
            return [[float(v) for v in values]]
        n = max(2, int(duration / dt) + 1)
        path: list = []
        for i in range(n):
            t = min(i * dt, duration)
            sample = traj.Sample(t)
            values = spec.ExtractJointValues(sample, self.robot, arm_indices)
            path.append([float(v) for v in values])
        return path

    # ------------------------------------------------------------------ #
    # 4) Send trajectory to the GUI client
    # ------------------------------------------------------------------ #
    def SendTrajectory(
        self,
        delta_time: Optional[float] = None,
        wait_ack: bool = True,
        wait_result: bool = False,
        timeout: float = 30.0,
    ) -> Optional[dict]:
        """Dispatch the cached PlanGrab() trajectory to the GUI client.

        wait_ack:    block until trajectory_ack arrives (or timeout).
        wait_result: also block until trajectory_result arrives.
        On send, live-state updates are resumed (the planned motion is now
        about to happen on the real robot, so the env should follow live).
        """
        if self.pending_trajectory is None:
            raise RuntimeError("No pending trajectory. Call PlanGrab() first.")
        if not self.tcp_connected:
            raise RuntimeError("No GUI client connected")

        traj_id = str(uuid.uuid4())
        msg = {
            "type": "trajectory",
            "trajectory_id": traj_id,
            "side": self.pending_trajectory["side"],
            "path": self.pending_trajectory["path"],
            "delta_time": (delta_time
                           if delta_time is not None
                           else self.pending_trajectory["delta_time"]),
        }

        waiter = {
            "ack_event": threading.Event(),
            "result_event": threading.Event(),
            "ack": None,
            "result": None,
        }
        if wait_ack or wait_result:
            self._trajectory_waiters[traj_id] = waiter
        try:
            sent = self._tcp.send_json(msg)
            if not sent:
                raise RuntimeError("Failed to send trajectory")

            # Clear cache and resume live updates: from the GUI's perspective
            # the robot will start moving and our env should follow.
            self.pending_trajectory = None
            self.ResumeLiveState()

            if wait_ack:
                if not waiter["ack_event"].wait(timeout=timeout):
                    raise TimeoutError("Timed out waiting for trajectory_ack")
            if wait_result:
                if not waiter["result_event"].wait(timeout=timeout):
                    raise TimeoutError("Timed out waiting for trajectory_result")
        finally:
            self._trajectory_waiters.pop(traj_id, None)

        return {
            "trajectory_id": traj_id,
            "ack": waiter["ack"],
            "result": waiter["result"],
        }
