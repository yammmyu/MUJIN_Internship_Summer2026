"""
Stream an RGB image and joint state over TCP, display the image inside an
OpenXR HMD, render the live left/right controller positions as cubes (with
their colour and size encoding the grab/trigger state), and publish the live
HMD + controller 6DoF poses back to the server.

Modelled on ``xr_examples/hello_xr/main.py`` but built on the higher-level
``xr.utils.gl.ContextObject`` wrapper so the example stays focused on the
networking and pose-capture pieces.

* Connects to ``REMOTE_HOST:REMOTE_PORT`` to receive a multiplexed stream of
  JPEG-encoded RGB frames and joint-state JSON messages.
* In a separate thread, pushes the latest HMD + controller pose (21 floats:
  ``[head_p xyz, head_q xyzw, left_p xyz, left_q xyzw, right_p xyz, right_q xyzw]``)
  plus two trigger bools to ``REMOTE_HOST:REMOTE_PORT2``.

Defaults come from the ``REMOTE_HOST`` / ``REMOTE_PORT`` / ``REMOTE_PORT2``
environment variables and can be overridden on the command line.
"""

import argparse
import logging
import os
import signal
import sys
from typing import List

import xr
from xr.utils.gl import ContextObject
from xr.utils.gl.glfw_util import GLFWOffscreenContextProvider

from pico_vr.pico_vr_common.protocol import JointStateBuffer

from .cube_renderer import CubeInstance, CubeRenderer
from .image_layer import ImageLayer
from .pose_input import PoseInput, PoseSample
from .tcp_stream import DownstreamReceiver, UpstreamSender

logger = logging.getLogger("xr_examples.pico_vr_client.main")

# Hand cube appearance. The cube uses hello_xr's per-face RGB scheme so its
# orientation is legible; grab state is encoded by shrinking it, mirroring the
# grip-driven scaling in hello_xr.
_CUBE_SIZE_OPEN_M = 0.08    # 8 cm cube when relaxed
_CUBE_SIZE_GRAB_M = 0.045   # shrinks to ~4.5 cm when squeezed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host",
        default=os.environ.get("REMOTE_HOST", "127.0.0.1"),
        help="Remote host (default: $REMOTE_HOST or 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("REMOTE_PORT", "5555")),
        help="Downstream port; image+joint stream (default: $REMOTE_PORT or 5555)",
    )
    parser.add_argument(
        "--port2",
        type=int,
        default=int(os.environ.get("REMOTE_PORT2", "5556")),
        help="Upstream port; HMD + controller pose (default: $REMOTE_PORT2 or 5556)",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args()


def _build_hand_cubes(snap: PoseSample) -> List[CubeInstance]:
    """Translate a PoseSample into the cube instances the renderer draws.

    Left/right pose assignments are swapped at render time so the cubes appear
    on the side that matches the operator's view. The wire protocol is
    unaffected — snap.to_joint_state() still emits unswapped left/right poses.
    """
    cubes: List[CubeInstance] = []
    # Left cube driven by the right-hand pose, and vice versa.
    if snap.right_tracked:
        cubes.append(CubeInstance(
            pose=snap.right,
            size=_CUBE_SIZE_GRAB_M if snap.right_grab else _CUBE_SIZE_OPEN_M,
        ))
    if snap.left_tracked:
        cubes.append(CubeInstance(
            pose=snap.left,
            size=_CUBE_SIZE_GRAB_M if snap.left_grab else _CUBE_SIZE_OPEN_M,
        ))
    return cubes


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        format="%(asctime)s.%(msecs)03d %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
        level=logging.DEBUG if args.verbose else logging.INFO,
    )
    logger.info(
        f"Client: downstream tcp://{args.host}:{args.port}, "
        f"upstream tcp://{args.host}:{args.port2}"
    )

    incoming_joints = JointStateBuffer()
    outgoing_joints = JointStateBuffer()
    downstream = DownstreamReceiver(args.host, args.port, incoming_joints)
    upstream = UpstreamSender(args.host, args.port2, outgoing_joints)
    downstream.start()
    upstream.start()

    stop_requested = {"flag": False}

    def _handle_sigint(_signum, _frame):
        logger.info("Interrupt received, shutting down")
        stop_requested["flag"] = True

    signal.signal(signal.SIGINT, _handle_sigint)

    try:
        with ContextObject(
            instance_create_info=xr.InstanceCreateInfo(
                application_info=xr.ApplicationInfo(
                    application_name="pyopenxr_pico_vr_client",
                    application_version=1,
                    engine_name="pyopenxr",
                    engine_version=1,
                    api_version=xr.XR_API_VERSION_1_0,
                ),
                enabled_extension_names=[xr.KHR_OPENGL_ENABLE_EXTENSION_NAME],
            ),
            # LOCAL (not VIEW) so xr.locate_views() returns the actual head pose
            # in world coordinates each frame. The cube renderer needs that
            # for its view matrix to be correct; with VIEW the view pose is
            # near-zero and cubes positioned in LOCAL space end up out of FOV.
            reference_space_create_info=xr.ReferenceSpaceCreateInfo(
                reference_space_type=xr.ReferenceSpaceType.LOCAL,
            ),
            context_provider=GLFWOffscreenContextProvider(),
        ) as context:
            # Register the pose + grab actions on the action set ContextObject
            # already created. This MUST happen before context.frame_loop() runs,
            # since frame_loop calls xrAttachSessionActionSets and the spec
            # forbids adding actions after attachment.
            pose_input = PoseInput(
                instance=context.instance,
                session=context.session,
                action_set=context.default_action_set,
            )
            pose_input.initialize()

            layer = ImageLayer()
            layer.initialize()
            cube_renderer = CubeRenderer()
            cube_renderer.initialize()

            views_seen = [False]
            hand_cubes: List[CubeInstance] = []
            try:
                for frame_state in context.frame_loop():
                    if stop_requested["flag"]:
                        break
                    if not frame_state.should_render:
                        logger.debug("Runtime requested skipped render this frame")

                    # Sample HMD + controller poses + grab state.
                    if pose_input.sync():
                        snap = pose_input.sample(frame_state.predicted_display_time)
                        outgoing_joints.set(snap.to_joint_state())
                        # hand_cubes = _build_hand_cubes(snap)

                    # Pull the latest image and refresh the texture.
                    frame, frame_id = downstream.latest_frame()
                    if frame is not None:
                        layer.update_texture(frame, frame_id)

                    rendered = 0
                    for view in context.view_loop(frame_state):
                        # Image first (clears depth, doesn't write it). It is
                        # positioned as a 3D quad at the FOV-filling position
                        # in view space, so reprojection keeps it head-stable.
                        layer.draw(view)
                        # Cubes after, world-space at the controller poses.
                        cube_renderer.draw(view, hand_cubes)
                        rendered += 1
                    if rendered > 0 and not views_seen[0]:
                        views_seen[0] = True
                        logger.info(f"First frame rendered to {rendered} eye(s)")
            finally:
                cube_renderer.destroy()
                layer.destroy()
                pose_input.destroy()
    finally:
        upstream.stop()
        downstream.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
