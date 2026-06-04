import dataclasses
import math
import threading
import time
import typing

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from pico_vr.pico_vr_server.server import _TRIGGER_LABELS
from pico_vr.pico_vr_common.protocol import JointState


class VRMixin:
    """VR 遥操：手柄 JointState 回调驱动机器人，以及相机画面 VR 串流。"""

    def _handle_vr_joints(self, state: JointState) -> None:
        @dataclasses.dataclass(frozen=True)
        class Position:
            px: float
            py: float
            pz: float
            yaw: float
            pitch: float
            roll: float

            def format(self) -> str:
                return (
                    f"pos=({self.px:+.3f},{self.py:+.3f},{self.pz:+.3f}) "
                    f"ypr=({math.degrees(self.yaw):+6.1f}°,{math.degrees(self.pitch):+6.1f}°,{math.degrees(self.roll):+6.1f}°)"
                )

            def get_action_delta(self, previous_position, pose_mult=1.0, qut_mult=0.5, max_pose_value=0.02, min_pos_value=0.0005, max_qut_value=0.02, min_qut_value=0.001) -> typing.Optional[list[float]]:
                assert 1.0 >= pose_mult > 0
                assert 0.5 >= qut_mult > 0
                action_delta = [self.px - previous_position.px, self.py - previous_position.py, self.pz - previous_position.pz, self.yaw - previous_position.yaw, self.pitch - previous_position.pitch, self.roll - previous_position.roll]
                for index in range(3):
                    action_delta[index] *= pose_mult
                    if abs(action_delta[index]) > max_pose_value:
                        action_delta[index] = max_pose_value * np.sign(action_delta[index])
                    elif abs(action_delta[index]) < min_pos_value:
                        action_delta[index] = 0.0
                for index in range(3, 6):
                    action_delta[index] *= qut_mult
                    if abs(action_delta[index]) > max_qut_value:
                        action_delta[index] = max_qut_value * np.sign(action_delta[index])
                    elif abs(action_delta[index]) < min_qut_value:
                        action_delta[index] = 0.0
                if any(abs(value) > 1e-5 for value in action_delta):
                    return action_delta
                return None

        def _format_pose(slab: list[float]) -> Position:

            def transform_vr_to_robot_full(px, py, pz, qx, qy, qz, qw):
                """
                完整转换VR手柄到机械臂坐标系（位置+姿态）

                坐标系映射:
                    X_robot = -Z_vr
                    Y_robot = -X_vr
                    Z_robot = Y_vr
                """
                # ========== 位置转换 ==========
                pos_robot = np.array([-pz, -px, py])

                # ========== 姿态转换 ==========
                # 定义坐标系转换矩阵
                # VR坐标系到机械臂坐标系的旋转矩阵
                R_vr_to_robot = np.array([
                    [0, 0, -1],  # 机械臂X轴 = -VR的Z轴
                    [-1, 0, 0],  # 机械臂Y轴 = -VR的X轴
                    [0, 1, 0]  # 机械臂Z轴 = VR的Y轴
                ])

                # VR手柄的四元数 → 旋转矩阵
                q_vr = [qx, qy, qz, qw]
                R_vr_hand = R.from_quat(q_vr).as_matrix()

                # 转换到机械臂坐标系下的旋转矩阵
                # R_robot_hand = R_vr_to_robot * R_vr_hand * R_vr_to_robot^T
                R_robot_hand = R_vr_to_robot @ R_vr_hand @ R_vr_to_robot.T

                # 旋转矩阵 → 四元数
                q_robot = R.from_matrix(R_robot_hand).as_quat()

                # TODO: 抓手与robot base坐标转换
                q_robot[1] *= -1

                return pos_robot, q_robot

            pos, quat = transform_vr_to_robot_full(*slab)
            # py, pz, -px, qx, qy, qz, qw = slab
            px, py, pz = pos
            qx, qy, qz, qw = quat
            sin_p = 2.0 * (qw * qx - qz * qy)
            if abs(sin_p) >= 1.0:
                pitch = math.copysign(math.pi / 2.0, sin_p)
            else:
                pitch = math.asin(sin_p)
            yaw = math.atan2(
                2.0 * (qw * qy + qx * qz),
                1.0 - 2.0 * (qx * qx + qy * qy),
            )
            roll = math.atan2(
                2.0 * (qw * qz + qx * qy),
                1.0 - 2.0 * (qz * qz + qx * qx),
            )
            return Position(px, py, pz, yaw, pitch, roll)

        def _fmt_buttons(triggers: list[bool]) -> list[str]:
            """Format face-button + thumbstick-click bits (triggers[2:8]). Empty if none set."""
            pressed = [
                _TRIGGER_LABELS[i]
                for i in range(2, min(len(triggers), len(_TRIGGER_LABELS)))
                if triggers[i]
            ]
            return pressed

        def _fmt_stick(axes: list[float], offset: int) -> tuple[float, float]:
            """Format one (x, y) thumbstick reading from axes[offset:offset+2]."""
            return axes[offset], axes[offset + 1]

        def _vr_auto_execution_thread() -> None:
            print("Starting vr execution thread...")
            while self.is_vr_control:
                now = time.time()
                if now - self.last_joint_update_timestamp > 0.2:
                    print("Stopping vr execution thread, due to VR connection time out: 200ms")
                    self.vr_actions = []
                    self.is_vr_control = False
                    break
                while self.vr_actions:
                    head, left_hand, right_hand, left_grab, right_grab = self.vr_actions.pop(0)
                    if head is not None:
                        head_yaw_pos = np.clip(math.radians(head.yaw), -1.0, 1.0)
                        head_pitch_pos = np.clip(math.radians(head.pitch), -0.5, 0.5)
                        self.robot.move_head([head_yaw_pos, head_pitch_pos])
                    robot_actions = {}
                    if left_hand is not None:
                        robot_actions["left_arm"] = {
                            "action_data": left_hand,
                            "control_type": "DELTA_POSE"
                        }
                    if right_hand is not None:
                        robot_actions["right_arm"] = {
                            "action_data": right_hand,
                            "control_type": "DELTA_POSE"
                        }
                    if left_grab is not None or right_grab is not None:
                        self.robot.move_gripper([self.left_gripper_pos, self.right_gripper_pos])
                    if robot_actions:
                        arm_states, _ = self.robot.arm_joint_states()
                        head_states, _ = self.robot.head_joint_states()
                        waist_states, _ = self.robot.waist_joint_states()
                        assert arm_states
                        assert head_states
                        assert waist_states

                        # 构建完整的机器人状态（参考SDK文档的格式）
                        robot_states = {
                            "head": head_states,
                            "waist": waist_states,
                            "arm": arm_states
                        }

                        # 使用轨迹跟踪控制
                        print(f"Dummy execution, robot_actions: {robot_actions}")
                        self.robot_controller.trajectory_tracking_control(
                            int(time.time() * 1e9),
                            robot_states,
                            [robot_actions],
                            "base_link",
                            0.001  # 较短的执行时间
                        )
                        time.sleep(0.001)
                time.sleep(0.02)
            print("Stopping vr execution thread...")

        if self.is_vr_control:
            if self.vr_execution_thread is None:
                self.vr_actions = []
                self.vr_execution_thread = threading.Thread(target=_vr_auto_execution_thread, daemon=True)
                self.vr_execution_thread.start()
        elif self.vr_execution_thread is not None:
            self.vr_actions = []
            self.vr_execution_thread.join()
            self.vr_execution_thread = None

        now = time.time()

        if self.is_vr_control and now - self.last_joint_update_timestamp > 0.2:
            print("Resetting vr execution thread, due to VR connection time out: 200ms")
            self.is_vr_control = False
            self.vr_actions = []
            self.previous_vr_positions = []

        self.last_joint_update_timestamp = now

        positions = state.positions
        triggers = state.triggers
        axes = state.axes
        assert len(triggers) >= 2
        left_grab = bool(triggers[0])
        right_grab = bool(triggers[1])
        buttons = _fmt_buttons(triggers)
        assert len(axes) == 4
        left_stick = _fmt_stick(axes, 0)
        right_stick = _fmt_stick(axes, 2)
        assert len(positions) == 21
        head_position = _format_pose(positions[0:7])
        left_hand_position = _format_pose(positions[7:14])
        right_hand_position = _format_pose(positions[14:21])

        # head, left, right, left gripper, right gripper
        action = [None, None, None, None, None]
        if self.previous_vr_positions and self.is_vr_control:
            if "L_Y" in buttons:
                action[1] = left_hand_position.get_action_delta(self.previous_vr_positions[1])
            if "R_B" in buttons:
                action[2] = right_hand_position.get_action_delta(self.previous_vr_positions[2])
            # TODO: gripper
            if left_grab and left_grab != self.previous_vr_positions[3]:
                self.left_gripper_pos = 0.0 if self.left_gripper_pos > 0.5 else 1.0
                action[3] = self.left_gripper_pos > 0.5
            if right_grab and right_grab != self.previous_vr_positions[4]:
                self.right_gripper_pos = 0.0 if self.right_gripper_pos > 0.5 else 1.0
                action[4] = self.right_gripper_pos > 0.5
        if any(act is not None for act in action):
            self.vr_actions.append(action)
        elif any(any(value is not None for value in act[:3]) for act in self.vr_actions):
            # clear all pending move action when buttons released
            self.vr_actions = []

        self.previous_vr_positions = [head_position, left_hand_position, right_hand_position, left_grab, right_grab]

        if not hasattr(self, 'vr_info'):
            self.vr_info = [now, 0]
        else:
            self.vr_info[-1] += 1
            if now - self.vr_info[0] > 1:
                print("now: ", now, "fps: ", self.vr_info[-1] / (now - self.vr_info[0]))
                self.vr_info = [now, 0]
            else:
                return

        print("=" * 10)
        print("head_position: ", head_position.format())
        print("left_hand_position: ", left_hand_position.format())
        print("right_hand_position: ", right_hand_position.format())
        print("buttons: ", buttons)
        print("left_grab: ", left_grab)
        print("right_grab: ", right_grab)
        print("left_stick: ", left_stick)
        print("right_stick: ", right_stick)
        print("=" * 10)
        # if len(positions) == 21:
        #     head = positions[0:7]
        #     left = positions[7:14]
        #     right = positions[14:21]
        #     logger.info(
        #         f"xr pose t={state.timestamp:.3f}  "
        #         f"{_format_pose('H', head)}  |  "
        #         f"{_format_pose('L', left)} grab={_fmt_trigger(left_grab)} stk={left_stick}  |  "
        #         f"{_format_pose('R', right)} grab={_fmt_trigger(right_grab)} stk={right_stick}  "
        #         f"btn={buttons}"
        #     )
        # elif len(positions) == 14:
        #     left = positions[0:7]
        #     right = positions[7:14]
        #     logger.info(
        #         f"controller pose t={state.timestamp:.3f}  "
        #         f"{_format_pose('L', left)} grab={_fmt_trigger(left_grab)} stk={left_stick}  |  "
        #         f"{_format_pose('R', right)} grab={_fmt_trigger(right_grab)} stk={right_stick}  "
        #         f"btn={buttons}"
        #     )
        # else:
        #     preview = ", ".join(f"{p:+.3f}" for p in positions[:6])
        #     logger.info(
        #         f"upstream joints t={state.timestamp:.3f} n={len(positions)} "
        #         f"positions=[{preview}] triggers={triggers} axes={axes}"
        #     )

    def start_vr_stream_thread(self):
        """启动 VR 串流线程：将 self.camera_images 合成为一张 image，推给 DummyServer。"""
        target_period = 1.0 / 30.0

        def _to_rgb_uint8(img):
            """Normalize an entry from self.camera_images to an HxWx3 RGB uint8 ndarray."""
            if img is None:
                return None
            if isinstance(img, (list, tuple)):
                if not img:
                    return None
                img = img[-1]
            if not isinstance(img, np.ndarray):
                return None
            if img.ndim == 2:
                # 深度或灰度：归一化到 8-bit，再展成三通道
                arr = img.astype(np.float32)
                if np.isfinite(arr).any():
                    valid = arr[np.isfinite(arr)]
                    vmin = float(valid.min()) if valid.size else 0.0
                    vmax = float(valid.max()) if valid.size else 1.0
                else:
                    vmin, vmax = 0.0, 1.0
                span = max(vmax - vmin, 1e-6)
                arr = np.clip((arr - vmin) / span, 0.0, 1.0)
                arr = (arr * 255.0).astype(np.uint8)
                return np.stack([arr, arr, arr], axis=-1)
            if img.ndim == 3 and img.shape[2] >= 3:
                arr = img[:, :, :3]
                if arr.dtype != np.uint8:
                    arr = np.clip(arr, 0, 255).astype(np.uint8)
                return arr
            return None

        def _compose(frames):
            """水平拼接所有相机的最新帧，统一到相同高度。"""
            if not frames:
                return None
            target_h = max(f.shape[0] for f in frames)
            resized = []
            for f in frames:
                h, w = f.shape[:2]
                if h != target_h:
                    new_w = max(1, int(round(w * target_h / h)))
                    f = cv2.resize(f, (new_w, target_h), interpolation=cv2.INTER_AREA)
                resized.append(f)
            return np.concatenate(resized, axis=1)

        def stream_loop():
            next_tick = time.monotonic()
            while True:
                try:
                    frames = []
                    # 当前：从兼容镜像 camera_images 读取（显示线程已镜像 env 抓取的帧）。
                    # for name in sorted(self.camera_images.keys()):
                    #     rgb = _to_rgb_uint8(self.camera_images.get(name))
                    #     if rgb is not None:
                    #         frames.append(rgb)
                    # 未来切换到 env 单一数据源（VR 作为共同请求者，与显示共享同一次抓取）：
                    for name in sorted(self._selected_display_cameras()):
                        rgb = _to_rgb_uint8(self.env.get_frame(name))
                        if rgb is not None:
                            frames.append(rgb)
                    composite = _compose(frames)
                    if composite is not None:
                        self.dummy_server.set_image(composite, color_order="RGB")
                except Exception as e:
                    print(f"VR 串流合成错误: {e}")
                next_tick += target_period
                sleep_for = next_tick - time.monotonic()
                if sleep_for > 0:
                    time.sleep(sleep_for)
                else:
                    next_tick = time.monotonic()

        stream_thread = threading.Thread(target=stream_loop, daemon=True)
        stream_thread.start()
