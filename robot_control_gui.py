import copy
import dataclasses
import os
import signal
import tkinter as tk
import typing
from tkinter import ttk
import cv2
import base64
import json
import http
import numpy as np
from PIL import Image, ImageTk
import threading
import time
import math
import rclpy
from a2d_sdk.robot import RobotDds as Robot, CosineCamera as Camera, RobotController
from scipy.ndimage import uniform_filter1d
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R
from camera_pose import compute_point_B_world
from control_wheel_example import WheelController
from robot_info_server import create_robot_info_http_server, RobotInfo
from constants import *
# pico vr
from pico_vr.pico_vr_server.server import DummyServer, _TRIGGER_LABELS
from pico_vr.pico_vr_common.protocol import JointState


PC4080_HOST = "10.12.11.144"
PC4080_PORT = 9000


class ExponentialSmoother:
    """
    实时平滑，每次只处理新来的一个动作
    适合推理时的流式输出
    """

    def __init__(self, alpha=0.3):
        """
        alpha: 平滑因子，0-1之间
               越大响应越快，越小越平滑
        """
        self.alpha = alpha
        self.prev_smoothed = None

    def set_alpha(self, new_alpha):
        if self.alpha != new_alpha:
            print(f"update alpha from {self.alpha} to {new_alpha}")
            self.alpha = new_alpha

    def clear(self):
        self.prev_smoothed = None

    def smooth(self, action):
        """
        action: 当前要执行的动作 (8,)
        """
        if isinstance(action, list):
            action = np.array(action)
        assert isinstance(action, np.ndarray)
        if self.prev_smoothed is None:
            self.prev_smoothed = action
            return action

        # 指数移动平均
        smoothed = self.alpha * action + (1 - self.alpha) * self.prev_smoothed
        self.prev_smoothed = smoothed
        return smoothed


class RobotCoordinateTransformer:
    def __init__(self, urdf_params=None):
        # 从URDF中提取的固定偏移量 (单位: 米)
        self.offsets = {
            'lift_base': [0, 0, 0.6485],              # base_link -> link-up-down_body
            'body_pitch_origin': [0.131, 0, 0],        # link-up-down_body -> link-pitch_body
            'head_yaw_origin': [0.441, 0, 0],          # link-pitch_body -> link-yaw_head
            'head_pitch_origin': [-0.050238, 0, 0.060065], # link-yaw_head -> link-pitch_head
            'camera_offset': [0, 0, 0]                 # link-pitch_head -> 相机光学中心
        }

        # 机械臂段长度 (用于IK求解)
        self.arm_lengths = [0.188, 0.305, 0.1975, 0.181, 0.23]

    def get_transformation_matrix(self, translation, rpy):
        """创建齐次变换矩阵"""
        mat = np.eye(4)
        mat[:3, :3] = R.from_euler('xyz', rpy).as_matrix()
        mat[:3, 3] = translation
        return mat

    def compute_camera_to_base_matrix(self, joint_angles):
        """
        计算相机坐标系到机器人基座的变换矩阵
        joint_angles: [waist_lift, waist_pitch, head_yaw, head_pitch]
        """
        w_lift, w_pitch, h_yaw, h_pitch = joint_angles

        # 1. Base -> Lift Body (Prismatic)
        T_lift = np.eye(4)
        T_lift[2, 3] = self.offsets['lift_base'][2] + w_lift

        # 2. Lift Body -> Pitch Body (Revolute)
        T_body_pitch = self.get_transformation_matrix(self.offsets['body_pitch_origin'], [1.5708, -1.5708, 0])
        R_body_pitch = R.from_rotvec([0, 0, -w_pitch]).as_matrix()
        T_body_pitch[:3, :3] = T_body_pitch[:3, :3] @ R_body_pitch

        # 3. Pitch Body -> Yaw Head (Fixed/Revolute)
        T_head_yaw = self.get_transformation_matrix(self.offsets['head_yaw_origin'], [1.5708, 0, 1.5708])
        R_head_yaw = R.from_rotvec([0, 0, h_yaw]).as_matrix()
        T_head_yaw[:3, :3] = T_head_yaw[:3, :3] @ R_head_yaw

        # 4. Yaw Head -> Pitch Head (Fixed/Revolute)
        T_head_pitch = self.get_transformation_matrix(self.offsets['head_pitch_origin'], [1.5708, 0, 0])
        R_head_pitch = R.from_rotvec([0, 0, h_pitch]).as_matrix()
        T_head_pitch[:3, :3] = T_head_pitch[:3, :3] @ R_head_pitch

        # 组合所有变换
        T_base_camera = T_lift @ T_body_pitch @ T_head_yaw @ T_head_pitch
        return T_base_camera

    def pixel_to_world(self, pixel_x, pixel_y, depth_value, intrinsics, joint_angles):
        """将像素点转换为世界坐标系下的位置"""
        fx, fy = intrinsics['fx'], intrinsics['fy']
        cx, cy = intrinsics['cx'], intrinsics['cy']
        z = depth_value * 0.001 # mm -> m

        x_cam = (pixel_x - cx) * z / fx
        y_cam = (pixel_y - cy) * z / fy
        p_cam = np.array([x_cam, y_cam, z, 1.0])

        T_base_camera = self.compute_camera_to_base_matrix(joint_angles)
        p_base = T_base_camera @ p_cam
        return p_base[:3]

    def compute_arm_fk(self, side, joint_angles):
        """
        计算手臂末端在基座坐标系下的位置 (简易版)
        joint_angles: [q1, q2, q3, q4, q5, q6, q7]
        """
        # 初始位置 (手臂挂载点)
        p = np.array([0.3, 0.025 if side == 'left' else -0.025, 0.7])

        # 简化正运动学：将各段长度沿当前旋转方向累加
        # 注意：这里是简化模型，实际应使用 URDF 的完整 DH 参数或变换矩阵链
        current_rot = R.from_euler('xyz', [0, 0, 0])

        for i in range(len(self.arm_lengths)):
            # 更新当前旋转 (简化：假设每个关节绕特定轴旋转)
            if i < len(joint_angles):
                # 简化轴向映射
                axis = [0,0,1] if i%2==0 else [0,1,0]
                current_rot = current_rot * R.from_rotvec(np.array(axis) * joint_angles[i])

            # 沿当前方向移动一段距离
            direction = current_rot.apply([1, 0, 0]) # 假设臂沿X轴延伸
            p = p + direction * self.arm_lengths[i]

        return p

    def solve_ik(self, side, target_pos, current_joints):
        """
        使用数值优化方法求解逆运动学 (IK)
        target_pos: [x, y, z] 目标位置
        current_joints: 当前关节角度 (用于作为优化起点，减少突跳)
        """
        from scipy.optimize import minimize

        def objective(q):
            # 计算当前关节角下的末端位置与目标位置的欧氏距离
            current_pos = self.compute_arm_fk(side, q)
            return np.linalg.norm(current_pos - target_pos)

        # 关节限位 (从 URDF 提取)
        bounds = [(-3.14, 3.14)] * 7 # 简化处理，实际应根据 URDF limit 设置

        res = minimize(objective, current_joints, bounds=bounds, method='L-BFGS-B')

        if res.success:
            return res.x
        else:
            return None


class RobotControlGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("智元G1机器人控制界面")
        self.root.geometry("1500x950")
        self.root.minsize(1200, 800)
        self._setup_styles()
        
        # 初始化机器人和相机
        self.robot = Robot()
        self.camera = Camera(["hand_left", "hand_right", "head", "head_depth", "head_center_fisheye"])
        self.robot_controller = RobotController()

        # Slam
        rclpy.init(args=None)
        self.wheel_controller = WheelController()
        wheel_thread = threading.Thread(target=rclpy.spin, args=(self.wheel_controller,), daemon=True)
        wheel_thread.start()

        # 等待初始化
        time.sleep(1.0)
        # 状态栏用于显示消息（替代messagebox）
        self.status_text = tk.StringVar()
        self.status_text.set("就绪")
        
        # 相机图像缓存
        self.camera_images = {
            "hand_left": None,  # cache latest two images for inference
            # "hand_right": None,  # cache latest two images for inference
            "head": None,
            "head_depth": None,  # cache latest one image
            # "head_center_fisheye": None
        }
        self.last_two_left_arm_joint_values = []
        # 由推理数据采集线程独占缓存的相机（start_camera_thread 不再重复 cache）
        self.inference_managed_cameras = set()
        # 相机显示缩放尺寸（由 _rebuild_camera_display 根据排版动态更新）
        self.camera_tile_size = (320, 240)

        # 相机内参缓存
        self.camera_intrinsics = {
            "hand_left": None,
            # "hand_right": None,
            "head": None,
            "head_depth": None,
            # "head_center_fisheye": None
        }

        # update robot_info
        self.robot_info = RobotInfo()
        create_robot_info_http_server(self.robot_info)

        # VR 串流：把 camera_images 合成的画面通过 DummyServer 推给客户端
        self.dummy_server = DummyServer(
            host="0.0.0.0",
            port=5555,
            port2=5556,
            image_path=None,
            use_default_image=False,
            rate_hz=30.0,
            jpeg_quality=80,
            on_joints=self._handle_vr_joints,
        )
        self.dummy_server.start()
        self.last_joint_update_timestamp: float = 0.0
        self.previous_vr_positions = []
        self.vr_actions = []
        self.vr_execution_thread = None
        self.is_vr_control: bool = False

        # auto inference
        self.actions = []
        self.is_auto_inference: bool = False
        self.is_grabbing_target: bool = False
        self.inference_thread = None
        self.execution_thread = None

        # MONEY
        self.coordinates_3d = (0, 0, 0)
        self.camera_hand_coordinates_3d = (0, 0, 0)
        self.camera_target_coordinates_3d = (0, 0, 0)
        self.world_hand_coordinates_3d = (0, 0, 0)
        self.world_target_coordinates_3d = (0, 0, 0)
        # pre set
        head_states = self.robot.head_joint_states()[0]
        waist_states = self.robot.waist_joint_states()[0]
        if all(abs(head_states[i] - HEAD[i]) < 0.001 for i in range(len(HEAD))) and all(abs(waist_states[i] - WAIST[i]) < 0.001 for i in range(len(WAIST))):
            self.camera_hand_coordinates_3d = LEFT_HAND_READY_CAMERA_COORDINATE_3D
            self.world_hand_coordinates_3d = LEFT_HAND_READY_COORDINATE_3D

        self.delta_coordinates_3d = (0, 0, 0)
        self.hand_status_text = tk.StringVar()
        self.hand_status_text.set("null")
        self.target_status_text = tk.StringVar()
        self.target_status_text.set("null")
        
        # RGBD坐标转换相关
        self.depth_scale = 0.001  # 深度缩放因子（毫米到米）
        self.click_coordinates = []  # 存储点击的图像坐标
        self.current_camera_for_3d = "head"  # 默认用于3D坐标获取的相机
        self.coordinate_display = None  # 坐标显示标签，将在setup_camera_panel中创建

        # 坐标转换处理器
        self.transformer = RobotCoordinateTransformer()

        # 状态栏相关
        self.status_text = tk.StringVar()
        self.status_text.set("就绪")
        self.status_label = None
        
        # 控制参数
        self.waist_lift_pos = 0.0
        self.waist_pitch_pos = 0.0
        self.head_yaw_pos = 0.0
        self.head_pitch_pos = 0.0
        self.left_gripper_pos = 0.0
        self.right_gripper_pos = 0.0
        
        # 关节名称映射
        self.joint_names = {
            'arm': ['左臂关节1', '左臂关节2', '左臂关节3', '左臂关节4', 
                   '左臂关节5', '左臂关节6', '左臂关节7', '右臂关节1', 
                   '右臂关节2', '右臂关节3', '右臂关节4', '右臂关节5', 
                   '右臂关节6', '右臂关节7'],
            'head': ['头部偏航', '头部俯仰'],
            'waist': ['腰部升降', '腰部俯仰'],
            'gripper': ['左夹爪', '右夹爪']
        }
        
        self.setup_ui()
        self.start_inference_data_collection_thread()
        self.start_camera_thread()
        self.start_status_thread()
        self.start_vr_stream_thread()

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

    def _setup_styles(self):
        """统一配置 ttk 视觉样式：颜色、字体、内边距，让界面更现代化。"""
        style = ttk.Style()
        try:
            style.theme_use('clam')
        except tk.TclError:
            pass

        # 调色板
        PRIMARY = "#1976d2"      # 主操作（推理、抓取等）
        PRIMARY_HOVER = "#1565c0"
        SUCCESS = "#2e7d32"      # 安全/确认操作（回 Home、张开）
        SUCCESS_HOVER = "#1b5e20"
        WARN = "#ef6c00"         # 警示操作（前往抓取/释放）
        WARN_HOVER = "#e65100"
        DANGER = "#c62828"       # 危险操作（自动运行/复位）
        DANGER_HOVER = "#8e0000"
        MUTED = "#546e7a"        # 次要按钮（清除、复位坐标）
        MUTED_HOVER = "#37474f"
        TEXT = "#1a1a1a"
        BG = "#f5f6f8"
        CARD_BG = "#ffffff"

        self.root.configure(bg=BG)

        # ---- 文本/容器样式 ----
        style.configure("TFrame", background=BG)
        style.configure("Card.TFrame", background=CARD_BG)

        style.configure("TLabel", background=BG, foreground=TEXT, font=("Microsoft YaHei", 10))
        style.configure("Title.TLabel",
                        background=BG, foreground=PRIMARY,
                        font=("Microsoft YaHei", 18, "bold"))
        style.configure("Subtitle.TLabel",
                        background=BG, foreground="#666",
                        font=("Microsoft YaHei", 10))
        style.configure("Section.TLabel",
                        background=BG, foreground=TEXT,
                        font=("Microsoft YaHei", 11, "bold"))
        style.configure("Value.TLabel",
                        background=BG, foreground="#0d47a1",
                        font=("Consolas", 10))
        style.configure("Status.TLabel",
                        background="#263238", foreground="#e0f7fa",
                        font=("Consolas", 10), padding=(8, 4))

        style.configure("TLabelframe", background=BG, borderwidth=1, relief="solid")
        style.configure("TLabelframe.Label",
                        background=BG, foreground=PRIMARY,
                        font=("Microsoft YaHei", 10, "bold"))

        style.configure("TNotebook", background=BG, borderwidth=0)
        style.configure("TNotebook.Tab",
                        font=("Microsoft YaHei", 10, "bold"),
                        padding=(16, 6))
        style.map("TNotebook.Tab",
                  background=[("selected", PRIMARY)],
                  foreground=[("selected", "white")])

        # ---- 按钮样式 ----
        base_btn = dict(font=("Microsoft YaHei", 10), padding=(10, 6), borderwidth=0)
        style.configure("TButton", **base_btn)

        def _color_button(name, base, hover, fg="white"):
            style.configure(name, background=base, foreground=fg, **base_btn)
            style.map(name,
                      background=[("active", hover), ("pressed", hover)],
                      foreground=[("active", fg)])

        _color_button("Primary.TButton", PRIMARY, PRIMARY_HOVER)
        _color_button("Success.TButton", SUCCESS, SUCCESS_HOVER)
        _color_button("Warn.TButton", WARN, WARN_HOVER)
        _color_button("Danger.TButton", DANGER, DANGER_HOVER)
        _color_button("Muted.TButton", MUTED, MUTED_HOVER)
        _color_button("Accent.TButton", PRIMARY, PRIMARY_HOVER)  # 兼容旧引用

        # 复选框
        style.configure("TCheckbutton",
                        background=BG, foreground=TEXT,
                        font=("Microsoft YaHei", 10))
        # 下拉框
        style.configure("TCombobox", padding=4)

    def setup_ui(self):
        """设置用户界面：顶部标题栏 + 左右分栏 + 底部状态栏。"""
        # ===== 顶部标题栏 =====
        header = ttk.Frame(self.root)
        header.pack(fill=tk.X, padx=16, pady=(12, 4))
        ttk.Label(header, text="智元 G1 机器人控制面板",
                  style="Title.TLabel").pack(side=tk.LEFT)
        ttk.Label(header, text="Mujin · Humanoid Control Console",
                  style="Subtitle.TLabel").pack(side=tk.LEFT, padx=12, pady=(8, 0))

        # ===== 底部状态栏（先建，固定贴底） =====
        status_bar = ttk.Frame(self.root)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X, padx=12, pady=(4, 10))
        self.status_label = ttk.Label(
            status_bar, textvariable=self.status_text,
            style="Status.TLabel", anchor=tk.W)
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(status_bar, text="清除",
                   style="Muted.TButton",
                   command=lambda: self.status_text.set("就绪")
                   ).pack(side=tk.RIGHT, padx=(8, 0))

        # ===== 主分栏 =====
        body = ttk.Frame(self.root)
        body.pack(fill=tk.BOTH, expand=True, padx=12, pady=4)

        left_frame = ttk.Frame(body)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))

        right_frame = ttk.Frame(body)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(8, 0))

        # 左侧：相机画面
        self.setup_camera_panel(left_frame)

        # 右侧：单一统一 Notebook 包含 [关节状态 / 抓取任务 / 手动调试]
        right_notebook = ttk.Notebook(right_frame)
        right_notebook.pack(fill=tk.BOTH, expand=True)

        status_tab = ttk.Frame(right_notebook)
        right_notebook.add(status_tab, text="📊  关节状态")
        self.setup_status_panel(status_tab)

        self.setup_pick_and_place_panel(right_notebook)
        self.setup_control_panel(right_notebook)
        
    def setup_camera_panel(self, parent):
        """设置相机图像面板：RGBD 控制条 + 相机勾选条 + 动态网格显示区。"""
        camera_frame = ttk.LabelFrame(parent, text="  📷  相机视图  ")
        camera_frame.pack(fill=tk.BOTH, expand=True)

        # 可显示的相机及其中文名（须为 Camera() 初始化时声明的相机）
        self.available_cameras = [
            "head", "head_depth", "hand_left", "hand_right", "head_center_fisheye"]
        self.camera_titles = {
            "head": "头部相机",
            "head_depth": "深度相机",
            "hand_left": "左手相机",
            "hand_right": "右手相机",
            "head_center_fisheye": "头部鱼眼相机",
        }

        # ---- RGBD 工具条 ----
        toolbar = ttk.Frame(camera_frame)
        toolbar.pack(fill=tk.X, padx=10, pady=(8, 6))

        self.rgbd_enabled_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(toolbar, text="启用 3D 坐标点击",
                        variable=self.rgbd_enabled_var,
                        command=self.toggle_rgbd_mode).pack(side=tk.LEFT)

        ttk.Label(toolbar, text="目标相机:").pack(side=tk.LEFT, padx=(16, 4))
        self.target_camera_var = tk.StringVar(value="head")
        camera_combo = ttk.Combobox(
            toolbar, textvariable=self.target_camera_var,
            values=["head", "head_depth", "hand_left", "hand_right"],
            state="readonly", width=14)
        camera_combo.pack(side=tk.LEFT)
        camera_combo.bind('<<ComboboxSelected>>', self.on_camera_selection_change)

        ttk.Button(toolbar, text="清除坐标记录",
                   style="Muted.TButton",
                   command=self.clear_coordinate_records
                   ).pack(side=tk.RIGHT)

        # ---- 相机勾选条：选择要显示的相机 ----
        select_bar = ttk.Frame(camera_frame)
        select_bar.pack(fill=tk.X, padx=10, pady=(0, 6))
        ttk.Label(select_bar, text="显示相机:").pack(side=tk.LEFT, padx=(0, 4))
        # 默认显示当前缓存中已有的相机（head / head_depth）
        self.camera_display_vars = {
            name: tk.BooleanVar(value=name in self.camera_images)
            for name in self.available_cameras
        }
        for name in self.available_cameras:
            ttk.Checkbutton(
                select_bar, text=self.camera_titles[name],
                variable=self.camera_display_vars[name],
                command=self._rebuild_camera_display).pack(side=tk.LEFT, padx=4)

        # ---- 相机显示区（动态网格） ----
        self.camera_labels = {}
        self.camera_display_area = ttk.Frame(camera_frame)
        self.camera_display_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        self._rebuild_camera_display()

    def _compute_camera_tile_size(self, ncols, nrows):
        """根据可用面板尺寸与网格规模，计算保持 4:3 的单格图像尺寸。"""
        try:
            self.root.update_idletasks()
            win_w = self.root.winfo_width()
            win_h = self.root.winfo_height()
        except Exception:
            win_w = win_h = 0
        if win_w < 100:
            win_w = 1500
        if win_h < 100:
            win_h = 950
        # 左侧相机面板约占窗口一半宽；高度扣除标题/工具条/状态栏
        panel_w = max(360, win_w // 2 - 60)
        panel_h = max(360, win_h - 220)
        # 每格扣除标题栏(~44)与内边距
        cell_w = max(160, panel_w // ncols - 16)
        cell_h = max(120, panel_h // nrows - 44)
        # 保持 4:3 比例，取受限的一边
        tile_w = min(cell_w, int(cell_h * 4 / 3))
        tile_h = int(tile_w * 3 / 4)
        return max(120, tile_w), max(90, tile_h)

    def _rebuild_camera_display(self):
        """根据用户勾选的相机重建显示区：动态网格排版 + 自适应缩放尺寸。"""
        # 当前选中的相机（按固定顺序）
        selected = [name for name in self.available_cameras
                    if self.camera_display_vars[name].get()]

        # 同步缓存字典的键：新增的置 None，取消的移除（停止抓取）
        for name in selected:
            self.camera_images.setdefault(name, None)
            self.camera_intrinsics.setdefault(name, None)
        for name in list(self.camera_images.keys()):
            if name not in selected:
                self.camera_images.pop(name, None)
                self.camera_intrinsics.pop(name, None)

        # 清空旧控件并重置标签字典
        for child in self.camera_display_area.winfo_children():
            child.destroy()
        self.camera_labels = {}

        if not selected:
            self.camera_tile_size = (320, 240)
            return

        # 计算网格列数（1→1，2~4→2，5→3）与每格缩放尺寸
        n = len(selected)
        ncols = 1 if n == 1 else 2 if n <= 4 else 3
        nrows = math.ceil(n / ncols)
        self.camera_tile_size = self._compute_camera_tile_size(ncols, nrows)

        # 让各行各列均分空间
        for c in range(ncols):
            self.camera_display_area.columnconfigure(c, weight=1, uniform="cam")
        for r in range(nrows):
            self.camera_display_area.rowconfigure(r, weight=1, uniform="cam")
        # 取消旧的多余行列权重
        for c in range(ncols, ncols + 3):
            self.camera_display_area.columnconfigure(c, weight=0, uniform="")
        for r in range(nrows, nrows + 5):
            self.camera_display_area.rowconfigure(r, weight=0, uniform="")

        for idx, name in enumerate(selected):
            r, c = divmod(idx, ncols)
            card = ttk.Frame(self.camera_display_area)
            card.grid(row=r, column=c, sticky="nsew", padx=4, pady=4)

            header = ttk.Frame(card)
            header.pack(fill=tk.X, pady=(0, 4))
            ttk.Label(header, text=f"📷  {self.camera_titles.get(name, name)}",
                      style="Section.TLabel").pack(side=tk.LEFT, anchor=tk.W)
            ttk.Button(header, text="💾 保存图片",
                       style="Primary.TButton",
                       command=lambda n=name: self.save_camera_frame(n)
                       ).pack(side=tk.RIGHT)

            label = ttk.Label(card, borderwidth=1, relief="solid",
                              background="#222", anchor=tk.CENTER)
            label.pack(fill=tk.BOTH, expand=True)
            self.camera_labels[name] = label
            self._bind_camera_click(name)

        # 重新应用鼠标样式
        self.toggle_rgbd_mode()

    def setup_status_panel(self, parent):
        """关节状态面板：左侧"手臂"，右侧"头部&腰部"+"夹爪"，用 grid 整齐对齐。"""
        wrapper = ttk.Frame(parent)
        wrapper.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # 让左右两列均分
        wrapper.columnconfigure(0, weight=1, uniform="cols")
        wrapper.columnconfigure(1, weight=1, uniform="cols")

        def _make_kv_row(parent_frame, row, name):
            ttk.Label(parent_frame, text=name + ":",
                      anchor=tk.W).grid(row=row, column=0, sticky="w", padx=8, pady=3)
            value_label = ttk.Label(parent_frame, text="0.000",
                                    style="Value.TLabel", anchor=tk.E, width=10)
            value_label.grid(row=row, column=1, sticky="e", padx=8, pady=3)
            return value_label

        # ---- 左列：手臂关节 ----
        arm_box = ttk.LabelFrame(wrapper, text="  🦾  手臂关节  ")
        arm_box.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        arm_box.columnconfigure(1, weight=1)

        self.arm_status_labels = []
        for i in range(14):
            self.arm_status_labels.append(
                _make_kv_row(arm_box, i, self.joint_names['arm'][i]))
        # AGV 角度
        self.arm_status_labels.append(_make_kv_row(arm_box, 14, "AGV 角度"))

        # ---- 右列：头部&腰部 + 夹爪 ----
        right_col = ttk.Frame(wrapper)
        right_col.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        right_col.columnconfigure(0, weight=1)

        head_waist_box = ttk.LabelFrame(right_col, text="  🧠  头部 & 腰部  ")
        head_waist_box.grid(row=0, column=0, sticky="new", pady=(0, 8))
        head_waist_box.columnconfigure(1, weight=1)

        self.head_waist_status_labels = {}
        row_idx = 0
        for i, name in enumerate(self.joint_names['head']):
            self.head_waist_status_labels[f"head_{i}"] = _make_kv_row(
                head_waist_box, row_idx, name)
            row_idx += 1
        for i, name in enumerate(self.joint_names['waist']):
            self.head_waist_status_labels[f"waist_{i}"] = _make_kv_row(
                head_waist_box, row_idx, name)
            row_idx += 1

        gripper_box = ttk.LabelFrame(right_col, text="  🤏  夹爪  ")
        gripper_box.grid(row=1, column=0, sticky="new")
        gripper_box.columnconfigure(1, weight=1)

        self.gripper_status_labels = []
        for i, name in enumerate(self.joint_names['gripper']):
            self.gripper_status_labels.append(
                _make_kv_row(gripper_box, i, name))

    @property
    def wheel_angle_deg(self):
        return self.wheel_controller.agv_angle

    @property
    def left_arm_joint_values(self):
        return self.robot.arm_joint_states()[0][:7]

    @property
    def left_hand_pos(self):
        position = self.robot_controller.get_motion_status()['frames']['arm_left_link7']['position']
        x = position['x']
        y = position['y']
        z = position['z']
        return x, y, z

    def update_hand_position(self):
        x, y, z = self.left_hand_pos
        print(x, y, z)
        self.world_hand_coordinates_3d = (x, y, z)
        self.camera_hand_coordinates_3d = self.coordinates_3d
        self.world_hand_coordinates_3d_value_label.config(text=f"{[round(v, 3) for v in self.world_hand_coordinates_3d]}")
        self.camera_hand_coordinates_3d_value_label.config(text=f"{[round(v, 3) for v in self.camera_hand_coordinates_3d]}")

    def update_target_position(self):
        yaw, pitch = self.robot.head_joint_states()[0]
        # degree to rad
        yaw_rad = math.radians(-90 + yaw)
        pitch_rad = math.radians(90 + pitch)
        roll_deg = 0  # TODO
        roll = math.radians(roll_deg)
        self.camera_target_coordinates_3d = self.coordinates_3d
        # B_w = compute_point_B_world(A_c, B_c, A_w, roll, pitch, yaw_rad)
        assert self.world_hand_coordinates_3d
        print(self.camera_hand_coordinates_3d)
        print(self.camera_target_coordinates_3d)
        print(self.world_hand_coordinates_3d)
        print(yaw_rad)
        self.world_target_coordinates_3d = compute_point_B_world(
            self.camera_hand_coordinates_3d,
            self.camera_target_coordinates_3d,
            self.world_hand_coordinates_3d,
            roll,
            pitch_rad,  # reversed
            yaw_rad  # reversed
        )
        self.world_target_coordinates_3d_value_label.config(text=f"{[round(v, 3) for v in self.world_target_coordinates_3d]}")
        self.camera_target_coordinates_3d_value_label.config(text=f"{[round(v, 3) for v in self.camera_target_coordinates_3d]}")
        self.delta_coordinates_3d = np.array(self.world_target_coordinates_3d) - self.world_hand_coordinates_3d
        self.delta_coordinates_3d_frame_value_label.config(text=f"{[round(v, 3) for v in self.delta_coordinates_3d]}")

    def run_trajectory(self, path_nodes, side, delta_time, validate=False, validate_step=0.1, release=False):
        if release:
            self.robot.move_gripper([0, 0])
        if validate:
            if side == "left":
                arm_joint_values = self.robot.arm_joint_states()[0][:7]
            elif side == "right":
                arm_joint_values = self.robot.arm_joint_states()[0][7:14]
            else:
                raise RuntimeError(f"Unknown side: {side}")
            assert arm_joint_values
            assert all(len(arm_joint_values) == len(path_nodes[0]) and abs(arm_joint_values[i] - path_nodes[0][i]) < validate_step for i in range(len(arm_joint_values)))
        try:
            for node in path_nodes:
                # 获取当前实时状态
                arm_states, _ = self.robot.arm_joint_states()
                head_states, _ = self.robot.head_joint_states()
                waist_states, _ = self.robot.waist_joint_states()
                assert arm_states
                assert head_states
                assert waist_states

                robot_states = {
                    "head": head_states,
                    "waist": waist_states,
                    "arm": arm_states,
                }

                # 构建关节空间控制动作
                # 假设 SDK 的 trajectory_tracking_control 在 JOINT 模式下
                # action_data 接受关节目标值
                arm_key = f"{side}_arm"
                robot_actions = [{
                    arm_key: {
                        "action_data": node,
                        "control_type": "ABS_JOINT"
                    }
                }]

                # print(robot_states)
                # print(robot_actions)
                # 发送控制命令
                self.robot_controller.trajectory_tracking_control(
                    int(time.time() * 1e9),
                    robot_states,
                    robot_actions,
                    "base_link",
                    delta_time  # 每个点执行时间 s
                )
                time.sleep(delta_time)
            print(f"{side}臂轨迹执行完毕", "success")
        except Exception as traj_e:
            print(f"轨迹执行错误: {traj_e}")

    def plan_and_run_picking_trajectory(self, delta_coordinates_3d, release=False, y_first=False, time_step=0.02):
        if release:
            # open gripper
            self.robot.move_gripper([0, 0])
        # Ignore z for now
        step = 0.005
        sign_x = 1 if delta_coordinates_3d[0] >= 0 else -1
        sign_y = 1 if delta_coordinates_3d[1] >= 0 else -1
        deltas = [[0.0, 0.0, 0.0]]
        node = [0.0, 0.0, 0.0]
        is_x_reached = False
        is_y_reached = False
        while True:
            next_delta = [0.0, 0.0, 0.0]
            if y_first and not is_y_reached:
                pass
            elif is_x_reached:
                pass
            elif abs(node[0] - delta_coordinates_3d[0]) <= step:
                # next_delta[0] = delta_coordinates_3d[0] - node[0]
                node[0] = delta_coordinates_3d[0]
                is_x_reached = True
            else:
                next_delta[0] = step * sign_x
                node[0] = round(node[0] + step * sign_x, 3)

            if is_y_reached:
                pass
            elif abs(node[1] - delta_coordinates_3d[1]) <= step:
                # next_delta[1] = delta_coordinates_3d[1] - node[1]
                node[1] = delta_coordinates_3d[1]
                is_y_reached = True
            else:
                next_delta[1] = step * sign_y
                node[1] = round(node[1] + step * sign_y, 3)

            print(node)
            print(next_delta)
            deltas.append(next_delta)
            print(is_x_reached, is_y_reached)
            if is_x_reached and is_y_reached:
                break
        print(f"Deltas: {deltas}")

        for delta in deltas:
            self.move_arm_relative('left', delta, time_step=time_step)

    def get_smooth_paths(self, raw_paths, smooth_step = 0.005, validate=False, validate_step=0.2, filter_mode=""):
        paths = copy.deepcopy(raw_paths)
        index = 0
        while True:
            if index >= len(paths) - 1:
                break
            current_path = paths[index]
            next_path = paths[index + 1]
            insert_path = copy.deepcopy(current_path)
            assert len(current_path) == len(next_path)
            has_inserted_path = False
            for i in range(min(len(current_path), 7)):
                diff = abs(next_path[i] - current_path[i])
                if validate:
                    assert diff < validate_step, f"diff={diff} >= validate_step={validate_step}"
                if diff > smooth_step:
                    has_inserted_path = True
                    if next_path[i] > current_path[i]:
                        insert_path[i] += smooth_step
                    else:
                        insert_path[i] -= smooth_step
            if has_inserted_path:
                paths.insert(index + 1, insert_path)
            index += 1
        if filter_mode == "moving_average":
            return uniform_filter1d(paths, size=5, axis=0, mode='nearest')
        elif filter_mode == "savgol":
            window_length = min(7, len(paths))
            return savgol_filter(paths, window_length=window_length, polyorder=min(window_length - 1, 3), axis=0)
        return paths

    def grasp_approach(self):
        arm_joint_values = self.left_arm_joint_values
        assert arm_joint_values
        assert len(arm_joint_values) == len(LEFT_HAND_HOME_JOINT_VALUES)
        if all(abs(arm_joint_values[i] - LEFT_HAND_HOME_JOINT_VALUES[i]) < 0.1 for i in range(len(arm_joint_values))):
            self.run_trajectory(HOME_TO_READY_PATHS, "left", 0.02, validate=True, release=True)
            time.sleep(0.2)
        arm_joint_values = self.left_arm_joint_values
        assert arm_joint_values
        assert all(abs(arm_joint_values[i] - LEFT_HAND_READY_JOINT_VALUES[i]) < 0.1 for i in range(len(arm_joint_values)))
        self.plan_and_run_picking_trajectory(self.delta_coordinates_3d, release=True, y_first=True)


    def grasp_depart_home(self, grasp=False, depart=False, validate=True, direct=False):
        if grasp:
            self.robot.move_gripper([1, 0])
            time.sleep(1)
        if depart:
            for i in range(10):
                self.move_arm_relative('left', [0, 0.01, 0.005], time_step=0.05)
            # for i in range(20):
            #     self.move_arm_relative('left', [-0.01, 0, 0], time_step=0.02)
        # delta_coordinates_3d = np.array(LEFT_HAND_HIGH_HOME_COORDINATE_3D) - self.left_hand_pos
        # if validate:
        #     for index, limit in enumerate([0.3, 0.3, 0.2]):
        #         # TODO: for safe
        #         assert abs(delta_coordinates_3d[index]) <= limit
        # self.plan_and_run_picking_trajectory(delta_coordinates_3d, release=False, time_step=0.02)
        # self.run_trajectory([LEFT_HAND_HIGH_HOME_JOINT_VALUES], "left", 1)
        self.run_trajectory(self.get_smooth_paths([self.left_arm_joint_values, LEFT_HAND_HIGH_HOME_JOINT_VALUES]), "left", 0.01)
        # self.run_trajectory(HIGH_HOME_TO_HOME_PATHS, "left", 0.1)
        # delta_coordinates_3d = np.array(LEFT_HAND_HOME_COORDINATE_3D) - self.left_hand_pos
        # if validate:
        #     for index, limit in enumerate([0.3, 0.3, 0.2]):
        #         # TODO: for safe
        #         assert abs(delta_coordinates_3d[index]) <= limit
        # self.plan_and_run_picking_trajectory(delta_coordinates_3d, release=False, time_step=0.02)
        # self.run_trajectory([LEFT_HAND_HOME_JOINT_VALUES], "left", 1)
        self.run_trajectory(self.get_smooth_paths([self.left_arm_joint_values, LEFT_HAND_HOME_JOINT_VALUES]), "left", 0.01)

    def release_part(self, path, target_wheel_angle_deg=None, home_wheel_angle_deg=None):
        if target_wheel_angle_deg is not None:
            if abs(self.wheel_angle_deg - target_wheel_angle_deg) > 5:
                self.wheel_controller.commands.append(target_wheel_angle_deg)
                while True:
                    if abs(self.wheel_angle_deg - target_wheel_angle_deg) <= 10:
                        break
                    time.sleep(0.1)
        self.run_trajectory(path, "left", 0.02, validate=True)
        self.robot.move_gripper([0, 0])
        time.sleep(1)
        if home_wheel_angle_deg is not None:
            self.wheel_controller.commands.append(home_wheel_angle_deg)
        reversed_paths = list(reversed(path))
        self.run_trajectory(reversed_paths, "left", 0.02, validate=False)
        self.run_trajectory([LEFT_HAND_HOME_JOINT_VALUES], "left", 0.2)

    def setup_pick_and_place_panel(self, parent):
        """抓取任务面板：按"预设位置 / 坐标 / 抓取 / 推理 / 夹爪"分组。"""
        tab = ttk.Frame(parent)
        parent.add(tab, text="🎯  抓取任务")

        # 用 Canvas+滚动条 包裹，内容多时可滚动
        canvas = tk.Canvas(tab, highlightthickness=0, bg="#f5f6f8")
        sb = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        body = ttk.Frame(canvas)
        body.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=body, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        # ===== 1. 预设位置 =====
        sec_home = ttk.LabelFrame(body, text="  🏠  预设位置  ")
        sec_home.pack(fill=tk.X, padx=10, pady=(10, 6))
        row = ttk.Frame(sec_home)
        row.pack(fill=tk.X, padx=8, pady=8)
        ttk.Button(row, text="回 Home (直接)",
                   style="Success.TButton",
                   command=lambda: self.run_trajectory(
                       [LEFT_HAND_HOME_JOINT_VALUES], "left", 0.5)
                   ).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(row, text="前往 Pre-Grasp",
                   style="Primary.TButton",
                   command=lambda: self.run_trajectory(
                       HOME_TO_LEFT_PRE_GRASP_PATHS, "left", 0.01,
                       validate=True, release=True)
                   ).pack(side=tk.LEFT, padx=6)
        ttk.Button(row, text="将物品放回",
                   style="Warn.TButton",
                   command=lambda: self.release_part(HOME_TO_PUT_BACK_TARGET)
                   ).pack(side=tk.LEFT, padx=6)

        # ===== 2. 坐标信息 =====
        sec_coords = ttk.LabelFrame(body, text="  📍  坐标信息  ")
        sec_coords.pack(fill=tk.X, padx=10, pady=6)

        def _coord_row(parent_frame, row, label_text, init_text):
            ttk.Label(parent_frame, text=label_text,
                      anchor=tk.W).grid(row=row, column=0, sticky="w", padx=8, pady=3)
            value_label = ttk.Label(parent_frame, text=init_text,
                                    style="Value.TLabel", anchor=tk.W)
            value_label.grid(row=row, column=1, sticky="ew", padx=8, pady=3)
            return value_label

        # 手部坐标块
        hand_box = ttk.Frame(sec_coords)
        hand_box.pack(fill=tk.X, padx=8, pady=(8, 4))
        hand_box.columnconfigure(1, weight=1)
        ttk.Label(hand_box, text="✋ 手部位置",
                  style="Section.TLabel").grid(row=0, column=0, columnspan=2,
                                               sticky="w", padx=8)
        self.world_hand_coordinates_3d_value_label = _coord_row(
            hand_box, 1, "  world_hand_3d",
            f"{[round(v, 3) for v in self.world_hand_coordinates_3d]}")
        self.camera_hand_coordinates_3d_value_label = _coord_row(
            hand_box, 2, "  camera_hand_3d",
            f"{[round(v, 3) for v in self.camera_hand_coordinates_3d]}")
        ttk.Button(hand_box, text="🔄 更新手部位置",
                   style="Primary.TButton",
                   command=lambda: self.update_hand_position()
                   ).grid(row=3, column=0, columnspan=2, sticky="w", padx=8, pady=(4, 8))

        ttk.Separator(sec_coords, orient="horizontal").pack(fill=tk.X, padx=8, pady=4)

        # 目标坐标块
        target_box = ttk.Frame(sec_coords)
        target_box.pack(fill=tk.X, padx=8, pady=(4, 8))
        target_box.columnconfigure(1, weight=1)
        ttk.Label(target_box, text="🎯 目标位置",
                  style="Section.TLabel").grid(row=0, column=0, columnspan=2,
                                               sticky="w", padx=8)
        self.world_target_coordinates_3d_value_label = _coord_row(
            target_box, 1, "  world_target_3d", "0.000")
        self.camera_target_coordinates_3d_value_label = _coord_row(
            target_box, 2, "  camera_target_3d", "0.000")
        self.delta_coordinates_3d_frame_value_label = _coord_row(
            target_box, 3, "  delta_3d", "0.000")
        ttk.Button(target_box, text="🔄 更新目标位置",
                   style="Primary.TButton",
                   command=lambda: self.update_target_position()
                   ).grid(row=4, column=0, columnspan=2, sticky="w", padx=8, pady=(4, 4))

        # ===== 3. 抓取动作 =====
        sec_grasp = ttk.LabelFrame(body, text="  🤖  抓取动作  ")
        sec_grasp.pack(fill=tk.X, padx=10, pady=6)
        row = ttk.Frame(sec_grasp)
        row.pack(fill=tk.X, padx=8, pady=8)
        ttk.Button(row, text="前往抓取点",
                   style="Warn.TButton",
                   command=lambda: self.grasp_approach()
                   ).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(row, text="抓取并返回 Home",
                   style="Primary.TButton",
                   command=lambda: self.grasp_depart_home(
                       grasp=True, depart=True, validate=True, direct=True)
                   ).pack(side=tk.LEFT, padx=6)
        ttk.Button(row, text="前往释放点释放物件",
                   style="Warn.TButton",
                   command=lambda: self.release_part(
                       HOME_TO_RELEASE_PATHS,
                       target_wheel_angle_deg=220, home_wheel_angle_deg=180)
                   ).pack(side=tk.LEFT, padx=6)

        # ===== 4. 夹爪 =====
        sec_grip = ttk.LabelFrame(body, text="  🤏  左夹爪  ")
        sec_grip.pack(fill=tk.X, padx=10, pady=6)
        row = ttk.Frame(sec_grip)
        row.pack(fill=tk.X, padx=8, pady=8)
        ttk.Button(row, text="张开",
                   style="Success.TButton",
                   command=lambda: self.move_gripper("left", 0.0)
                   ).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(row, text="闭合",
                   style="Warn.TButton",
                   command=lambda: self.move_gripper("left", 1.0)
                   ).pack(side=tk.LEFT, padx=6)

        # ===== 5. 推理控制 =====
        sec_inf = ttk.LabelFrame(body, text="  🧠  推理控制  ")
        sec_inf.pack(fill=tk.X, padx=10, pady=6)

        manual_row = ttk.Frame(sec_inf)
        manual_row.pack(fill=tk.X, padx=8, pady=(8, 4))
        ttk.Label(manual_row, text="单步:",
                  style="Section.TLabel").pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(manual_row, text="推理一次",
                   style="Primary.TButton",
                   command=lambda: self.inference_once()
                   ).pack(side=tk.LEFT, padx=4)
        ttk.Button(manual_row, text="执行一次推理轨迹",
                   style="Primary.TButton",
                   command=lambda: self.execute_inference_result(once=True)
                   ).pack(side=tk.LEFT, padx=4)
        ttk.Button(manual_row, text="执行剩余推理轨迹",
                   style="Primary.TButton",
                   command=lambda: self.execute_inference_result()
                   ).pack(side=tk.LEFT, padx=4)

        auto_row = ttk.Frame(sec_inf)
        auto_row.pack(fill=tk.X, padx=8, pady=(4, 8))
        ttk.Label(auto_row, text="自动:",
                  style="Section.TLabel").pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(auto_row, text="⚠ 开始自动运行",
                   style="Danger.TButton",
                   command=lambda: self.auto_inference()
                   ).pack(side=tk.LEFT, padx=4)
        ttk.Button(auto_row, text="■ 停止自动运行",
                   style="Muted.TButton",
                   command=lambda: self.auto_inference(stop=True)
                   ).pack(side=tk.LEFT, padx=4)

        # ===== 6. VR控制 =====
        sec_vr = ttk.LabelFrame(body, text="  ●  VR控制  ")
        sec_vr.pack(fill=tk.X, padx=10, pady=6)

        vr_auto_row = ttk.Frame(sec_vr)
        vr_auto_row.pack(fill=tk.X, padx=8, pady=(4, 8))
        ttk.Label(vr_auto_row, text="自动:",
                  style="Section.TLabel").pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(vr_auto_row, text="⚠ 开始遥操",
                   style="Danger.TButton",
                   command=lambda: setattr(self, 'is_vr_control', True)
                   ).pack(side=tk.LEFT, padx=4)
        ttk.Button(vr_auto_row, text="■ 停止遥操",
                   style="Muted.TButton",
                   command=lambda: setattr(self, 'is_vr_control', False)
                   ).pack(side=tk.LEFT, padx=4)

    def auto_inference(self, stop: bool = False):
        if stop:
            self.is_auto_inference = False
            if self.execution_thread is not None:
                print("Stopping execution_thread thread...")
                self.execution_thread.join()
                self.execution_thread = None
            if self.inference_thread is not None:
                print("Stopping auto_inference thread...")
                self.inference_thread.join()
                self.inference_thread = None
            return
        self.is_auto_inference = True

        def _run_auto_inference():
            while self.is_auto_inference:
                # TODO: test
                if self.inference_once(use_deep_copy=True):
                    time.sleep(0.03)
                else:
                    time.sleep(0.01)

        def _run_auto_execution():
            exponential_smoother = ExponentialSmoother(alpha=0.2)
            last_inference_timestamp: float = 0.0
            smooth_step = 0.002
            self.actions = []
            while self.is_auto_inference:
                # ignore traj execution time
                now = time.time()
                if self.actions:
                    actions = self.actions[:2]
                    gripper_action = actions[-1][-1]
                    current_left_arm_joint_values = self.left_arm_joint_values
                    exponential_smoother.prev_smoothed = np.array(current_left_arm_joint_values)
                    actions = [current_left_arm_joint_values] + [exponential_smoother.smooth(a[:7]) for a in actions]
                    # actions = [exponential_smoother.smooth(a[:7]) for a in actions]
                    print(f"Executing actions: {actions}, all actions: {len(self.actions)}")
                    # self.run_trajectory(self.get_smooth_paths([self.left_arm_joint_values] + actions, smooth_step=0.0002, validate=True, validate_step=0.3)[:7], "left", 0.0001, validate=True, validate_step=0.2)
                    # moving_average or savgol
                    filter_mode = "moving_average"
                    # use_uniform_filter1d = not self.is_grabbing_target
                    # if use_uniform_filter1d:
                    #     smooth_step = 0.005 if self.is_grabbing_target else 0.005
                    # else:
                    # self.run_trajectory(self.get_smooth_paths([self.left_arm_joint_values] + actions, smooth_step=smooth_step, validate=True, validate_step=0.3)[:7], "left", 0.0001, validate=True, validate_step=0.2)
                    self.run_trajectory(self.get_smooth_paths(actions, smooth_step=smooth_step, validate=True, validate_step=0.3, filter_mode=filter_mode)[:7], "left", 0.0001, validate=True, validate_step=0.2)
                    # self.run_trajectory(actions, "left", 0.02, validate=True, validate_step=0.3)
                    if gripper_action > 0.5:
                        if not self.is_grabbing_target:
                            self.robot.move_gripper([1, 0])
                            self.is_grabbing_target = True
                            self.actions = []
                            time.sleep(1)
                            for _ in range(20):
                                self.move_arm_relative('left', [0, 0, 0.002], time_step=0.02)
                            last_inference_timestamp = time.time() + 1
                            continue
                    self.actions = self.actions[2:]
                if not self.actions:
                    x, y, z = self.left_hand_pos
                    # stop and move to home
                    if self.is_grabbing_target and x < 0.565 and z > 0.9:
                        print("stop auto inference and move back to home position")
                        self.is_auto_inference = False
                        self.run_trajectory(self.get_smooth_paths([self.left_arm_joint_values, LEFT_HAND_HOME_JOINT_VALUES]), "left", 0.01)
                        break
                    # check robot postion when there's no action
                    # TODO: test: robot is next to container
                    if self.is_grabbing_target:
                        if z < 0.9:
                            alpha = 0.3
                            smooth_step = 0.002
                        # elif x > 0.6:
                        #     alpha = 0.6
                        #     smooth_step = 0.01
                        else:
                            alpha = 0.6
                            smooth_step = 0.01
                    else:
                        if x > 0.64:
                            alpha = 0.1
                            smooth_step = 0.002
                        elif x > 0.6:
                            alpha = 0.2
                            smooth_step = 0.0025
                        elif x > 0.55:
                            alpha = 0.4
                            smooth_step = 0.005
                        else:
                            alpha = 0.3
                            smooth_step = 0.01
                    exponential_smoother.set_alpha(alpha)


                # TODO: no need to lock
                if self.robot_info.inference_timestamp <= last_inference_timestamp + 0.001:
                    if not self.actions:
                        time.sleep(0.01)
                    continue
                last_inference_timestamp = self.robot_info.inference_timestamp
                if self.is_grabbing_target and any(v[-1] < 0.5 for v in self.robot_info.left_joint_predict_action_values):
                    print(f"robot is grabbing target, skip the old inference which would release target")
                    continue
                diff = now - last_inference_timestamp
                if diff > 0.7:
                    print(f"inference diff={diff} > 0.7, not use at all")
                    continue
                if diff > 0.5:
                    print(f"inference diff={diff} > 0.5, use the last 1 action")
                    self.actions.append(self.robot_info.left_joint_predict_action_values[-1])
                elif diff > 0.3:
                    print(f"inference diff={diff} > 0.3, use last 2 actions")
                    self.actions.extend(self.robot_info.left_joint_predict_action_values[-2:])
                elif self.actions:
                    if len(self.actions) <= 4:
                        self.actions.extend(self.robot_info.left_joint_predict_action_values[-4:])
                    else:
                        self.actions = self.actions[-2:] + self.robot_info.left_joint_predict_action_values[-4:]
                else:
                    self.actions = self.robot_info.left_joint_predict_action_values

        if self.inference_thread is None:
            self.inference_thread = threading.Thread(target=_run_auto_inference, daemon=True)
            print("Starting auto_inference thread...")
            self.inference_thread.start()
        if self.execution_thread is None:
            gripper_states, _ = self.robot.gripper_states()
            self.is_grabbing_target = gripper_states[0] > 0.5
            self.execution_thread = threading.Thread(target=_run_auto_execution, daemon=True)
            print("Starting execution_thread thread...")
            self.execution_thread.start()


    def inference_once(self, use_deep_copy: bool = False):
        # TODO: left_hand only
        if self.robot_info.timestamp - self.robot_info.inference_timestamp < 0.0001:
            return False
        with self.robot_info.lock:
            last_two_left_arm_joint_values = copy.deepcopy(self.last_two_left_arm_joint_values)
            assert len(last_two_left_arm_joint_values) == 2
            inference_timestamp = self.robot_info.timestamp

        camera_images = copy.deepcopy(self.camera_images["hand_left"]) if use_deep_copy else self.camera_images["hand_left"]
        assert camera_images is not None and len(camera_images) == 2

        def _encode_image(rgb_image):
            rgb_image_cv2 = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            _, buffer = cv2.imencode('.jpg', rgb_image_cv2, [cv2.IMWRITE_JPEG_QUALITY, 90])
            base64_string = base64.b64encode(buffer).decode('utf-8')
            return base64_string

        req = {
            'left_imgs': [_encode_image(image) for image in camera_images],
            'state': last_two_left_arm_joint_values,
        }

        def post_predict(host: str, port: int, req: dict, timeout: float = 60.0) -> dict:
            body = json.dumps(req).encode('utf-8')
            conn = http.client.HTTPConnection(host, port, timeout=timeout)
            try:
                conn.request(
                    'POST', '/predict', body=body,
                    headers={
                        'Content-Type': 'application/json; charset=utf-8',
                        'Content-Length': str(len(body)),
                    })
                resp = conn.getresponse()
                resp_body = resp.read()
                try:
                    resp_obj = json.loads(resp_body.decode('utf-8')) if resp_body else {}
                except Exception as e:
                    return {'error': f'failed to parse JSON response: {e!r} '
                                     f'(status={resp.status})'}
                if resp.status != 200:
                    if isinstance(resp_obj, dict) and 'error' in resp_obj:
                        return resp_obj
                    return {'error': f'HTTP {resp.status}: {resp_obj!r}'}
                return resp_obj
            finally:
                conn.close()

        resp = post_predict(PC4080_HOST, PC4080_PORT, req, timeout=10)

        if 'error' in resp:
            print(f'\nServer error: {resp["error"]}')
            return

        action = np.asarray(resp['action'], dtype=np.float32)
        with self.robot_info.lock:
            self.robot_info.left_joint_predict_start_values = copy.deepcopy(last_two_left_arm_joint_values)
            self.robot_info.left_joint_predict_action_values = action.tolist()
            self.robot_info.inference_timestamp = inference_timestamp
        # print("left_joint_predict_start_values=", self.robot_info.left_joint_predict_start_values)
        # print("left_joint_predict_action_values=", self.robot_info.left_joint_predict_action_values)
        return True


    def execute_inference_result(self, once: bool = False):
        def safety_check():
            return True
        with self.robot_info.lock:
            while self.robot_info.left_joint_predict_action_values:
                action = self.robot_info.left_joint_predict_action_values.pop(0)
                if not safety_check():
                    self.robot_info.left_joint_predict_start_values = None
                    self.robot_info.left_joint_predict_action_values = None
                    break
                # DANGEROUS!!!!
                self.run_trajectory([action[:7]], "left", 0.2, validate=True, validate_step=0.4)
                self.robot.move_gripper([1 if action[-1] > 0.5 else 0, 0])
                if once:
                    break
        
    def setup_control_panel(self, parent):
        """手动调试面板：腰/头/夹爪/机械臂方向控制 + 预设姿态 + 复位。"""
        tab = ttk.Frame(parent)
        parent.add(tab, text="🕹  手动调试")

        # 滚动容器
        canvas = tk.Canvas(tab, highlightthickness=0, bg="#f5f6f8")
        sb = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        body = ttk.Frame(canvas)
        body.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=body, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        def _arrow_btn(parent_frame, text, command, style="Primary.TButton"):
            return ttk.Button(parent_frame, text=text, style=style,
                              command=command, width=8)

        # ===== 腰部 =====
        sec_waist = ttk.LabelFrame(body, text="  🧍  腰部控制  ")
        sec_waist.pack(fill=tk.X, padx=10, pady=(10, 6))
        row1 = ttk.Frame(sec_waist); row1.pack(fill=tk.X, padx=8, pady=6)
        ttk.Label(row1, text="升降:", width=8).pack(side=tk.LEFT)
        _arrow_btn(row1, "▲ 上升", lambda: self.move_waist_lift(2.0)).pack(side=tk.LEFT, padx=3)
        _arrow_btn(row1, "▼ 下降", lambda: self.move_waist_lift(-2.0)).pack(side=tk.LEFT, padx=3)
        _arrow_btn(row1, "复位", lambda: self.move_waist_lift(0.0),
                   style="Muted.TButton").pack(side=tk.LEFT, padx=3)
        row2 = ttk.Frame(sec_waist); row2.pack(fill=tk.X, padx=8, pady=(0, 8))
        ttk.Label(row2, text="俯仰:", width=8).pack(side=tk.LEFT)
        _arrow_btn(row2, "↓ 前倾", lambda: self.move_waist_pitch(0.5)).pack(side=tk.LEFT, padx=3)
        _arrow_btn(row2, "↑ 后仰", lambda: self.move_waist_pitch(-0.5)).pack(side=tk.LEFT, padx=3)
        _arrow_btn(row2, "复位", lambda: self.move_waist_pitch(0.0),
                   style="Muted.TButton").pack(side=tk.LEFT, padx=3)

        # ===== 头部 =====
        sec_head = ttk.LabelFrame(body, text="  👀  头部控制  ")
        sec_head.pack(fill=tk.X, padx=10, pady=6)
        row1 = ttk.Frame(sec_head); row1.pack(fill=tk.X, padx=8, pady=6)
        ttk.Label(row1, text="左右:", width=8).pack(side=tk.LEFT)
        _arrow_btn(row1, "← 左转", lambda: self.move_head_yaw(0.3)).pack(side=tk.LEFT, padx=3)
        _arrow_btn(row1, "→ 右转", lambda: self.move_head_yaw(-0.3)).pack(side=tk.LEFT, padx=3)
        _arrow_btn(row1, "复位", lambda: self.move_head_yaw(0.0),
                   style="Muted.TButton").pack(side=tk.LEFT, padx=3)
        row2 = ttk.Frame(sec_head); row2.pack(fill=tk.X, padx=8, pady=(0, 8))
        ttk.Label(row2, text="俯仰:", width=8).pack(side=tk.LEFT)
        _arrow_btn(row2, "↑ 上扬", lambda: self.move_head_pitch(-0.3)).pack(side=tk.LEFT, padx=3)
        _arrow_btn(row2, "↓ 下俯", lambda: self.move_head_pitch(0.3)).pack(side=tk.LEFT, padx=3)
        _arrow_btn(row2, "复位", lambda: self.move_head_pitch(0.0),
                   style="Muted.TButton").pack(side=tk.LEFT, padx=3)

        # ===== 夹爪 =====
        sec_grip = ttk.LabelFrame(body, text="  🤏  夹爪控制  ")
        sec_grip.pack(fill=tk.X, padx=10, pady=6)
        for side, label in [("left", "左夹爪"), ("right", "右夹爪")]:
            row = ttk.Frame(sec_grip); row.pack(fill=tk.X, padx=8, pady=5)
            ttk.Label(row, text=label + ":", width=8).pack(side=tk.LEFT)
            _arrow_btn(row, "张开", lambda s=side: self.move_gripper(s, 0.0),
                       style="Success.TButton").pack(side=tk.LEFT, padx=3)
            _arrow_btn(row, "闭合", lambda s=side: self.move_gripper(s, 1.0),
                       style="Warn.TButton").pack(side=tk.LEFT, padx=3)

        # ===== 机械臂方向控制（3x3 网格） =====
        sec_arm = ttk.LabelFrame(body, text="  🦾  机械臂末端方向控制 (步长 5cm)  ")
        sec_arm.pack(fill=tk.X, padx=10, pady=6)

        def _build_dpad(parent_frame, side):
            grid = ttk.Frame(parent_frame)
            grid.pack(side=tk.LEFT, padx=12, pady=8)
            ttk.Label(grid, text=("左臂" if side == "left" else "右臂"),
                      style="Section.TLabel").grid(row=0, column=0, columnspan=3, pady=(0, 4))
            # 第一排: 前 / 上
            _arrow_btn(grid, "↑ 前", lambda: self.move_arm_relative(side, [0.05, 0, 0])
                       ).grid(row=1, column=1, padx=2, pady=2)
            _arrow_btn(grid, "▲ 上", lambda: self.move_arm_relative(side, [0, 0, 0.05])
                       ).grid(row=1, column=2, padx=2, pady=2)
            # 中排: 左 / 右
            _arrow_btn(grid, "← 左", lambda: self.move_arm_relative(side, [0, 0.05, 0])
                       ).grid(row=2, column=0, padx=2, pady=2)
            _arrow_btn(grid, "→ 右", lambda: self.move_arm_relative(side, [0, -0.05, 0])
                       ).grid(row=2, column=2, padx=2, pady=2)
            # 末排: 后 / 下
            _arrow_btn(grid, "↓ 后", lambda: self.move_arm_relative(side, [-0.05, 0, 0])
                       ).grid(row=3, column=1, padx=2, pady=2)
            _arrow_btn(grid, "▼ 下", lambda: self.move_arm_relative(side, [0, 0, -0.05])
                       ).grid(row=3, column=2, padx=2, pady=2)

        arms_row = ttk.Frame(sec_arm); arms_row.pack(fill=tk.X, padx=8, pady=4)
        _build_dpad(arms_row, "left")
        ttk.Separator(arms_row, orient="vertical").pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=8)
        _build_dpad(arms_row, "right")

        # ===== 预设姿态 =====
        sec_preset = ttk.LabelFrame(body, text="  ⭐  预设姿态  ")
        sec_preset.pack(fill=tk.X, padx=10, pady=6)
        row = ttk.Frame(sec_preset); row.pack(fill=tk.X, padx=8, pady=8)
        ttk.Button(row, text="双臂平举", style="Primary.TButton",
                   command=lambda: self.set_arm_preset("parallel")
                   ).pack(side=tk.LEFT, padx=3)
        ttk.Button(row, text="双臂下垂", style="Primary.TButton",
                   command=lambda: self.set_arm_preset("down")
                   ).pack(side=tk.LEFT, padx=3)
        ttk.Button(row, text="双臂前伸", style="Primary.TButton",
                   command=lambda: self.set_arm_preset("forward")
                   ).pack(side=tk.LEFT, padx=3)

        # ===== 全局复位 =====
        sec_reset = ttk.Frame(body)
        sec_reset.pack(fill=tk.X, padx=10, pady=(10, 14))
        ttk.Button(sec_reset, text="⚠  机器人全部复位",
                   style="Danger.TButton",
                   command=self.reset_robot
                   ).pack(fill=tk.X, ipady=4)

    def start_inference_data_collection_thread(self):
        # 由本线程独占采集/缓存的相机，start_camera_thread 不再重复 cache，
        # 以免覆盖与关节时间戳精确配对的帧
        self.inference_managed_cameras.add("hand_left")

        def update_inference_data():
            camera_name = "hand_left"
            # fps = 30  # hz
            fps = 31  # hz
            joint_values_by_timestamp = []
            next_update_camera_image_timestamp = 0.0
            while True:
                now = time.time()
                # update joint values
                arm_states, _ = self.robot.arm_joint_states()
                gripper_states, _ = self.robot.gripper_states()
                joint_values_by_timestamp.append((now, arm_states[:7] + [1 if gripper_states[0] > 0.5 else 0], arm_states[7:14] + [1 if gripper_states[1] > 0.5 else 0]))
                if now < next_update_camera_image_timestamp:
                    time.sleep(0.01)
                    continue
                # update camera images
                image, timestamp = self.camera.get_latest_image(camera_name)
                if image is not None:
                    timestamp_s = timestamp * 1e-9
                    wait_time_s = 0.001
                    if len(joint_values_by_timestamp) >= 2 and timestamp_s > self.robot_info.timestamp + wait_time_s:
                        next_update_camera_image_timestamp = timestamp_s + 1.0 / fps
                        if self.camera_images[camera_name] is None:
                            self.camera_images[camera_name] = [image, image]
                        else:
                            self.camera_images[camera_name] = [self.camera_images[camera_name][-1], image]

                        for i in range(len(joint_values_by_timestamp) - 1):
                            if joint_values_by_timestamp[i][0] <= timestamp_s and joint_values_by_timestamp[i + 1][0] > timestamp_s:
                                break
                        with self.robot_info.lock:
                            self.robot_info.timestamp = timestamp_s
                            self.robot_info.left_joint_values = joint_values_by_timestamp[i][1]
                            self.robot_info.right_joint_values = joint_values_by_timestamp[i][2]

                            left_arm_state = copy.deepcopy(joint_values_by_timestamp[i][1])
                            if not self.last_two_left_arm_joint_values:
                                self.last_two_left_arm_joint_values = [left_arm_state, left_arm_state]
                            else:
                                self.last_two_left_arm_joint_values = [self.last_two_left_arm_joint_values[-1], left_arm_state]
                        joint_values_by_timestamp = joint_values_by_timestamp[-1000:]
                time.sleep(0.01)  # 10ms

        inference_data_collection_thread = threading.Thread(target=update_inference_data, daemon=True)
        inference_data_collection_thread.start()
        
    def start_camera_thread(self):
        """启动相机图像更新线程"""
        def update_camera_images():
            while True:
                try:
                    # 获取各个相机的最新图像和内参
                    # 用快照遍历，避免显示区重建时（主线程改键）触发字典迭代错误
                    for camera_name in list(self.camera_images.keys()):
                        # 相机可能在遍历期间被取消勾选而移除
                        if camera_name not in self.camera_images:
                            continue
                        image, timestamp = self.camera.get_latest_image(camera_name)
                        if image is not None:
                            # print(f"获取到 {camera_name} 相机图像，图像 shape: {image.shape}")
                            if len(image.shape)== 3:
                                height, width = image.shape[:2]
                            else:
                                height, width = image.shape[0]
                            # print(f"获取到 {camera_name} 相机图像，图像 height, width: {height} {width}")
                            # 缓存最近两帧（与 hand_left 一致的 list 结构），
                            # 供推理 / VR 串流 / 保存图片 / 3D 点击等统一消费。
                            # 推理线程独占的相机（如 hand_left）不在此重复缓存，
                            # 避免覆盖其与关节时间戳精确配对的帧。
                            if (camera_name in self.camera_images
                                    and camera_name not in self.inference_managed_cameras):
                                prev = self.camera_images[camera_name]
                                if isinstance(prev, list) and len(prev) == 2:
                                    self.camera_images[camera_name] = [prev[-1], image.copy()]
                                else:
                                    self.camera_images[camera_name] = [image.copy(), image.copy()]

                            self.update_camera_display(camera_name, image)

                        # 获取相机内参（只需获取一次）
                        if self.camera_intrinsics.get(camera_name) is None:
                            try:
                                # 尝试获取相机信息 - 根据SDK文档，CosineCamera可能有不同的接口
                                camera_info = None
                                if hasattr(self.camera, 'get_camera_info'):
                                    camera_info = self.camera.get_camera_info(camera_name)
                                elif hasattr(self.camera, 'get_intrinsics'):
                                    camera_info = self.camera.get_intrinsics(camera_name)
                                
                                if camera_info:
                                    self.camera_intrinsics[camera_name] = camera_info
                                    print(f"获取到 {camera_name} 相机内参")
                                else:
                                    # 如果无法获取内参，使用默认参数
                                    self.camera_intrinsics[camera_name] = self.get_default_camera_intrinsics(camera_name)
                                    print(f"使用默认内参 for {camera_name}")
                            except Exception as info_e:
                                # 如果无法获取内参，使用默认参数
                                self.camera_intrinsics[camera_name] = self.get_default_camera_intrinsics(camera_name)
                                print(f"使用默认内参 for {camera_name}: {info_e}")

                    time.sleep(0.1)  # 100ms更新一次
                except Exception as e:
                    print(f"相机更新错误: {e}")
                    time.sleep(1)
        
        camera_thread = threading.Thread(target=update_camera_images, daemon=True)
        camera_thread.start()
        
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
                    for name in sorted(self.camera_images.keys()):
                        rgb = _to_rgb_uint8(self.camera_images.get(name))
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

    def start_status_thread(self):
        """启动状态更新线程"""
        def update_robot_status():
            while True:
                try:
                    # 获取手臂关节状态
                    arm_states, _ = self.robot.arm_joint_states()
                    if arm_states and len(arm_states) == 14:
                        for i, state in enumerate(arm_states):
                            if i < len(self.arm_status_labels):
                                self.root.after(0, lambda label=self.arm_status_labels[i], value=state: 
                                              label.config(text=f"{value:.3f}"))
                    self.arm_status_labels[-1].config(text=f"{self.wheel_controller.agv_angle:.3f}")

                    # 获取头部关节状态
                    head_states, _ = self.robot.head_joint_states()
                    if head_states and len(head_states) == 2:
                        for i, state in enumerate(head_states):
                            key = f"head_{i}"
                            if key in self.head_waist_status_labels and state is not None:
                                self.root.after(0, lambda label=self.head_waist_status_labels[key], value=state:
                                              label.config(text=f"{value:.3f}"))
                    
                    # 获取腰部关节状态
                    waist_states, _ = self.robot.waist_joint_states()
                    if waist_states and len(waist_states) == 2:
                        for i, state in enumerate(waist_states):
                            key = f"waist_{i}"
                            if key in self.head_waist_status_labels and state is not None:
                                self.root.after(0, lambda label=self.head_waist_status_labels[key], value=state:
                                              label.config(text=f"{value:.3f}"))
                    
                    # 获取夹爪状态（使用arm_joint_states的最后两个值）
                    # try:
                    #     # if arm_states and len(arm_states) >= 2:
                    #     #     # 假设夹爪状态是手臂关节的最后两个值
                    #     #     gripper_states = arm_states[-2:]
                    #     for i, state in enumerate(gripper_states):
                    #         if i < len(self.gripper_status_labels) and state is not None:
                    #             self.root.after(0, lambda label=self.gripper_status_labels[i], value=state:
                    #                           label.config(text=f"{value:.3f}"))
                    # except Exception as e:
                    #     print(f"夹爪状态更新错误: {e}")
                    
                    time.sleep(0.1)  # 100ms更新一次
                except Exception as e:
                    print(f"状态更新错误: {e}")
                    time.sleep(1)
        
        status_thread = threading.Thread(target=update_robot_status, daemon=True)
        status_thread.start()
        
    def save_camera_frame(self, camera_name):
        """将指定相机的当前帧保存为本地 jpg 图片。"""
        try:
            image = self.camera_images.get(camera_name)
            # 部分相机缓存的是最近两帧的列表，取最新一帧
            if isinstance(image, (list, tuple)):
                image = image[-1] if image else None
            if image is None or not isinstance(image, np.ndarray) or image.size == 0:
                self.status_text.set(f"保存失败：{camera_name} 相机暂无图像")
                return

            # 转换为可保存的 RGB PIL 图像（与显示逻辑保持一致）
            if len(image.shape) == 3 and image.shape[2] == 3:
                arr = image if image.dtype == np.uint8 else image.astype(np.uint8)
                if hasattr(self, "bgr_cameras") and camera_name in self.bgr_cameras:
                    arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(arr)
            else:
                # 深度 / 灰度图：归一化并做伪彩色映射后保存
                depth_2d = image[:, :, 0] if len(image.shape) == 3 else image
                if depth_2d.dtype == np.uint16:
                    valid_pixels = depth_2d[depth_2d > 0]
                    if valid_pixels.size > 0:
                        min_val, max_val = np.min(valid_pixels), np.max(valid_pixels)
                    else:
                        min_val, max_val = 0, 1
                    if max_val > min_val:
                        depth_norm = ((depth_2d.astype(np.float32) - min_val) /
                                      (max_val - min_val) * 255).astype(np.uint8)
                    else:
                        depth_norm = np.zeros_like(depth_2d, dtype=np.uint8)
                else:
                    depth_norm = depth_2d.astype(np.uint8)
                depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_TURBO)
                depth_rgb = cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(depth_rgb)

            save_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "saved_images")
            os.makedirs(save_dir, exist_ok=True)
            filename = f"{camera_name}_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
            filepath = os.path.join(save_dir, filename)
            pil_image.convert("RGB").save(filepath, "JPEG", quality=95)
            self.status_text.set(f"已保存图片: {filepath}")
            print(f"已保存图片: {filepath}")
        except Exception as e:
            self.status_text.set(f"保存图片失败: {e}")
            print(f"保存图片失败: {e}")

    def update_camera_display(self, camera_name, image):
        """更新相机图像显示"""
        try:
            if image is None or image.size == 0:
                return

            # print(f"{camera_name}: shape={image.shape}, dtype={image.dtype}")

            if not isinstance(image, np.ndarray):
                print(f"不支持类型: {type(image)}")
                return

            # ================= RGB / 彩色图 =================
            if len(image.shape) == 3 and image.shape[2] == 3:
                if image.dtype != np.uint8:
                    image = image.astype(np.uint8)

                # ✅ 不做默认转换，直接用
                image_rgb = image

                # 🔥 可选：针对特定相机做转换（如果你确认某些是BGR）
                if hasattr(self, "bgr_cameras") and camera_name in self.bgr_cameras:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                pil_image = Image.fromarray(image_rgb)

            # ================= 深度 / 灰度 =================
            elif len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):

                depth_2d = image[:, :, 0] if len(image.shape) == 3 else image

                # ========= 归一化 =========
                if depth_2d.dtype == np.uint16:
                    valid_pixels = depth_2d[depth_2d > 0]

                    if valid_pixels.size > 0:
                        min_val = np.min(valid_pixels)
                        max_val = np.max(valid_pixels)
                    else:
                        min_val, max_val = 0, 1

                    if max_val > min_val:
                        depth_norm = ((depth_2d.astype(np.float32) - min_val) /
                                    (max_val - min_val) * 255).astype(np.uint8)
                    else:
                        depth_norm = np.zeros_like(depth_2d, dtype=np.uint8)
                else:
                    depth_norm = depth_2d.astype(np.uint8)

                # ========= 颜色映射 =========
                depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_TURBO)

                # ⚠️ 这里必须转（因为 applyColorMap 一定是 BGR）
                depth_rgb = cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB)

                pil_image = Image.fromarray(depth_rgb)
                # 原始深度数据的缓存改由 start_camera_thread 统一以 list 结构维护

            else:
                print(f"不支持shape: {image.shape}")
                return

            # ================= resize =================
            # 使用按排版动态计算的尺寸，避免多相机时画面超出显示区
            tile_w, tile_h = getattr(self, "camera_tile_size", (320, 240))
            pil_image = pil_image.resize((tile_w, tile_h), Image.Resampling.LANCZOS)


            photo = ImageTk.PhotoImage(pil_image)

            if camera_name in self.camera_labels:
                self.root.after(
                    0,
                    lambda label=self.camera_labels[camera_name], img=photo:
                    self.update_label_image(label, img)
                )

        except Exception as e:
            print(f"更新相机显示错误: {e}")
    
    def update_label_image(self, label, photo):
        """更新标签图像"""
        try:
            label.config(image=photo)
            label.image = photo  # 保持引用
        except tk.TclError:
            # 标签可能在重建显示区时已被销毁，忽略这次延迟回调
            pass
    
    def show_status(self, message, message_type="info"):
        """在状态栏显示消息"""
        timestamp = time.strftime("%H:%M:%S")
        status_message = f"[{timestamp}] {message}"
        self.status_text.set(status_message)
        print(f"[{message_type.upper()}] {message}")
    
    def clear_status(self):
        """清除状态栏"""
        self.status_text.set("就绪")
    
    def get_default_camera_intrinsics(self, camera_name):
        """获取默认相机内参（当无法从SDK获取时）"""
        # 基于典型RGBD相机参数的默认值
        defaults = {
            "head": {
                'width': 1280, 'height': 800,
                'fx': 900.0, 'fy': 900.0,
                'cx': 640.0, 'cy': 400.0,
                'distortion': [0.0, 0.0, 0.0, 0.0, 0.0]
            },
            "head_depth": {
                'width': 1280, 'height': 800,
                'fx': 900.0, 'fy': 900.0,
                'cx': 640.0, 'cy': 400.0,
                'distortion': [0.0, 0.0, 0.0, 0.0, 0.0]
            },
            "hand_left": {
                'width': 320, 'height': 240,
                'fx': 300.0, 'fy': 300.0,
                'cx': 160.0, 'cy': 120.0,
                'distortion': [0.0, 0.0, 0.0, 0.0, 0.0]
            },
            "hand_right": {
                'width': 320, 'height': 240,
                'fx': 300.0, 'fy': 300.0,
                'cx': 160.0, 'cy': 120.0,
                'distortion': [0.0, 0.0, 0.0, 0.0, 0.0]
            },
            "head_center_fisheye": {
                'width': 640, 'height': 480,
                'fx': 400.0, 'fy': 400.0,
                'cx': 320.0, 'cy': 240.0,
                'distortion': [0.1, -0.1, 0.0, 0.0, 0.0]
            }
        }
        return defaults.get(camera_name, defaults["head"])
    
    def pixel_to_3d_coordinate(self, camera_name, pixel_x, pixel_y, depth_value):
        """将像素坐标和深度值转换为3D空间坐标"""
        try:
            # 获取相机内参
            intrinsics = self.camera_intrinsics.get(camera_name)
            if not intrinsics:
                print(f"无法获取 {camera_name} 的相机内参")
                return [None, None, None]
            
            # 提取内参
            fx = intrinsics.get('fx', 615.0)
            fy = intrinsics.get('fy', 615.0)
            cx = intrinsics.get('cx', 320.0)
            cy = intrinsics.get('cy', 240.0)
            
            # 检查深度值是否有效
            if depth_value is None or depth_value <= 0:
                print(f"无效的深度值: {depth_value}")
                return [None, None, None]
            # 将深度值从毫米转换为米
            depth_meters = depth_value * self.depth_scale
            
            # 使用针孔相机模型计算3D坐标
            # X = (u - cx) * Z / fx
            # Y = (v - cy) * Z / fy
            # Z = depth
            x = (pixel_x - cx) * depth_meters / fx
            y = (pixel_y - cy) * depth_meters / fy
            z = depth_meters
            
            return [x, y, z]
            
        except Exception as e:
            print(f"像素到3D坐标转换失败: {e}")
            return [None, None, None]
    
    def _latest_camera_frame(self, camera_name):
        """从 camera_images 取指定相机的最新一帧。

        camera_images 内所有相机统一为「最近两帧」的 list 结构，
        此处返回最新一帧（list[-1]）；兼容历史的单帧 ndarray。
        """
        entry = self.camera_images.get(camera_name)
        if isinstance(entry, (list, tuple)):
            return entry[-1] if entry else None
        return entry

    def get_depth_at_pixel(self, camera_name, pixel_x, pixel_y):
        """获取指定像素位置的深度值"""
        try:
            # 获取深度图像
            if camera_name == "head":
                # 如果是RGB相机，尝试获取对应的深度相机图像
                depth_image = self._latest_camera_frame("head_depth")
                if depth_image is None:
                    print("无法获取深度图像")
                    return None
            elif camera_name == "head_depth":
                depth_image = self._latest_camera_frame("head_depth")
            else:
                # 手部相机可能没有深度信息
                print(f"{camera_name} 没有深度信息")
                return None
            
            if depth_image is None:
                print("depth_image is None")
                return None
            
            # 确保图像是numpy数组
            if not isinstance(depth_image, np.ndarray):
                print("深度图像格式错误")
                return None
            
            # 获取图像尺寸
            height, width = depth_image.shape[:2]
            print(f"深度图像尺寸: {width}x{height}, 请求坐标: ({pixel_x}, {pixel_y})")
            
            # 确保坐标在图像范围内
            if 0 <= pixel_x < width and 0 <= pixel_y < height:
                # 获取深度值
                depth_value = depth_image[pixel_y, pixel_x]
                
                # 处理不同类型的深度值
                if isinstance(depth_value, (int, float, np.number)):
                    print(f"深度信息 : {float(depth_value)}")
                    return float(depth_value)
                elif isinstance(depth_value, np.ndarray):
                    # 如果是数组，取第一个元素
                    if depth_value.size > 0:
                        print(f"深度信息是数组，取第一个元素 : {float(depth_value.flat[0])} 数值size: {depth_value.size}")
                        return float(depth_value.flat[0])
                    else:
                        print("深度数组为空")
                        return None
                else:
                    print(f"深度值类型错误: {type(depth_value)}, 值: {depth_value}")
                    return None
            else:
                print(f"像素坐标 ({pixel_x}, {pixel_y}) 超出图像范围 ({width}, {height})")
                return None
                
        except Exception as e:
            print(f"获取深度值失败: {e}")
            return None
    
    def handle_image_click(self, camera_name, event):
        """处理图像点击事件"""
        try:
            # 获取点击的像素坐标（基于显示尺寸320x240）
            display_x = event.x
            display_y = event.y
            
            # 根据相机类型使用已知的实际尺寸
            # 这些尺寸来自之前的日志信息
            if camera_name == "head" or camera_name == "head_depth":
                original_width, original_height = 1280, 800
            elif camera_name == "hand_left" or camera_name == "hand_right":
                original_width, original_height = 848, 480
            else:
                # 默认尺寸
                original_width, original_height = 640, 480

            if camera_name == "head":
                display_width, display_height = 800, 600
            else:
                display_width, display_height = 256, 195
            
            # 坐标映射：将显示坐标转换为原始图像坐标
            pixel_x = int(display_x * original_width / display_width)
            pixel_y = int(display_y * original_height / display_height)
            
            print(f"点击了 {camera_name} 图像，显示坐标: ({display_x}, {display_y}) -> 原始坐标: ({pixel_x}, {pixel_y}), 原始尺寸: {original_width}x{original_height}")
            
            
            # 获取深度值
            depth_value = self.get_depth_at_pixel(camera_name, pixel_x, pixel_y)
            if depth_value is None or depth_value <= 0:
                self.show_status("无法获取该位置的深度信息或深度值无效", "warning")
                return
            
            # 转换为3D坐标
            coordinates_3d = self.pixel_to_3d_coordinate(camera_name, pixel_x, pixel_y, depth_value)
            self.coordinates_3d = coordinates_3d
            if coordinates_3d and coordinates_3d[0] is not None and coordinates_3d[1] is not None and coordinates_3d[2] is not None:
                x, y, z = coordinates_3d

                # --- 新增：转换为世界坐标系 ---
                try:
                    # 获取当前机器人关节状态
                    arm_states, _ = self.robot.arm_joint_states()
                    head_states, _ = self.robot.head_joint_states()
                    waist_states, _ = self.robot.waist_joint_states()

                    # 提取关键关节角度 [腰部升降(m), 腰部俯仰(rad), 头部偏航(rad), 头部俯仰(rad)]
                    # 注意：waist_states[1] 是高度(cm)，转换为米
                    joint_angles = [
                        (waist_states[1] * 0.01 if waist_states and len(waist_states) >= 2 else 0.0),
                        (waist_states[0] if waist_states and len(waist_states) >= 1 else 0.0),
                        (head_states[0] if head_states and len(head_states) >= 1 else 0.0),
                        (head_states[1] if head_states and len(head_states) >= 2 else 0.0)
                    ]

                    world_coords = self.transformer.pixel_to_world(
                        pixel_x, pixel_y, depth_value,
                        self.camera_intrinsics[camera_name],
                        joint_angles
                    )
                    world_x, world_y, world_z = world_coords
                except Exception as world_e:
                    print(f"世界坐标转换失败: {world_e}")
                    world_x, world_y, world_z = None, None, None

                status_msg = (f"相机: {camera_name}, 像素: ({pixel_x}, {pixel_y}), "
                             f"深度: {depth_value:.1f}mm, 3D:({x:.3f}, {y:.3f}, {z:.3f})m, "
                             f"世界:({f'{world_x:.3f}' if world_x is not None else 'N/A'}, "
                             f"{f'{world_y:.3f}' if world_y is not None else 'N/A'}, "
                             f"{f'{world_z:.3f}' if world_z is not None else 'N/A'})m")
                self.show_status(status_msg, "info")
                print(f"点击了 {camera_name} 图像，像素: ({pixel_x}, {pixel_y}), 世界坐标: {world_coords}")
                # 存储点击坐标用于后续处理
                self.click_coordinates.append({
                    'camera': camera_name,
                    'pixel': (pixel_x, pixel_y),
                    'depth': depth_value,
                    '3d': coordinates_3d,
                    'world_3d': world_coords if 'world_coords' in locals() else None,
                    'timestamp': time.time()
                })
            else:
                self.show_status("3D坐标转换失败", "error")
                self.show_status(f"处理点击事件失败: {e}", "error")
                
        except Exception as e:
            print(f"处理图像点击事件失败: {e}")
            self.show_status(f"处理点击事件失败: {e}", "error")
    
    def transform_to_robot_base(self, camera_coords, camera_name):
        """将相机坐标系下的3D坐标转换到机器人基座坐标系"""
        try:
            # 这里需要实现相机到机器人基座的外参变换
            # 由于需要相机的外参矩阵，这里先返回原始坐标并给出提示
            print(f"需要将 {camera_name} 的相机坐标转换到机器人基座坐标系")
            print("这需要相机的外参矩阵（相机到机器人基座的变换矩阵）")
            return camera_coords
        except Exception as e:
            print(f"坐标系转换失败: {e}")
            return camera_coords
    
    def move_waist_lift(self, position):
        """控制腰部升降 - 使用move_waist函数，height单位为厘米"""
        try:
            # 获取当前腰部状态  [pitch(rad), height(cm)]
            waist_states, _ = self.robot.waist_joint_states()
            if waist_states and len(waist_states) >= 2:
                current_pitch = waist_states[0] if waist_states[0] is not None else 0.0  # 俯仰角（弧度）
                current_height = (waist_states[1] * 100) if waist_states[1] is not None else 0.0 # 高度（厘米）
                if position == 0.0:  # 复位
                    target_height = 0.0  # 0厘米
                else:
                    # 基于当前高度进行相对控制，步长为5厘米
                    target_height = np.clip(current_height + (position * 5.0), 0.0, 100.0)  # 限制在±15厘米
                
                # 使用move_waist函数，参数为[pitch(rad), height(cm)]
                self.robot.move_waist([current_pitch, target_height])
                
                print(f"腰部升降控制: 目标高度={target_height:.1f}cm, 当前高度={current_height:.1f}cm, 变化={target_height - current_height:.1f}cm")
                                
                # 验证控制结果
                time.sleep(0.5)
                new_states, _ = self.robot.waist_joint_states()
                if new_states and new_states[1] is not None and abs(new_states[1] - current_height) < 0.5:  # 0.5cm容差
                    print("⚠️ 腰部升降控制可能无效，高度未变化")
            else:
                self.show_status("无法获取腰部关节状态", "error")
        except Exception as e:
            self.show_status(f"腰部升降控制失败: {e}", "error")
    
    def move_waist_pitch(self, position):
        """控制腰部俯仰 - 使用move_waist函数，pitch单位为弧度"""
        try:
            # 获取当前腰部状态 [pitch(rad), height(cm)]
            waist_states, _ = self.robot.waist_joint_states()
            if waist_states and len(waist_states) >= 2:
                current_pitch = waist_states[0] if waist_states[0] is not None else 0.0  # 俯仰角（弧度）
                current_height = waist_states[1] if waist_states[1] is not None else 0.0  # 高度（厘米）
                if position == 0.0:  # 复位
                    target_pitch = 0.0  # 0弧度
                else:
                    # 基于当前俯仰角进行相对控制，步长为0.2弧度（约11.5度）
                    target_pitch = np.clip(current_pitch + (position * 0.2), -0.8, 0.8)  # 限制在±0.8弧度
                
                # 使用move_waist函数，参数为[pitch(rad), height(cm)]
                self.robot.move_waist([target_pitch, current_height])
                
                print(f"腰部俯仰控制: 目标俯仰={target_pitch:.3f}rad, 当前俯仰={current_pitch:.3f}rad, 变化={target_pitch - current_pitch:.3f}rad")
                                
                # 验证控制结果
                time.sleep(0.5)
                new_states, _ = self.robot.waist_joint_states()
                if new_states and new_states[0] is not None and abs(new_states[0] - current_pitch) < 0.02:  # 0.02弧度容差
                    print("⚠️ 腰部俯仰控制可能无效，角度未变化")
            else:
                self.show_status("无法获取腰部关节状态", "error")
        except Exception as e:
            self.show_status(f"腰部俯仰控制失败: {e}", "error")
    
    def move_head_yaw(self, position):
        """控制头部偏航"""
        try:
            self.head_yaw_pos = np.clip(position, -1.0, 1.0)
            self.robot.move_head([self.head_yaw_pos, self.head_pitch_pos])
            print(f"头部偏航移动到: {self.head_yaw_pos}")
        except Exception as e:
            self.show_status(f"头部偏航控制失败: {e}", "error")
    
    def move_head_pitch(self, position):
        """控制头部俯仰"""
        try:
            self.head_pitch_pos = np.clip(position, -0.5, 0.5)
            self.robot.move_head([self.head_yaw_pos, self.head_pitch_pos])
            print(f"头部俯仰移动到: {self.head_pitch_pos}")
        except Exception as e:
            self.show_status(f"头部俯仰控制失败: {e}", "error")
    
    def move_gripper(self, side, position):
        """控制夹爪"""
        try:
            if side == "left":
                self.left_gripper_pos = np.clip(position, 0.0, 1.0)
                gripper_cmd = [self.left_gripper_pos, self.right_gripper_pos]
            else:
                self.right_gripper_pos = np.clip(position, 0.0, 1.0)
                gripper_cmd = [self.left_gripper_pos, self.right_gripper_pos]
            
            self.robot.move_gripper(gripper_cmd)
            print(f"{side}夹爪移动到: {position}")
        except Exception as e:
            self.show_status(f"夹爪控制失败: {e}", "error")

    def move_arm_to_position(self, arm_side, position, orientation=None):
        """控制机械臂末端移动到指定位置 - 使用SDK的set_end_effector_pose_control接口
        Args:
            arm_side: 'left' 或 'right'
            position: [x, y, z] 目标位置（米）
            orientation: [roll, pitch, yaw] 目标姿态（弧度，可选）
        """
        try:
            # 将欧拉角转换为四元数
            if orientation is None:
                # 如果只提供位置，保持当前姿态
                orientation = [0.0, 0.0, 0.0]
            
            roll, pitch, yaw = orientation
            # 简化的欧拉角到四元数转换
            cy = np.cos(yaw * 0.5)
            sy = np.sin(yaw * 0.5)
            cp = np.cos(pitch * 0.5)
            sp = np.sin(pitch * 0.5)
            cr = np.cos(roll * 0.5)
            sr = np.sin(roll * 0.5)
            
            qw = cr * cp * cy + sr * sp * sy
            qx = sr * cp * cy - cr * sp * sy
            qy = cr * sp * cy + sr * cp * sy
            qz = cr * cp * sy - sr * sp * cy
            
            # 构建末端位姿字典
            pose_dict = {
                'x': position[0],
                'y': position[1], 
                'z': position[2],
                'qx': qx,
                'qy': qy,
                'qz': qz,
                'qw': qw
            }
            
            # 设置控制组
            control_group = [f"{arm_side}_arm"]
            
            # 使用SDK的set_end_effector_pose_control接口
            if arm_side == "left":
                self.robot_controller.set_end_effector_pose_control(
                    lifetime=2.0,  # 2秒有效时间
                    control_group=control_group,
                    left_pose=pose_dict,
                    right_pose=None
                )
            else:
                self.robot_controller.set_end_effector_pose_control(
                    lifetime=2.0,  # 2秒有效时间
                    control_group=control_group,
                    left_pose=None,
                    right_pose=pose_dict
                )
            
            print(f"{arm_side}臂末端绝对位姿控制: 位置={position}m, 姿态={orientation}rad")
            
        except Exception as e:
            self.show_status(f"机械臂位置控制失败: {e}", "error")
    
    def move_arm_relative(self, arm_side, delta_position, delta_orientation=None, time_step=1.0):
        """相对移动机械臂末端 - 使用DELTA_POSE控制模式
        Args:
            arm_side: 'left' 或 'right'
            delta_position: [dx, dy, dz] 相对位置变化（米）
            delta_orientation: [droll, dpitch, dyaw] 相对姿态变化（弧度，可选）
        """
        try:
            if delta_orientation is None:
                delta_orientation = [0.0, 0.0, 0.0]
            
            # 将欧拉角转换为四元数
            roll, pitch, yaw = delta_orientation
            cy = np.cos(yaw * 0.5)
            sy = np.sin(yaw * 0.5)
            cp = np.cos(pitch * 0.5)
            sp = np.sin(pitch * 0.5)
            cr = np.cos(roll * 0.5)
            sr = np.sin(roll * 0.5)
            
            qw = cr * cp * cy + sr * sp * sy
            qx = sr * cp * cy - cr * sp * sy
            qy = cr * sp * cy + sr * cp * sy
            qz = cr * cp * sy - sr * sp * cy
            
            # 构建相对位姿字典（使用很小的数值作为相对变化）
            pose_dict = {
                'x': delta_position[0],
                'y': delta_position[1], 
                'z': delta_position[2],
                'qx': qx,
                'qy': qy,
                'qz': qz,
                'qw': qw
            }
            
            # 设置控制组
            control_group = [f"{arm_side}_arm"]
            
            # 使用DELTA_POSE模式进行相对控制
            # 注意：DELTA_POSE模式下，实际实现可能需要使用轨迹跟踪控制
            print(f"{arm_side}臂相对移动: 位置变化={delta_position}m, 姿态变化={delta_orientation}rad")
            
            # 由于DELTA_POSE是控制模式，我们需要通过轨迹跟踪控制来实现
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
            
            # 构建6维动作数据（位置3 + 姿态3）
            action_data = delta_position + delta_orientation
            robot_actions = [{
                f"{arm_side}_arm": {
                    "action_data": action_data,
                    "control_type": "DELTA_POSE"
                }
            }]
            
            
            # 使用轨迹跟踪控制
            self.robot_controller.trajectory_tracking_control(
                int(time.time() * 1e9),
                robot_states,
                robot_actions,
                "base_link",
                time_step  # 较短的执行时间
            )
            time.sleep(time_step)
        except Exception as e:
            print(f"机械臂相对移动失败: {e}")
            self.show_status(f"机械臂相对移动失败: {e}", "error")
    
    def set_arm_preset(self, preset_name):
        """设置机械臂预设位置"""
        try:
            # 获取当前状态
            arm_states, _ = self.robot.arm_joint_states()
            head_states, _ = self.robot.head_joint_states()
            waist_states, _ = self.robot.waist_joint_states()
            
            robot_states = {
                "head": head_states,
                "waist": waist_states,
                "arm": arm_states
            }
            
            if preset_name == "parallel":
                # 双臂平举 - 使用相对移动逐步达到目标位置
                robot_actions = [
                    {
                        "left_arm": {"action_data": [0.0, 0.0, 0.1, 0.0, 0.0, 0.0], "control_type": "DELTA_POSE"},
                        "right_arm": {"action_data": [0.0, 0.0, 0.1, 0.0, 0.0, 0.0], "control_type": "DELTA_POSE"}
                    }
                ]
            elif preset_name == "down":
                # 双臂下垂
                robot_actions = [
                    {
                        "left_arm": {"action_data": [0.0, 0.0, -0.1, 0.0, 0.0, 0.0], "control_type": "DELTA_POSE"},
                        "right_arm": {"action_data": [0.0, 0.0, -0.1, 0.0, 0.0, 0.0], "control_type": "DELTA_POSE"}
                    }
                ]
            elif preset_name == "forward":
                # 双臂前伸
                robot_actions = [
                    {
                        "left_arm": {"action_data": [0.1, 0.0, 0.0, 0.0, 0.0, 0.0], "control_type": "DELTA_POSE"},
                        "right_arm": {"action_data": [0.1, 0.0, 0.0, 0.0, 0.0, 0.0], "control_type": "DELTA_POSE"}
                    }
                ]
            else:
                return
            
            self.robot_controller.trajectory_tracking_control(
                int(time.time() * 1e9),
                robot_states,
                robot_actions,
                "base_link",
                1.0
            )
            
            print(f"机械臂预设位置已设置: {preset_name}")
            
        except Exception as e:
            self.show_status(f"设置机械臂预设位置失败: {e}", "error")

    def execute_end_effector_control(self):
        """执行末端位姿控制"""
        try:
            # 获取输入值
            arm_side = self.arm_side_var.get()
            x = float(self.x_var.get())
            y = float(self.y_var.get())
            z = float(self.z_var.get())
            roll = float(self.roll_var.get())
            pitch = float(self.pitch_var.get())
            yaw = float(self.yaw_var.get())

            # 验证输入值
            if not (-1.0 <= x <= 1.0 and -1.0 <= y <= 1.0 and 0.0 <= z <= 1.5):
                self.show_status("位置值超出安全范围！X/Y: ±1.0m, Z: 0-1.5m", "warning")
                return

            if not (-3.14 <= roll <= 3.14 and -3.14 <= pitch <= 3.14 and -3.14 <= yaw <= 3.14):
                self.show_status("姿态值超出安全范围！±3.14弧度", "warning")
                return

            position = [x, y, z]
            orientation = [roll, pitch, yaw]

            # 检查是否启用安全模式
            if self.safe_mode_var.get():
                self.show_status("安全模式启用，正在规划轨迹...", "info")

                # 获取当前关节状态
                arm_states, _ = self.robot.arm_joint_states()
                if not arm_states or len(arm_states) < 14:
                    self.show_status("无法获取当前关节状态，规划失败", "error")
                    return

                start_joints = arm_states[:7] if arm_side == "left" else arm_states[7:14]

                # 规划路径 (关节空间)
                path = self.planner.plan_path(arm_side, start_joints, position)

                if path is None:
                    self.show_status("RRT 规划失败：未找到无碰撞路径", "error")
                    return

                # 平滑路径
                path = self.planner.smooth_path(path)

                # 执行轨迹：将关节路径转换为一系列动作
                self.show_status(f"规划成功，执行 {len(path)} 个路径点...", "success")

                # 这里通过轨迹跟踪控制执行- 简化实现：依次发送路径点
                # 实际应构建一个完整的 robot_actions 列表
                for i, joints in enumerate(path):
                    # 为了演示，我们将关节角转换为 SDK 能接受的动作
                    # 注意：SDK 轨迹跟踪通常需要目标状态和动作
                    # 这里采用一种简单的循环发送方式，或构建一个长轨迹
                    # 为简化，我们直接使用关节控制接口（如果SDK支持）或模拟轨迹

                    # 模拟轨迹发送 (实际应根据 SDK TrajectoryTrackingControl 接口构建)
                    # 这里我们将每个点作为一个小目标发送
                    # 为了不阻塞GUI，实际应在线程中执行
                    def run_trajectory(path_nodes, side):
                        try:
                            for node in path_nodes:
                                # 获取当前实时状态
                                arm_states, _ = self.robot.arm_joint_states()
                                head_states, _ = self.robot.head_joint_states()
                                waist_states, _ = self.robot.waist_joint_states()

                                robot_states = {
                                    "head": head_states if head_states else [0.0, 0.0],
                                    "waist": waist_states if waist_states else [0.0, 0.0],
                                    "arm": arm_states if arm_states else [0.0] * 14
                                }

                                # 构建关节空间控制动作
                                # 假设 SDK 的 trajectory_tracking_control 在 JOINT 模式下
                                # action_data 接受关节目标值
                                arm_key = f"{side}_arm"
                                robot_actions = [{
                                    arm_key: {
                                        "action_data": node,
                                        "control_type": "JOINT"
                                    }
                                }]

                                # 发送控制命令
                                self.robot_controller.trajectory_tracking_control(
                                    int(time.time() * 1e9),
                                    robot_states,
                                    robot_actions,
                                    "base_link",
                                    0.2  # 每个点执行时间 200ms
                                )
                                time.sleep(0.2)
                            self.show_status(f"{side}臂轨迹执行完毕", "success")
                        except Exception as traj_e:
                            print(f"轨迹执行错误: {traj_e}")
                            self.show_status(f"轨迹执行失败: {traj_e}", "error")

                    threading.Thread(target=run_trajectory, args=(path, arm_side)).start()
                    break # 仅启动一个线程

                return

            # 非安全模式：直接执行
            print(f"执行末端位姿控制: {arm_side}臂, 位置={position}m, 姿态={orientation}rad")
            self.move_arm_to_position(arm_side, position, orientation)
            self.show_status(f"{arm_side}臂末端位姿控制命令已发送", "success")

        except ValueError as e:
            self.show_status(f"输入值格式错误: {e}", "error")
        except Exception as e:
            self.show_status(f"末端位姿控制失败: {e}", "error")
    
    def get_current_pose(self):
        """获取当前末端位姿"""
        try:
            arm_side = self.arm_side_var.get()
            
            # 这里应该调用实际的SDK接口来获取当前末端位姿
            # 目前使用估算值
            current_pose = self.get_current_end_effector_pose(arm_side)
            
            if current_pose:
                pos = current_pose['position']
                orient = current_pose['orientation']
                self.show_status(f"{arm_side}臂当前位置: X={pos[0]:.3f}, Y={pos[1]:.3f}, Z={pos[2]:.3f}m, 姿态: Roll={orient[0]:.3f}, Pitch={orient[1]:.3f}, Yaw={orient[2]:.3f}rad", "info")
            else:
                self.show_status(f"无法获取{arm_side}臂当前位姿", "warning")
                
        except Exception as e:
            self.show_status(f"获取当前位姿失败: {e}", "error")   

    def get_current_end_effector_pose(self, arm_side):
        """获取当前末端执行器位姿"""
        try:
            # 这里需要根据实际的SDK接口来获取末端位姿
            # 目前使用手臂关节状态来估算
            arm_states, _ = self.robot.arm_joint_states()
            if arm_states and len(arm_states) >= 7:
                if arm_side == "left":
                    # 左臂的关节状态（假设前7个关节是左臂）
                    joint_angles = arm_states[:7]
                else:
                    # 右臂的关节状态（假设后7个关节是右臂）
                    joint_angles = arm_states[7:14] if len(arm_states) >= 14 else arm_states[:7]
                
                # 这里应该使用正运动学计算末端位姿
                # 目前返回一个估算值
                print(f"{arm_side}臂当前关节角度: {joint_angles[:3]}")
                
                # 返回估算的末端位置（需要根据实际机器人模型调整）
                return {
                    'position': [0.3, 0.2 if arm_side == "left" else -0.2, 0.6],
                    'orientation': [0.0, 0.0, 0.0, 1.0]  # 四元数
                }
            else:
                print(f"无法获取{arm_side}臂关节状态")
                return None
                
        except Exception as e:
            print(f"获取{arm_side}臂末端位姿失败: {e}")
            return None

    def update_coordinate_display(self):
        """更新坐标显示"""
        try:
            if hasattr(self, 'click_coordinates') and self.click_coordinates:
                # 显示最近的几条记录
                recent_records = self.click_coordinates[-3:]  # 最近3条
                display_text = "最近获取的3D坐标:\n"
                
                for i, record in enumerate(reversed(recent_records)):
                    try:
                        camera = record.get('camera', 'unknown')
                        pixel = record.get('pixel', (0, 0))
                        depth = record.get('depth', 0)
                        coords = record.get('3d', [0, 0, 0])
                        
                        # 安全地获取坐标值
                        x = coords[0] if coords and len(coords) > 0 else 0
                        y = coords[1] if coords and len(coords) > 1 else 0  
                        z = coords[2] if coords and len(coords) > 2 else 0
                        
                        display_text += f"{i+1}. {camera}: ({pixel[0]},{pixel[1]}) "
                        display_text += f"深度:{depth:.0f}mm -> "
                        display_text += f"3D:({x:.3f},{y:.3f},{z:.3f})\n"
                    except Exception as record_e:
                        print(f"处理坐标记录失败: {record_e}")
                        display_text += f"{i+1}. 记录格式错误\n"
                # 显示总记录数
                display_text += f"\n总计: {len(self.click_coordinates)} 条记录"
            else:
                display_text = "暂无坐标记录\n点击图像获取3D坐标"
            
            if hasattr(self, 'coordinate_display'):
                self.coordinate_display.config(text=display_text)
            
        except Exception as e:
            print(f"更新坐标显示失败: {e}")
            if hasattr(self, 'coordinate_display'):
                self.coordinate_display.config(text="显示更新失败")

    def reset_robot(self):
        """机器人复位"""
        try:
            # 直接执行复位，不显示确认对话框
            self.robot.reset()
            # 重置内部状态
            self.waist_lift_pos = 0.0
            self.waist_pitch_pos = 0.0
            self.head_yaw_pos = 0.0
            self.head_pitch_pos = 0.0
            self.left_gripper_pos = 0.0
            self.right_gripper_pos = 0.0
            self.show_status("机器人已复位到初始位置", "success")
        except Exception as e:
            self.show_status(f"机器人复位失败: {e}", "error")
    
    def _bind_camera_click(self, camera_name):
        """绑定相机图像点击事件"""
        def on_click(event):
            if self.rgbd_enabled_var.get():
                self.handle_image_click(camera_name, event)
        
        if camera_name in self.camera_labels:
            self.camera_labels[camera_name].bind('<Button-1>', on_click)
            # 添加鼠标悬停效果
            self.camera_labels[camera_name].bind('<Enter>', 
                lambda e: self.camera_labels[camera_name].config(cursor="hand2" if self.rgbd_enabled_var.get() else ""))
            self.camera_labels[camera_name].bind('<Leave>', 
                lambda e: self.camera_labels[camera_name].config(cursor=""))
    
    def toggle_rgbd_mode(self):
        """切换RGBD模式"""
        enabled = self.rgbd_enabled_var.get()
        # 更新所有相机的鼠标样式
        for camera_name in self.camera_labels:
            cursor = "hand2" if enabled else ""
            self.camera_labels[camera_name].config(cursor=cursor)
        
        status = "启用" if enabled else "禁用"
        print(f"RGBD 3D坐标获取功能已{status}")
    
    def on_camera_selection_change(self, event=None):
        """相机选择改变时的处理"""
        self.current_camera_for_3d = self.target_camera_var.get()
        print(f"当前目标相机切换为: {self.current_camera_for_3d}")

    def clear_coordinate_records(self):
        """清除坐标记录"""
        self.click_coordinates.clear()
        self.show_status("已清除所有坐标记录", "info")
    
    def export_coordinates_to_file(self):
        """导出坐标记录到文件"""
        try:
            if not self.click_coordinates:
                self.show_status("没有坐标记录可以导出", "warning")
                return
            
            import json
            from datetime import datetime
            
            filename = f"coordinates_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            # 准备导出数据
            export_data = []
            for record in self.click_coordinates:
                export_data.append({
                    'camera': record['camera'],
                    'pixel_x': record['pixel'][0],
                    'pixel_y': record['pixel'][1],
                    'depth_mm': record['depth'],
                    'x_m': record['3d'][0],
                    'y_m': record['3d'][1],
                    'z_m': record['3d'][2],
                    'timestamp': record['timestamp']
                })
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            self.show_status(f"坐标记录已导出到: {filename}", "success")
            
        except Exception as e:
            self.show_status(f"导出失败: {e}", "error")
    
    def on_closing(self):
        """窗口关闭时的处理：依次释放各资源，最后强制退出进程。

        每个资源单独 try/except，避免前一个清理失败阻断后续；
        最后用 os._exit() 兜底，绕过被 DDS/rclpy 原生线程卡住的解释器退出。
        """
        # 防止重复进入（信号 + 窗口关闭可能同时触发）
        if getattr(self, "_is_closing", False):
            return
        self._is_closing = True

        for label, fn in [
            ("VR 串流服务器", self.dummy_server.stop),
            ("相机", self.camera.close),
            ("机器人", self.robot.shutdown),
            ("ROS 节点", self.wheel_controller.destroy_node),
            ("rclpy", rclpy.shutdown),
        ]:
            try:
                fn()
            except Exception as e:
                print(f"关闭{label}出错: {e}")

        try:
            self.root.destroy()
        except Exception as e:
            print(f"销毁窗口出错: {e}")
        finally:
            os._exit(0)

def main():
    root = tk.Tk()
    app = RobotControlGUI(root)  # 样式由 _setup_styles 统一配置
    root.protocol("WM_DELETE_WINDOW", app.on_closing)

    # Ctrl+C：注册 SIGINT 处理器，转交主线程的 on_closing
    signal.signal(signal.SIGINT, lambda *_: app.on_closing())

    # 心跳：周期性把控制权交还给 Python 解释器，
    # 否则 Tk 的 C 事件循环在空闲时不会处理挂起的 SIGINT。
    def _tick():
        root.after(200, _tick)
    root.after(200, _tick)

    try:
        root.mainloop()
    except KeyboardInterrupt:
        app.on_closing()


if __name__ == "__main__":
    main()