import copy
import tkinter as tk
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
from scipy.spatial.transform import Rotation as R
from camera_pose import compute_point_B_world
from control_wheel_example import WheelController
from robot_info_server import create_robot_info_http_server, RobotInfo
from constants import *

PC4080_HOST = "10.12.11.144"
PC4080_PORT = 9000

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


class RobotControlGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("智元G1机器人控制界面")
        self.root.geometry("1400x900")
        
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
            "hand_right": None,  # cache latest two images for inference
            "head": None,
            "head_depth": None,  # cache latest one image
            "head_center_fisheye": None
        }
        self.last_two_left_arm_joint_values = []

        # 相机内参缓存
        self.camera_intrinsics = {
            "hand_left": None,
            "hand_right": None, 
            "head": None,
            "head_depth": None,
            "head_center_fisheye": None
        }

        # update robot_info
        self.robot_info = RobotInfo()
        create_robot_info_http_server(self.robot_info)

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
        self.start_camera_thread()
        self.start_status_thread()
        
    def setup_ui(self):
        """设置用户界面"""
        # 主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 顶部标题
        title_label = ttk.Label(main_frame, text="智元G1机器人控制面板")
        title_label.pack(pady=10)
        
        # 创建左右分栏
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # 左侧面板：相机图像
        self.setup_camera_panel(left_frame)
        
        # 右侧面板：状态和控制
        self.setup_status_panel(right_frame)


        jog_frame = ttk.LabelFrame(right_frame, text="Jog控制面板")
        jog_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        # 创建标签页
        notebook = ttk.Notebook(jog_frame)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.setup_pick_and_place_panel(notebook)
        self.setup_control_panel(notebook)

        # 底部状态栏
        status_frame = ttk.Frame(self.root, relief=tk.SUNKEN)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=2)
        
        self.status_label = ttk.Label(status_frame, textvariable=self.status_text, anchor=tk.W)
        self.status_label.pack(side=tk.LEFT, padx=5, pady=2)
        
        # 清除状态按钮
        ttk.Button(status_frame, text="清除", 
                  command=lambda: self.status_text.set("就绪")).pack(side=tk.RIGHT, padx=5)
        
    def setup_camera_panel(self, parent):
        """设置相机图像面板"""
        camera_frame = ttk.LabelFrame(parent, text="相机图像")
        camera_frame.pack(fill=tk.BOTH, expand=True)
        
        # 添加RGBD坐标转换控制面板
        rgbd_control_frame = ttk.Frame(camera_frame)
        rgbd_control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(rgbd_control_frame, text="RGBD坐标转换:").pack(side=tk.LEFT, padx=5)
        self.rgbd_enabled_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(rgbd_control_frame, text="启用3D坐标获取", 
                         variable=self.rgbd_enabled_var,
                         command=self.toggle_rgbd_mode).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(rgbd_control_frame, text="目标相机:").pack(side=tk.LEFT, padx=5)
        self.target_camera_var = tk.StringVar(value="head")
        camera_combo = ttk.Combobox(rgbd_control_frame, textvariable=self.target_camera_var,
                                   values=["head", "head_depth", "hand_left", "hand_right"],
                                   state="readonly", width=12)
        camera_combo.pack(side=tk.LEFT, padx=5)
        camera_combo.bind('<<ComboboxSelected>>', self.on_camera_selection_change)
        
        ttk.Button(rgbd_control_frame, text="清除记录", 
                  command=self.clear_coordinate_records).pack(side=tk.LEFT, padx=5)
        
        # 创建3x2网格显示相机
        self.camera_labels = {}
        
        # 第一行：左手、右手和头部鱼眼相机
        top_frame = ttk.Frame(camera_frame)
        top_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # 左手相机
        left_frame = ttk.Frame(top_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        ttk.Label(left_frame, text="左手相机").pack()
        self.camera_labels["hand_left"] = ttk.Label(left_frame, borderwidth=2, relief="solid")
        self.camera_labels["hand_left"].pack(pady=5)
        self._bind_camera_click("hand_left")
        
        # 右手相机
        right_frame = ttk.Frame(top_frame)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        ttk.Label(right_frame, text="右手相机").pack()
        self.camera_labels["hand_right"] = ttk.Label(right_frame, borderwidth=2, relief="solid")
        self.camera_labels["hand_right"].pack(pady=5)
        self._bind_camera_click("hand_right")

        depth_frame = ttk.Frame(top_frame)
        depth_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        ttk.Label(depth_frame, text="头部深度相机").pack()
        self.camera_labels["head_depth"] = ttk.Label(depth_frame, borderwidth=2, relief="solid")
        self.camera_labels["head_depth"].pack(pady=5)
        self._bind_camera_click("head_depth")
        
        # 第二行：头部RGB和深度相机
        bottom_frame = ttk.Frame(camera_frame)
        bottom_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # 头部RGB相机
        head_frame = ttk.Frame(bottom_frame)
        head_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        ttk.Label(head_frame, text="头部RGB相机 (点击获取3D坐标)").pack()
        self.camera_labels["head"] = ttk.Label(head_frame, borderwidth=2, relief="solid")
        self.camera_labels["head"].pack(pady=5)
        self._bind_camera_click("head")
        
    def setup_status_panel(self, parent):
        """设置状态面板"""
        status_frame = ttk.LabelFrame(parent, text="关节状态")
        status_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # 创建标签页
        notebook = ttk.Notebook(status_frame)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 手臂关节状态
        arm_frame = ttk.Frame(notebook)
        notebook.add(arm_frame, text="手臂关节")
        
        self.arm_status_labels = []
        for i in range(14):
            frame = ttk.Frame(arm_frame)
            frame.pack(fill=tk.X, padx=5, pady=2)
            name_label = ttk.Label(frame, text=f"{self.joint_names['arm'][i]}:", width=15)
            name_label.pack(side=tk.LEFT)
            value_label = ttk.Label(frame, text="0.000", width=10)
            value_label.pack(side=tk.LEFT)
            self.arm_status_labels.append(value_label)
        # agv angle
        frame = ttk.Frame(arm_frame)
        frame.pack(fill=tk.X, padx=5, pady=2)
        name_label = ttk.Label(frame, text="agv_angle:", width=15)
        name_label.pack(side=tk.LEFT)
        value_label = ttk.Label(frame, text="0.000", width=10)
        value_label.pack(side=tk.LEFT)
        self.arm_status_labels.append(value_label)
        
        # 头部和腰部状态
        head_waist_frame = ttk.Frame(notebook)
        notebook.add(head_waist_frame, text="头部&腰部")
        
        self.head_waist_status_labels = {}
        
        # 头部状态
        for i, name in enumerate(self.joint_names['head']):
            frame = ttk.Frame(head_waist_frame)
            frame.pack(fill=tk.X, padx=5, pady=5)
            ttk.Label(frame, text=f"{name}:", width=15).pack(side=tk.LEFT)
            value_label = ttk.Label(frame, text="0.000", width=10)
            value_label.pack(side=tk.LEFT)
            self.head_waist_status_labels[f"head_{i}"] = value_label
        
        # 腰部状态 - 根据SDK文档: [pitch(rad), height(cm)]
        for i, name in enumerate(self.joint_names['waist']):
            frame = ttk.Frame(head_waist_frame)
            frame.pack(fill=tk.X, padx=5, pady=5)
            ttk.Label(frame, text=f"{name}:", width=15).pack(side=tk.LEFT)
            value_label = ttk.Label(frame, text="0.000", width=10)
            value_label.pack(side=tk.LEFT)
            self.head_waist_status_labels[f"waist_{i}"] = value_label
        
        # 夹爪状态
        gripper_frame = ttk.Frame(notebook)
        notebook.add(gripper_frame, text="夹爪")
        
        self.gripper_status_labels = []
        for i, name in enumerate(self.joint_names['gripper']):
            frame = ttk.Frame(gripper_frame)
            frame.pack(fill=tk.X, padx=5, pady=5)
            ttk.Label(frame, text=f"{name}:", width=15).pack(side=tk.LEFT)
            value_label = ttk.Label(frame, text="0.000", width=10)
            value_label.pack(side=tk.LEFT)
            self.gripper_status_labels.append(value_label)

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

    def run_trajectory(self, path_nodes, side, delta_time, validate=False, release=False):
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
            assert len(arm_joint_values) == len(path_nodes[0])
            assert all(abs(arm_joint_values[i] - path_nodes[0][i]) < 0.1 for i in range(len(arm_joint_values)))
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

                print(robot_states)
                print(robot_actions)
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

    def get_smooth_paths(self, raw_paths):
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
            for i in range(len(current_path)):
                if abs(next_path[i] - current_path[i]) > 0.005:
                    has_inserted_path = True
                    if next_path[i] > current_path[i]:
                        insert_path[i] += 0.005
                    else:
                        insert_path[i] -= 0.005
            if has_inserted_path:
                paths.insert(index + 1, insert_path)
            index += 1
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

    def release_part(self):
        if abs(self.wheel_angle_deg - 215) > 5:
            self.wheel_controller.commands.append(215)
            while True:
                if abs(self.wheel_angle_deg - 205) <= 10:
                    break
                time.sleep(0.1)
        self.run_trajectory(HOME_TO_RELEASE_PATHS, "left", 0.02, validate=True)
        self.robot.move_gripper([0, 0])
        time.sleep(1)
        self.wheel_controller.commands.append(180)
        reversed_paths = list(reversed(HOME_TO_RELEASE_PATHS))
        self.run_trajectory(reversed_paths, "left", 0.02, validate=False)
        self.run_trajectory([LEFT_HAND_HOME_JOINT_VALUES], "left", 0.2)
        time.sleep(0.2)

    def setup_pick_and_place_panel(self, parent):
        """设置控制面板"""
        # 创建滚动区域
        frame = ttk.Frame(parent)
        parent.add(frame, text="pickAndPlace")

        home_frame = ttk.Frame(frame)
        home_frame.pack(fill=tk.X, pady=5)

        ttk.Button(home_frame, text="回Home位置(！直接模式)", command=lambda: self.run_trajectory([LEFT_HAND_HOME_JOINT_VALUES], "left", 2)).pack(side=tk.LEFT, padx=2)

        # ttk.Button(home_frame, text="回Home位置(！步进模式)", command=lambda: self.grasp_depart_home(grasp=False, validate=False)).pack(side=tk.LEFT, padx=2)

        ttk.Button(home_frame, text="前往Ready位置(!处于Home位置)", command=lambda: self.run_trajectory(HOME_TO_READY_PATHS, "left", 0.02, validate=True, release=True)).pack(side=tk.LEFT, padx=2)

        hand_frame = ttk.Frame(frame)
        hand_frame.pack(fill=tk.X, padx=5, pady=5)

        world_hand_coordinates_3d_frame = ttk.Frame(hand_frame)
        world_hand_coordinates_3d_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(world_hand_coordinates_3d_frame, text="world_hand_coordinates_3d:\n").pack(side=tk.LEFT)
        self.world_hand_coordinates_3d_value_label = ttk.Label(world_hand_coordinates_3d_frame, text=f"{[round(v, 3) for v in self.world_hand_coordinates_3d]}")
        self.world_hand_coordinates_3d_value_label.pack(side=tk.LEFT)

        camera_hand_coordinates_3d_frame = ttk.Frame(hand_frame)
        camera_hand_coordinates_3d_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(camera_hand_coordinates_3d_frame, text="camera_hand_coordinates_3d:\n").pack(side=tk.LEFT)
        self.camera_hand_coordinates_3d_value_label = ttk.Label(camera_hand_coordinates_3d_frame, text=f"{[round(v, 3) for v in self.camera_hand_coordinates_3d]}")
        self.camera_hand_coordinates_3d_value_label.pack(side=tk.LEFT)

        ttk.Button(hand_frame, text="更新手部位置", command=lambda: self.update_hand_position()).pack(side=tk.LEFT, padx=2)

        target_frame = ttk.Frame(frame)
        target_frame.pack(fill=tk.X, padx=5, pady=5)

        world_target_coordinates_3d_frame = ttk.Frame(target_frame)
        world_target_coordinates_3d_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(world_target_coordinates_3d_frame, text="world_target_coordinates_3d:\n").pack(side=tk.LEFT)
        self.world_target_coordinates_3d_value_label = ttk.Label(world_target_coordinates_3d_frame, text="0.000")
        self.world_target_coordinates_3d_value_label.pack(side=tk.LEFT)

        camera_target_coordinates_3d_frame = ttk.Frame(target_frame)
        camera_target_coordinates_3d_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(camera_target_coordinates_3d_frame, text="camera_target_coordinates_3d:\n").pack(side=tk.LEFT)
        self.camera_target_coordinates_3d_value_label = ttk.Label(camera_target_coordinates_3d_frame, text="0.000")
        self.camera_target_coordinates_3d_value_label.pack(side=tk.LEFT)

        delta_coordinates_3d_frame = ttk.Frame(target_frame)
        delta_coordinates_3d_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(delta_coordinates_3d_frame, text="delta_coordinates_3d:\n").pack(side=tk.LEFT)
        self.delta_coordinates_3d_frame_value_label = ttk.Label(delta_coordinates_3d_frame, text="0.000")
        self.delta_coordinates_3d_frame_value_label.pack(side=tk.LEFT)

        ttk.Button(target_frame, text="更新目标位置", command=lambda: self.update_target_position()).pack(side=tk.LEFT, padx=2)

        grasp_frame = ttk.Frame(frame)
        grasp_frame.pack(fill=tk.X, pady=5)

        ttk.Button(grasp_frame, text="前往抓取点", command=lambda: self.grasp_approach()).pack(side=tk.LEFT, padx=2)

        ttk.Button(grasp_frame, text="抓取并返回Home", command=lambda: self.grasp_depart_home(grasp=True, depart=True, validate=True, direct=True)).pack(side=tk.LEFT, padx=2)

        ttk.Button(grasp_frame, text="前往释放点释放物件", command=lambda: self.release_part()).pack(side=tk.LEFT, padx=2)

        inference_frame = ttk.Frame(frame)
        inference_frame.pack(fill=tk.X, pady=5)

        ttk.Button(inference_frame, text="InferenceOnce", command=lambda: self.inference_once()).pack(side=tk.LEFT, padx=2)

        ttk.Button(inference_frame, text="ExecuteInferenceResultOnce", command=lambda: self.execute_inference_result(once=True)).pack(side=tk.LEFT, padx=2)

        ttk.Button(inference_frame, text="ExecuteInferenceResult", command=lambda: self.execute_inference_result()).pack(side=tk.LEFT, padx=2)

    def inference_once(self):
        # TODO: left_hand only
        camera_images = self.camera_images["hand_left"]
        assert camera_images is not None and len(camera_images) == 2
        assert len(self.last_two_left_arm_joint_values) == 2

        def _encode_image(rgb_image):
            rgb_image_cv2 = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            _, buffer = cv2.imencode('.jpg', rgb_image_cv2, [cv2.IMWRITE_JPEG_QUALITY, 90])
            base64_string = base64.b64encode(buffer).decode('utf-8')
            return base64_string

        req = {
            'left_imgs': [_encode_image(image) for image in camera_images],
            'state': self.last_two_left_arm_joint_values,
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
        self.robot_info.left_joint_predict_start_values = left_arm_state
        self.robot_info.left_joint_predict_action_values = action.tolist()
        print("left_joint_predict_start_values=", self.robot_info.left_joint_predict_start_values)
        print("left_joint_predict_action_values=", self.robot_info.left_joint_predict_action_values)


    def execute_inference_result(self, once: bool = False):
        def safety_check():
            return True
        while self.robot_info.left_joint_predict_action_values:
            action = self.robot_info.left_joint_predict_action_values.pop(0)
            if not safety_check():
                self.robot_info.left_joint_predict_start_values = None
                self.robot_info.left_joint_predict_action_values = None
                break
            # DANGEROUS!!!!
            self.run_trajectory([action[:7]], "left", 0.2, validate=True)
            self.robot.move_gripper([1 if action[-1] > 0.5 else 0, 0])
            if once:
                break
        
    def setup_control_panel(self, parent):
        """设置控制面板"""
        # 创建滚动区域
        scroll_frame = ttk.Frame(parent)
        parent.add(scroll_frame, text="jog")
        # scroll_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # 创建画布和滚动条
        canvas = tk.Canvas(scroll_frame, height=400)
        scrollbar = ttk.Scrollbar(scroll_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # 打包滚动组件
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # 在可滚动框架中创建控制面板
        control_frame = ttk.LabelFrame(scrollable_frame, text="机器人控制")
        control_frame.pack(fill=tk.X, pady=5)

        # 腰部控制
        waist_frame = ttk.Frame(control_frame)
        waist_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(waist_frame, text="腰部控制:").pack(anchor=tk.W)
        
        # 腰部升降
        lift_frame = ttk.Frame(waist_frame)
        lift_frame.pack(fill=tk.X, pady=5)
        ttk.Label(lift_frame, text="升降:", width=8).pack(side=tk.LEFT)
        ttk.Button(lift_frame, text="上升", command=lambda: self.move_waist_lift(2.0)).pack(side=tk.LEFT, padx=2)
        ttk.Button(lift_frame, text="下降", command=lambda: self.move_waist_lift(-2.0)).pack(side=tk.LEFT, padx=2)
        ttk.Button(lift_frame, text="复位", command=lambda: self.move_waist_lift(0.0)).pack(side=tk.LEFT, padx=2)
        
        # 腰部俯仰
        pitch_frame = ttk.Frame(waist_frame)
        pitch_frame.pack(fill=tk.X, pady=5)
        ttk.Label(pitch_frame, text="俯仰:", width=8).pack(side=tk.LEFT)
        ttk.Button(pitch_frame, text="前倾", command=lambda: self.move_waist_pitch(0.5)).pack(side=tk.LEFT, padx=2)
        ttk.Button(pitch_frame, text="后仰", command=lambda: self.move_waist_pitch(-0.5)).pack(side=tk.LEFT, padx=2)
        ttk.Button(pitch_frame, text="复位", command=lambda: self.move_waist_pitch(0.0)).pack(side=tk.LEFT, padx=2)
        
        # 头部控制
        head_frame = ttk.Frame(control_frame)
        head_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(head_frame, text="头部控制:").pack(anchor=tk.W)
        
        # 头部偏航
        yaw_frame = ttk.Frame(head_frame)
        yaw_frame.pack(fill=tk.X, pady=5)
        ttk.Label(yaw_frame, text="左右:", width=8).pack(side=tk.LEFT)
        ttk.Button(yaw_frame, text="左转", command=lambda: self.move_head_yaw(0.3)).pack(side=tk.LEFT, padx=2)
        ttk.Button(yaw_frame, text="右转", command=lambda: self.move_head_yaw(-0.3)).pack(side=tk.LEFT, padx=2)
        ttk.Button(yaw_frame, text="复位", command=lambda: self.move_head_yaw(0.0)).pack(side=tk.LEFT, padx=2)
        
        # 头部俯仰
        head_pitch_frame = ttk.Frame(head_frame)
        head_pitch_frame.pack(fill=tk.X, pady=5)
        ttk.Label(head_pitch_frame, text="俯仰:", width=8).pack(side=tk.LEFT)
        ttk.Button(head_pitch_frame, text="上扬", command=lambda: self.move_head_pitch(-0.3)).pack(side=tk.LEFT, padx=2)
        ttk.Button(head_pitch_frame, text="下俯", command=lambda: self.move_head_pitch(0.3)).pack(side=tk.LEFT, padx=2)
        ttk.Button(head_pitch_frame, text="复位", command=lambda: self.move_head_pitch(0.0)).pack(side=tk.LEFT, padx=2)
        
        # 夹爪控制
        gripper_frame = ttk.Frame(control_frame)
        gripper_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(gripper_frame, text="夹爪控制:").pack(anchor=tk.W)
        
        # 左夹爪
        left_gripper_frame = ttk.Frame(gripper_frame)
        left_gripper_frame.pack(fill=tk.X, pady=5)
        ttk.Label(left_gripper_frame, text="左夹爪:", width=8).pack(side=tk.LEFT)
        ttk.Button(left_gripper_frame, text="张开", command=lambda: self.move_gripper("left", 0.0)).pack(side=tk.LEFT, padx=2)
        ttk.Button(left_gripper_frame, text="闭合", command=lambda: self.move_gripper("left", 1.0)).pack(side=tk.LEFT, padx=2)
        
        # 右夹爪
        right_gripper_frame = ttk.Frame(gripper_frame)
        right_gripper_frame.pack(fill=tk.X, pady=5)
        ttk.Label(right_gripper_frame, text="右夹爪:", width=8).pack(side=tk.LEFT)
        ttk.Button(right_gripper_frame, text="张开", command=lambda: self.move_gripper("right", 0.0)).pack(side=tk.LEFT, padx=2)
        ttk.Button(right_gripper_frame, text="闭合", command=lambda: self.move_gripper("right", 1.0)).pack(side=tk.LEFT, padx=2)

        # 机械臂末端位置控制
        arm_frame = ttk.Frame(control_frame)
        arm_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(arm_frame, text="机械臂末端控制:").pack(anchor=tk.W)
        
        # 左臂控制
        left_arm_frame = ttk.Frame(arm_frame)
        left_arm_frame.pack(fill=tk.X, pady=5)
        ttk.Label(left_arm_frame, text="左臂:", width=8).pack(side=tk.LEFT)
        ttk.Button(left_arm_frame, text="向前", 
                  command=lambda: self.move_arm_relative("left", [0.05, 0, 0])).pack(side=tk.LEFT, padx=2)
        ttk.Button(left_arm_frame, text="向后", 
                  command=lambda: self.move_arm_relative("left", [-0.05, 0, 0])).pack(side=tk.LEFT, padx=2)
        ttk.Button(left_arm_frame, text="向左", 
                  command=lambda: self.move_arm_relative("left", [0, 0.05, 0])).pack(side=tk.LEFT, padx=2)
        ttk.Button(left_arm_frame, text="向右", 
                  command=lambda: self.move_arm_relative("left", [0, -0.05, 0])).pack(side=tk.LEFT, padx=2)
        ttk.Button(left_arm_frame, text="向上", 
                  command=lambda: self.move_arm_relative("left", [0, 0, 0.05])).pack(side=tk.LEFT, padx=2)
        ttk.Button(left_arm_frame, text="向下", 
                  command=lambda: self.move_arm_relative("left", [0, 0, -0.05])).pack(side=tk.LEFT, padx=2)
        
        # 右臂控制
        right_arm_frame = ttk.Frame(arm_frame)
        right_arm_frame.pack(fill=tk.X, pady=5)
        ttk.Label(right_arm_frame, text="右臂:", width=8).pack(side=tk.LEFT)
        ttk.Button(right_arm_frame, text="向前", 
                  command=lambda: self.move_arm_relative("right", [0.05, 0, 0])).pack(side=tk.LEFT, padx=2)
        ttk.Button(right_arm_frame, text="向后", 
                  command=lambda: self.move_arm_relative("right", [-0.05, 0, 0])).pack(side=tk.LEFT, padx=2)
        ttk.Button(right_arm_frame, text="向左", 
                  command=lambda: self.move_arm_relative("right", [0, 0.05, 0])).pack(side=tk.LEFT, padx=2)
        ttk.Button(right_arm_frame, text="向右", 
                  command=lambda: self.move_arm_relative("right", [0, -0.05, 0])).pack(side=tk.LEFT, padx=2)
        ttk.Button(right_arm_frame, text="向上", 
                  command=lambda: self.move_arm_relative("right", [0, 0, 0.05])).pack(side=tk.LEFT, padx=2)
        ttk.Button(right_arm_frame, text="向下", 
                  command=lambda: self.move_arm_relative("right", [0, 0, -0.05])).pack(side=tk.LEFT, padx=2)
        
        # 预设位置
        preset_frame = ttk.Frame(arm_frame)
        preset_frame.pack(fill=tk.X, pady=5)
        ttk.Label(preset_frame, text="预设:", width=8).pack(side=tk.LEFT)
        ttk.Button(preset_frame, text="双臂平举", 
                  command=lambda: self.set_arm_preset("parallel")).pack(side=tk.LEFT, padx=2)
        ttk.Button(preset_frame, text="双臂下垂", 
                  command=lambda: self.set_arm_preset("down")).pack(side=tk.LEFT, padx=2)
        ttk.Button(preset_frame, text="双臂前伸", 
                  command=lambda: self.set_arm_preset("forward")).pack(side=tk.LEFT, padx=2)

        # 复位按钮
        reset_frame = ttk.Frame(control_frame)
        reset_frame.pack(fill=tk.X, padx=10, pady=10)
        ttk.Button(reset_frame, text="机器人复位", command=self.reset_robot, style="Accent.TButton").pack()
        
    def start_camera_thread(self):
        """启动相机图像更新线程"""
        def update_camera_images():
            while True:
                try:

                    left_arm_state = self.left_arm_joint_values + [self.robot.gripper_states()[0][0]]
                    if not self.last_two_left_arm_joint_values:
                        self.last_two_left_arm_joint_values = [left_arm_state, left_arm_state]
                    else:
                        self.last_two_left_arm_joint_values = [self.last_two_left_arm_joint_values[-1], left_arm_state]
                    # 获取各个相机的最新图像和内参
                    for camera_name in self.camera_images.keys():
                        image, timestamp = self.camera.get_latest_image(camera_name)
                        if image is not None:
                            # print(f"获取到 {camera_name} 相机图像，图像 shape: {image.shape}")
                            if len(image.shape)== 3:
                                height, width = image.shape[:2]
                            else:
                                height, width = image.shape[0]
                            # print(f"获取到 {camera_name} 相机图像，图像 height, width: {height} {width}")
                            # 对于深度相机，保存原始图像数据
                            if camera_name == "head_depth":
                                self.camera_images[camera_name] = image.copy()
                            elif "hand" in camera_name:
                                if self.camera_images[camera_name] is None:
                                    self.camera_images[camera_name] = [image, image]
                                else:
                                    self.camera_images[camera_name] = [self.camera_images[camera_name][-1], image]

                            self.update_camera_display(camera_name, image)
                        
                        # 获取相机内参（只需获取一次）
                        if self.camera_intrinsics[camera_name] is None:
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

                    # update robot_info
                    gripper_states, _ = self.robot.gripper_states()
                    self.robot_info.left_joint_values = arm_states[:7] + [gripper_states[0]]
                    self.robot_info.right_joint_values = arm_states[7:14] + [gripper_states[1]]

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
                    try:
                        # if arm_states and len(arm_states) >= 2:
                        #     # 假设夹爪状态是手臂关节的最后两个值
                        #     gripper_states = arm_states[-2:]
                        for i, state in enumerate(gripper_states):
                            if i < len(self.gripper_status_labels) and state is not None:
                                self.root.after(0, lambda label=self.gripper_status_labels[i], value=state:
                                              label.config(text=f"{value:.3f}"))
                    except Exception as e:
                        print(f"夹爪状态更新错误: {e}")
                    
                    time.sleep(0.1)  # 100ms更新一次
                except Exception as e:
                    print(f"状态更新错误: {e}")
                    time.sleep(1)
        
        status_thread = threading.Thread(target=update_robot_status, daemon=True)
        status_thread.start()
        
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

                original_depth = image.copy()

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

                if camera_name == "head_depth":
                    self.camera_images[camera_name] = original_depth

            else:
                print(f"不支持shape: {image.shape}")
                return

            # ================= resize =================
            if camera_name == "head":
                pil_image = pil_image.resize((800, 600), Image.Resampling.LANCZOS)
            else:
                # pil_image = pil_image.resize((320, 240), Image.Resampling.LANCZOS)
                pil_image = pil_image.resize((256, 195), Image.Resampling.LANCZOS)

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
        label.config(image=photo)
        label.image = photo  # 保持引用
    
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
    
    def get_depth_at_pixel(self, camera_name, pixel_x, pixel_y):
        """获取指定像素位置的深度值"""
        try:
            # 获取深度图像
            if camera_name == "head":
                # 如果是RGB相机，尝试获取对应的深度相机图像
                depth_image = self.camera_images.get("head_depth")
                if depth_image is None:
                    print("无法获取深度图像")
                    return None
            elif camera_name == "head_depth":
                depth_image = self.camera_images.get("head_depth")
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
    
    def update_label_image(self, label, photo):
        """更新标签图像"""
        label.config(image=photo)
        label.image = photo  # 保持引用
    
    def show_status(self, message, message_type="info"):
        """在状态栏显示消息"""
        timestamp = time.strftime("%H:%M:%S")
        status_message = f"[{timestamp}] {message}"
        self.status_text.set(status_message)
        print(f"[{message_type.upper()}] {message}")
    
    def clear_status(self):
        """清除状态栏"""
        self.status_text.set("就绪")

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
        """窗口关闭时的处理"""
        try:
            self.camera.close()
            self.robot.shutdown()
            self.root.destroy()
        except Exception as e:
            print(f"关闭时出错: {e}")
            self.root.destroy()

def main():
    root = tk.Tk()
    
    # 设置主题样式
    style = ttk.Style()
    style.theme_use('clam')
    
    # 配置按钮样式
    style.configure("Accent.TButton", foreground="white", background="#007acc", padding=5)
    
    app = RobotControlGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        app.on_closing()

if __name__ == "__main__":
    main()