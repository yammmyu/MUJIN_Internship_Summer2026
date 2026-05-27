import threading
import time
import tkinter as tk
from tkinter import ttk

import numpy as np


class ManualControlMixin:
    """手动调试：腰/头/夹爪/机械臂方向控制、预设姿态、末端位姿与复位。"""

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
