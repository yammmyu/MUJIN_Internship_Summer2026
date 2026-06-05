import copy
import math
import time
import tkinter as tk
from tkinter import ttk

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import savgol_filter

from camera_pose import compute_point_B_world
from constants import *


class PickPlaceMixin:
    """抓取任务：手/目标坐标、轨迹规划与执行、抓取动作、抓取任务面板。"""

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
                   command=lambda: self.inference.inference_once()
                   ).pack(side=tk.LEFT, padx=4)
        ttk.Button(manual_row, text="执行一次推理轨迹",
                   style="Primary.TButton",
                   command=lambda: self.inference.execute_inference_result(once=True)
                   ).pack(side=tk.LEFT, padx=4)
        ttk.Button(manual_row, text="执行剩余推理轨迹",
                   style="Primary.TButton",
                   command=lambda: self.inference.execute_inference_result()
                   ).pack(side=tk.LEFT, padx=4)

        auto_row = ttk.Frame(sec_inf)
        auto_row.pack(fill=tk.X, padx=8, pady=(4, 8))
        ttk.Label(auto_row, text="自动:",
                  style="Section.TLabel").pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(auto_row, text="⚠ 开始自动运行",
                   style="Danger.TButton",
                   command=print("Hello World") #lambda: self.inference.auto_inference() #temporarily disabled
                   ).pack(side=tk.LEFT, padx=4)
        ttk.Button(auto_row, text="■ 停止自动运行",
                   style="Muted.TButton",
                   command=print("No Hello World") #lambda: self.inference.auto_inference(stop=True) #temporarily disabled
                   ).pack(side=tk.LEFT, padx=4)

        # ===== 仿真预览 + 真机释放（先在仿真里预览，确认后再解锁真机执行）=====
        sim_row = ttk.Frame(sec_inf)
        sim_row.pack(fill=tk.X, padx=8, pady=(4, 8))
        ttk.Label(sim_row, text="真机:",
                  style="Section.TLabel").pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(sim_row, text="🟦 启动仿真预览",
                   style="Primary.TButton",
                   command=lambda: self.launch_sim()
                   ).pack(side=tk.LEFT, padx=4)
        # Release the LAST sim-executed trajectory to the robot (only path to hardware).
        ttk.Button(sim_row, text="🚀 释放到真机",
                   style="Danger.TButton",
                   command=lambda: self.env.release_to_robot()
                   ).pack(side=tk.LEFT, padx=4)
        # Hard stop: drop everything pending on the robot.
        ttk.Button(sim_row, text="⛔ 急停真机",
                   style="Muted.TButton",
                   command=lambda: self.env.lock_robot()
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
