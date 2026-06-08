import copy
import math
import threading
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

    def _run_validation(self, once=False):
        """Step the last prediction through the sim off the Tk thread (it can take a few seconds
        while the sim plays the trajectory) and report the result to the status bar.

        once=True validates+stages the NEXT unexecuted action row; once=False the REMAINING
        rows. Both become no-ops once the chunk is fully consumed (see _refresh_validation_buttons).
        """
        def worker():
            ok, reason = self.inference.execute_inference_result(once=once)
            remaining = self.inference.steps_remaining()
            msg = (f"✅ 仿真验证通过（剩余 {remaining} 步），可释放到真机" if ok
                   else f"❌ 仿真验证失败：{reason}")
            def done():
                self.status_text.set(msg)
                self._refresh_validation_buttons()
                self._refresh_release_buttons()    # a successful validate stages new substeps
            self.root.after(0, done)
        self.status_text.set("仿真验证中…")
        threading.Thread(target=worker, daemon=True).start()

    def _run_inference_once(self):
        """推理一次, off the Tk thread (the server round-trip can block for seconds). On
        completion a fresh chunk exists, so re-enable the step-through buttons."""
        def worker():
            ok = self.inference.inference_once()
            remaining = self.inference.steps_remaining()
            msg = (f"✅ 推理完成，共 {remaining} 步" if ok else "❌ 推理失败")
            def done():
                self.status_text.set(msg)
                self._refresh_validation_buttons()
            self.root.after(0, done)
        self.status_text.set("推理中…")
        threading.Thread(target=worker, daemon=True).start()

    def _run_release_substeps(self, remaining=False):
        """释放 staged substeps to the real robot. remaining=False sends the next single substep
        (one 33ms tick); remaining=True streams all staged substeps at the 33ms tick. No-op (with
        a status note) when nothing is staged."""
        if remaining:
            n = self.env.release_remaining_substeps()
        else:
            n = self.env.release_next_substep()
        staged = self.env.staged_substeps
        if n > 0:
            self.status_text.set(f"🚀 已下发 {n} 条指令到真机（剩余待释放 {staged} 子步）")
        else:
            self.status_text.set("⚠ 没有待释放的子步（先在仿真中验证）")
        self._refresh_release_buttons()            # staged buffer shrank (maybe now empty)

    def _refresh_validation_buttons(self):
        """Enable the 单步/整条 buttons only while the current chunk has unexecuted steps; grey
        them out (and they no-op anyway) once it's consumed or there's no prediction."""
        try:
            remaining = self.inference.steps_remaining()
        except Exception:
            remaining = 0
        flag = "!disabled" if remaining > 0 else "disabled"
        for name in ("_btn_validate_step", "_btn_validate_rest"):
            btn = getattr(self, name, None)
            if btn is not None:
                btn.state([flag])

    def _update_substep_monitor(self):
        """Refresh the live joints readout + the rolling next-10 staged-substep table (each cell
        shows the joint value and its delta vs the previous row / the live pose). Reschedules
        itself ~5 Hz; the staged buffer draining in real time produces the rolling effect."""
        try:
            # --- live actual left-arm joints (delta reference for substep #0) ---
            try:
                actual = list(self.left_arm_joint_values)
            except Exception:
                actual = None
            if actual is not None and len(actual) >= 7:
                self._monitor_actual_var.set(
                    "   ".join(f"J{k+1}:{actual[k]:+.3f}" for k in range(7)))
                prev = np.asarray(actual[:7], dtype=np.float64)
            else:
                self._monitor_actual_var.set("（无法读取关节）")
                prev = None

            # --- next up-to-10 staged substeps, with per-joint delta vs the previous row ---
            try:
                subs = self.env.staged_preview(self._monitor_rows)
            except Exception:
                subs = []
            for i in range(self._monitor_rows):
                iid = f"subrow{i}"
                if i < len(subs):
                    q = np.asarray(subs[i], dtype=np.float64)
                    if prev is not None and len(prev) >= 7:
                        d = q - prev
                        vals = [f"{q[k]:+.3f} (Δ{d[k]:+.3f})" for k in range(7)]
                    else:
                        vals = [f"{q[k]:+.3f}" for k in range(7)]
                    self._monitor_tree.item(iid, values=(i, *vals))
                    prev = q
                else:
                    self._monitor_tree.item(iid, values=(i, *[""] * 7))
        except tk.TclError:
            self._monitor_after_id = None       # widget destroyed (window closed) -> stop
            return
        finally:
            if getattr(self, "_monitor_after_id", None) is not None:
                self._monitor_after_id = self.root.after(200, self._update_substep_monitor)

    def _refresh_release_buttons(self):
        """Enable the 释放(单步)/释放(剩余) buttons only while substeps are staged for release;
        grey them out (and they no-op anyway) once the staged buffer is empty."""
        try:
            staged = self.env.staged_substeps
        except Exception:
            staged = 0
        flag = "!disabled" if staged > 0 else "disabled"
        for name in ("_btn_release_step", "_btn_release_rest"):
            btn = getattr(self, name, None)
            if btn is not None:
                btn.state([flag])

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
                   command=lambda: self._run_inference_once()
                   ).pack(side=tk.LEFT, padx=4)
        # "执行" now means VALIDATE-IN-SIM (step + self-collision + readback); it stages the
        # sim-validated trajectory but does NOT touch the robot. Run off the Tk thread so the
        # GUI doesn't freeze while the sim plays the trajectory. 单步 steps to the NEXT predicted
        # target; 整条 runs the REMAINING targets; both disable once the chunk is consumed.
        self._btn_validate_step = ttk.Button(manual_row, text="仿真验证(单步)",
                   style="Primary.TButton",
                   command=lambda: self._run_validation(once=True))
        self._btn_validate_step.pack(side=tk.LEFT, padx=4)
        self._btn_validate_rest = ttk.Button(manual_row, text="仿真验证(整条)",
                   style="Primary.TButton",
                   command=lambda: self._run_validation(once=False))
        self._btn_validate_rest.pack(side=tk.LEFT, padx=4)
        # No prediction yet -> start disabled; 推理一次 re-enables them.
        self._refresh_validation_buttons()

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
        # Release staged sim-validated substeps to the robot (only path to hardware). 仿真验证
        # accumulates substeps; 释放(单步) sends the next one (one 33ms tick), 释放(剩余) streams
        # the rest at the 33ms tick.
        self._btn_release_step = ttk.Button(sim_row, text="🚀 释放子步(单步)",
                   style="Danger.TButton",
                   command=lambda: self._run_release_substeps(remaining=False))
        self._btn_release_step.pack(side=tk.LEFT, padx=4)
        self._btn_release_rest = ttk.Button(sim_row, text="🚀 释放子步(剩余)",
                   style="Danger.TButton",
                   command=lambda: self._run_release_substeps(remaining=True))
        self._btn_release_rest.pack(side=tk.LEFT, padx=4)
        # Nothing staged yet -> start disabled; 仿真验证 enables them.
        self._refresh_release_buttons()
        # E-STOP: latched; drops pending + actively holds. Physical E-stop remains primary.
        # It also clears the staged buffer, so grey out the release buttons.
        ttk.Button(sim_row, text="⛔ 急停",
                   style="Danger.TButton",
                   command=lambda: (self.env.lock_robot(), self._refresh_release_buttons())
                   ).pack(side=tk.LEFT, padx=4)
        # Clear the latched E-stop (only after the operator confirms the arm is safe).
        ttk.Button(sim_row, text="重置急停",
                   style="Muted.TButton",
                   command=lambda: self.env.reset_estop()
                   ).pack(side=tk.LEFT, padx=4)

        # ===== 子步监视：实时左臂7关节 + 待释放子步滚动表（含每关节增量） =====
        sec_monitor = ttk.LabelFrame(body, text="  📈  子步监视（左臂 7 关节，单位 rad）  ")
        sec_monitor.pack(fill=tk.X, padx=10, pady=6)

        # Live actual left-arm joints (pulled from the robot each refresh tick).
        live_row = ttk.Frame(sec_monitor)
        live_row.pack(fill=tk.X, padx=8, pady=(8, 4))
        ttk.Label(live_row, text="实时关节:",
                  style="Section.TLabel").pack(side=tk.LEFT, padx=(0, 6))
        self._monitor_actual_var = tk.StringVar(value="（等待数据）")
        ttk.Label(live_row, textvariable=self._monitor_actual_var,
                  style="Value.TLabel").pack(side=tk.LEFT)

        ttk.Label(sec_monitor,
                  text="待释放子步（# 0 = 下一个释放；括号内为相对上一行/实时关节的增量 Δ）",
                  anchor=tk.W).pack(fill=tk.X, padx=8, pady=(2, 2))

        # Rolling table: one row per upcoming substep, value "q (Δ)" per joint. Rows are
        # pre-created and rewritten in place each tick so substeps appear to scroll up as
        # they are released.
        cols = ("idx", "j1", "j2", "j3", "j4", "j5", "j6", "j7")
        tree = ttk.Treeview(sec_monitor, columns=cols, show="headings", height=10)
        tree.heading("idx", text="#")
        tree.column("idx", width=32, anchor="center", stretch=False)
        for k, c in enumerate(cols[1:], start=1):
            tree.heading(c, text=f"J{k}")
            tree.column(c, width=120, anchor="center", stretch=True)
        tree.pack(fill=tk.X, padx=8, pady=(0, 8))
        self._monitor_tree = tree
        self._monitor_rows = 10
        for i in range(self._monitor_rows):
            tree.insert("", "end", iid=f"subrow{i}", values=(i, *[""] * 7))

        # Start the periodic refresh (idempotent — only schedules once).
        if getattr(self, "_monitor_after_id", None) is None:
            self._monitor_after_id = self.root.after(200, self._update_substep_monitor)

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
