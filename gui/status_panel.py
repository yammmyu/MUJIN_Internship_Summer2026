import threading
import time
import tkinter as tk
from tkinter import ttk


class StatusMixin:
    """关节状态面板、状态刷新线程与状态栏消息。"""

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

                    # 获取夹爪状态
                    gripper_states, _ = self.robot.gripper_states()
                    for i, state in enumerate(gripper_states):
                        if i < len(self.gripper_status_labels) and state is not None:
                            self.root.after(0, lambda label=self.gripper_status_labels[i], value=1 if state > 0.5 else 0:
                                          label.config(text=f"{value:.3f}"))

                    time.sleep(0.1)  # 100ms更新一次
                except Exception as e:
                    print(f"状态更新错误: {e}")
                    time.sleep(1)

        status_thread = threading.Thread(target=update_robot_status, daemon=True)
        status_thread.start()

    def show_status(self, message, message_type="info"):
        """在状态栏显示消息"""
        timestamp = time.strftime("%H:%M:%S")
        status_message = f"[{timestamp}] {message}"
        self.status_text.set(status_message)
        print(f"[{message_type.upper()}] {message}")

    def clear_status(self):
        """清除状态栏"""
        self.status_text.set("就绪")
