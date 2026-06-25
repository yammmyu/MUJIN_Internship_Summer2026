"""智元 G1 机器人控制 GUI 主入口（精简版）。

只保留三件事：相机视图、左夹爪开合、以及策略推理（手动单步 / 自动运行 /
仿真预览 / 真机释放 / 子步监视）。功能拆分到 gui/ 包下的 Mixin：

  - StyleMixin      : ttk 主题
  - CameraMixin     : 相机视图（纯显示）
  - InferenceMixin  : 左夹爪 + 推理控制 + 子步监视

本文件负责组装、初始化机器人/相机/环境/推理与共享状态、搭建界面、生命周期清理。
"""

import os
import signal
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk
import argparse

import rclpy

from a2d_sdk.robot import RobotDds as Robot, RobotController
from control_wheel_example import WheelController
from robot_info_server import create_robot_info_http_server, RobotInfo

from gui import StyleMixin, CameraMixin, InferenceMixin
from real_world import HumanoidEnv, InferenceController
from real_world.humanoid_env import RECORD_HZ
from real_world.sim_backend import SimEnv


class RobotControlGUI(StyleMixin, CameraMixin, InferenceMixin):
    def __init__(self, root, camera_mode="all"):
        self.root = root
        self.root.title("智元G1机器人控制界面")
        self.root.geometry("1500x950")
        self.root.minsize(1100, 720)
        self._setup_styles()

        # 机器人（相机由 HumanoidEnv 持有，见下方 self.env）
        self.robot = Robot()
        self.robot_controller = RobotController()

        # data 模式下保持 3 路相机常开，普通模式不订阅任何相机（无视频流带宽）
        camera_names = ["hand_left", "hand_right", "head"] if camera_mode == "data" else []
        self.camera_mode = camera_mode

        # rclpy / 轮控后台 spin —— 保持 DDS 关节状态持续刷新
        rclpy.init(args=None)
        self.wheel_controller = WheelController()
        threading.Thread(target=rclpy.spin, args=(self.wheel_controller,), daemon=True).start()
        time.sleep(1.0)             # 等待初始化

        # 相机帧缓存（显示线程把 env 拉到的帧镜像到此，供「保存图片」读取）
        self.camera_images = {}
        # 相机显示缩放尺寸（由 _rebuild_camera_display 根据排版动态更新）
        self.camera_tile_size = (320, 240)

        # robot_info HTTP 服务（推理预测/可视化）
        self.robot_info = RobotInfo()
        create_robot_info_http_server(self.robot_info)

        # HumanoidEnv：相机的唯一持有者与抓取者；real=True 启用真机释放管线。
        self.env = HumanoidEnv(
            robot=self.robot,
            robot_controller=self.robot_controller,
            cameras=camera_names,
            frequency=RECORD_HZ,
            real=True,
        )

        # In-process PyBullet 预览：用户按「启动仿真预览」时在自有线程上懒加载（所有 p.* 调用
        # 都在该线程），attach 到 self.env.sim 供执行循环驱动。None 表示未启动。
        self._sim_thread = None
        self._sim_stop = threading.Event()

        # 策略推理控制器（复用注入的 env）
        self.inference = InferenceController(self.env, self.robot_info)

        # 恢复上次保存的调参（平滑/执行参数），在搭建界面前应用，使控件初值与运行时一致。
        self._load_and_apply_tuning()

        # 左夹爪状态（move_gripper 读写；右夹爪界面不暴露，保持不动）
        self.left_gripper_pos = 0.0
        self.right_gripper_pos = 0.0

        # 状态栏
        self.status_text = tk.StringVar()
        self.status_text.set("就绪")
        self.status_label = None

        self.setup_ui()
        self.env.start()           # 启动相机采集 + 执行线程（GUI 持有 env 生命周期）
        self.inference.start()
        self.start_camera_thread()

    def setup_ui(self):
        """顶部标题栏 + 左右分栏（左相机 / 右推理控制）+ 底部状态栏。"""
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
        ttk.Button(status_bar, text="清除", style="Muted.TButton",
                   command=lambda: self.status_text.set("就绪")
                   ).pack(side=tk.RIGHT, padx=(8, 0))

        # ===== 主分栏 =====
        body = ttk.Frame(self.root)
        body.pack(fill=tk.BOTH, expand=True, padx=12, pady=4)

        left_frame = ttk.Frame(body)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))

        right_frame = ttk.LabelFrame(body, text="  🧠  推理控制  ")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(8, 0))

        self.setup_camera_panel(left_frame)         # 左：相机视图
        self.setup_inference_panel(right_frame)     # 右：左夹爪 + 推理控制 + 子步监视

    # ===================== sim preview (in-process PyBullet) =====================
    def launch_sim(self):
        """Start the PyBullet preview on its own thread (idempotent)."""
        if self._sim_thread is not None and self._sim_thread.is_alive():
            self.status_text.set("仿真预览已在运行")
            return
        self._sim_stop.clear()
        self._sim_thread = threading.Thread(target=self._sim_loop, daemon=True)
        self._sim_thread.start()
        self.status_text.set("仿真预览已启动")

    def _sim_loop(self):
        """Owns the PyBullet connection: build it here, seed to the robot's current pose,
        attach to env, then step until stopped. All p.* calls stay on this thread; the exec
        thread only calls sim.command() (no p.*)."""
        try:
            sim = SimEnv(direct=False)
        except Exception as e:
            print(f"[GUI] sim launch failed: {e}")
            return
        try:
            arm14 = self.robot.arm_joint_states()[0]            # both arms (rad)
            grip = self.robot.gripper_states()[0]               # [left, right] in [0,1]
            body_pitch = self.robot.waist_joint_states()[0][0]  # waist pitch (rad)
            # Match the sim to the physical robot on load: both arms, both grippers, torso pitch.
            sim.reset_full(arm14=arm14, body_pitch=body_pitch, gripper_lr=grip)
            self.env.set_seed(arm14[:7])    # IK warm-starts from where the real left arm is
        except Exception as e:
            print(f"[GUI] sim seed failed: {e}")
        self.env.sim = sim              # exec loop now drives the preview
        try:
            while not self._sim_stop.is_set() and sim.connected():
                sim.step()
        finally:
            self.env.sim = None
            sim.disconnect()

    def _stop_sim(self):
        self._sim_stop.set()
        if self._sim_thread is not None:
            self._sim_thread.join(timeout=3.0)

    def on_closing(self):
        """窗口关闭时依次释放各资源，最后强制退出进程。

        每个资源单独 try/except，避免前一个清理失败阻断后续；
        最后用 os._exit() 兜底，绕过被 DDS/rclpy 原生线程卡住的解释器退出。
        """
        if getattr(self, "_is_closing", False):     # 信号 + 窗口关闭可能同时触发
            return
        self._is_closing = True

        for label, fn in [
            ("推理控制器", self.inference.stop),
            ("仿真预览", self._stop_sim),     # 先停仿真步进线程，再关 env
            ("HumanoidEnv", self.env.stop),   # 先停采集/执行线程，再关其持有的相机
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


def _safety_preflight():
    """Run the safety-invariant suite (scripts/test_safety_invariants.py) before the GUI builds
    the robot. A failure means a safety regression and BLOCKS launch — the GUI can drive
    hardware, so it must not start on a broken release pipeline. Set
    HUMANOID_SKIP_SAFETY_PREFLIGHT=1 to bypass (logs a loud warning)."""
    if os.environ.get("HUMANOID_SKIP_SAFETY_PREFLIGHT") == "1":
        print("\n*** WARNING: safety pre-flight SKIPPED via HUMANOID_SKIP_SAFETY_PREFLIGHT=1 ***\n")
        return
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
    print("[startup] running safety pre-flight (scripts/test_safety_invariants.py)…")
    try:
        from test_safety_invariants import run as run_safety
        run_safety()
    except Exception as e:
        print(f"\n*** SAFETY PRE-FLIGHT FAILED: {type(e).__name__}: {e}\n"
              f"*** Refusing to launch the control GUI. Fix the regression, or set\n"
              f"*** HUMANOID_SKIP_SAFETY_PREFLIGHT=1 to bypass (NOT recommended).\n")
        sys.exit(1)
    print("[startup] safety pre-flight passed.\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", action="store_true")
    args = parser.parse_args()

    camera_mode = "data" if args.data else "all"

    _safety_preflight()    # block launch if the safety invariants regressed

    root = tk.Tk()
    app = RobotControlGUI(root, camera_mode)
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
