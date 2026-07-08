"""智元 G1 机器人控制 GUI 主入口（精简版）。

只保留三件事：相机视图、左夹爪开合、以及策略推理（手动单步 / 自动运行 /
仿真预览 / 真机释放 / 子步监视）。功能拆分到 gui/ 包下的 Mixin：

  - StyleMixin      : ttk 主题
  - CameraMixin     : 相机视图（纯显示）
  - InferenceMixin  : 左夹爪 + 推理控制 + 子步监视

本文件负责组装、初始化机器人/相机/环境/推理与共享状态、搭建界面、生命周期清理。
"""

import os
import logging
import signal
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk
import argparse

import rclpy

from a2d_sdk.robot import RobotDds as Robot, RobotController, Slam
from examples.control_wheel_example import WheelController
from servers.robot_info_server import create_robot_info_http_server, RobotInfo

from gui import StyleMixin, CameraMixin, InferenceMixin, VRMixin, DataCollectionMixin
from pico_vr.pico_vr_server.server import DummyServer
from real_world import HumanoidEnv, InferenceController
from real_world.humanoid_env import RECORD_HZ
from real_world.sim_backend import SimEnv


class RobotControlGUI(StyleMixin, CameraMixin, InferenceMixin, VRMixin, DataCollectionMixin):
    def __init__(self, root, camera_mode="all"):
        self.root = root
        self.root.title("智元G1机器人控制界面")
        self.root.geometry("1600x950")
        self.root.minsize(1180, 720)
        self._setup_styles()

        # 机器人（相机由 HumanoidEnv 持有，见下方 self.env）
        self.robot = Robot()
        self.robot_controller = RobotController()

        # Slam 必须在进程内实例化，否则 arm/head/waist 的 joint_states 会一直冻结，
        # 而 VR 执行线程每拍都读取这三者（并 assert 非空）。此处仅创建以解冻关节状态，
        # 不切换底盘导航模式，避免遥操过程中底盘被自动导航带动。
        try:
            self.slam = Slam()
            print("SLAM 模块初始化成功（已解冻关节状态）")
        except Exception as e:
            self.slam = None
            print(f"⚠️ SLAM 初始化失败，VR 遥操所需关节状态可能不可用: {e}")

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
        self.inference = InferenceController(self.env, self.robot_info, grasp_detector_path="data/grasp_detector/detector.pt")

        # 恢复上次保存的调参（平滑/执行参数），在搭建界面前应用，使控件初值与运行时一致。
        self._load_and_apply_tuning()

        # 左夹爪状态（move_gripper 读写；右夹爪界面不暴露，保持不动）
        self.left_gripper_pos = 0.0
        self.right_gripper_pos = 0.0

        # ---- VR 遥操共享状态（_handle_vr_joints / 执行线程读写）----
        # is_vr_control 是总开关：False 时回调照常解析手柄姿态但不下发任何机器人动作。
        self.is_vr_control = False
        self.vr_execution_thread = None
        self.vr_actions = []
        self.previous_vr_positions = []
        self.last_joint_update_timestamp = 0.0
        self.vr_buttons_pressed = set()
        # 最新摇杆轴快照（[L_x, L_y, R_x, R_y]），由 _handle_vr_joints 每拍刷新；
        # 50Hz 执行线程据此做摇杆腕部滚转（速率控制）。
        self.vr_axes = []

        # 状态栏
        self.status_text = tk.StringVar()
        self.status_text.set("就绪")
        self.status_label = None

        self.setup_ui()
        self.env.start()           # 启动相机采集 + 执行线程（GUI 持有 env 生命周期）
        self.inference.start()
        self.start_camera_thread()

        # ---- VR 遥操数据通道 ----
        # 上行(:5556)：头显客户端把 HMD+手柄 21 维姿态回传 → on_joints=_handle_vr_joints。
        #              这是把头显动作接到机器人的唯一连接点。
        # 下行(:5555)：start_vr_stream_thread 把相机画面合成后 set_image 推给头显。
        # use_default_image=False：不发内置占位图，等真实相机帧。
        self.dummy_server = DummyServer(
            on_joints=self._handle_vr_joints,
            use_default_image=False,
        )
        self.dummy_server.start()
        self.start_vr_stream_thread()

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

        # ===== 主区：顶层 Notebook（控制台 / VR 遥操 两个标签页）=====
        tabs = ttk.Notebook(self.root)
        tabs.pack(fill=tk.BOTH, expand=True, padx=12, pady=4)

        # ---- 标签页 1：控制台（相机视图 + 推理控制左右分栏）----
        body = ttk.Frame(tabs)
        tabs.add(body, text="  🧠  控制台  ")

        left_frame = ttk.Frame(body)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))

        # 右侧推理控制为固定宽面板（其内部 canvas 已按需预留宽度），左侧相机视图占据剩余空间并随
        # 窗口伸缩。此前右侧也是 expand=True，两栏按 50/50 平分宽度，而推理面板的 canvas 内容比默认
        # 窗口的一半还宽，于是右半被裁切。
        right_frame = ttk.LabelFrame(body, text="  🧠  推理控制  ")
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, expand=False, padx=(8, 0))

        self.setup_camera_panel(left_frame)         # 左：相机视图
        self.setup_inference_panel(right_frame)     # 右：左夹爪 + 推理控制 + 子步监视

        # ---- 标签页 2：VR 遥操（开关 + 灵敏度参数）----
        vr_tab = ttk.Frame(tabs)
        tabs.add(vr_tab, text="  🎮  VR 遥操  ")
        self.setup_vr_panel(vr_tab)

    def setup_vr_panel(self, parent):
        """VR 遥操控制：左侧启动/停止开关 + 灵敏度参数标签页，右侧数据采集面板。

        头显客户端连上 :5556 后即把手柄姿态喂给 _handle_vr_joints；但只有按下
        「启动」把 is_vr_control 置 True，回调才会真正下发机器人动作。执行线程在
        回调内按 is_vr_control 懒创建/回收，故开关本身即可驱动整条管线。

        数据采集面板复用 self.env 的录制 API（start/stop_recording），可在遥操过程中
        边操作边录制；右侧固定宽，左侧 VR 控制随窗口伸缩。
        """
        # 左：VR 控制（开关 + 灵敏度参数），占据剩余空间并随窗口伸缩。
        left = ttk.Frame(parent)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))

        top = ttk.Frame(left)
        top.pack(fill=tk.X, padx=8, pady=8)
        self.vr_toggle_btn = ttk.Button(
            top, text="启动 VR 遥操", style="Primary.TButton",
            command=self._toggle_vr,
        )
        self.vr_toggle_btn.pack(side=tk.LEFT)

        # 灵敏度参数面板需要一个 Notebook 容器（setup_vr_params_panel 调 parent.add）。
        nb = ttk.Notebook(left)
        nb.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
        self.setup_vr_params_panel(nb)

        # 右：数据采集（录制控制 + 末端位姿实时），固定宽面板。
        right = ttk.LabelFrame(parent, text="  📦  数据采集  ")
        right.pack(side=tk.RIGHT, fill=tk.Y, expand=False, padx=(8, 0))
        self.setup_data_collection_panel(right)

    def _toggle_vr(self):
        """翻转 VR 遥操总开关并同步按钮/状态栏文案。"""
        self.is_vr_control = not self.is_vr_control
        if self.is_vr_control:
            self.vr_toggle_btn.config(text="停止 VR 遥操")
            self.status_text.set("VR 遥操：开（按住 L_Y / R_B 移动手臂）")
        else:
            self.vr_toggle_btn.config(text="启动 VR 遥操")
            self.status_text.set("VR 遥操：关")

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

        # VR 总开关先置 False，让执行线程自然退出，再关闭网络服务。
        self.is_vr_control = False

        for label, fn in [
            ("推理控制器", self.inference.stop),
            ("VR 服务", self.dummy_server.stop),
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
    """Run the safety-invariant suite (tests/test_safety_invariants.py) before the GUI builds
    the robot. A failure means a safety regression and BLOCKS launch — the GUI can drive
    hardware, so it must not start on a broken release pipeline. Set
    HUMANOID_SKIP_SAFETY_PREFLIGHT=1 to bypass (logs a loud warning)."""
    if os.environ.get("HUMANOID_SKIP_SAFETY_PREFLIGHT") == "1":
        print("\n*** WARNING: safety pre-flight SKIPPED via HUMANOID_SKIP_SAFETY_PREFLIGHT=1 ***\n")
        return
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests"))
    print("[startup] running safety pre-flight (tests/test_safety_invariants.py)…")
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
    # Configure the logging that the controllers (e.g. real_world.inference_controller) emit on,
    # so their INFO/WARNING messages actually reach the console. Without this only a bare
    # last-resort WARNING handler exists and INFO/DEBUG are dropped. Level is overridable via
    # HUMANOID_LOG_LEVEL (e.g. DEBUG to surface the per-inference hot-path traces).
    logging.basicConfig(
        level=os.environ.get("HUMANOID_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

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
