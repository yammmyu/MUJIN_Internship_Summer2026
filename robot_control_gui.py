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

# The GUI is always available; the hardware stack (ROS, the a2d SDK, the PyBullet
# sim, the ZeroMQ VR server, the policy env) only exists on the robot machine. Guard
# those imports so the console can also launch in --demo mode on any laptop, where
# the whole stack is replaced by gui.demo_backend. Names left as None here are only
# ever dereferenced on the real (non-demo) path.
from gui import (StyleMixin, CameraMixin, InferenceMixin, DetectorTuningMixin,
                 VRMixin, DataCollectionMixin, EvalMixin)
from real_world.timing import RECORD_HZ

try:
    import rclpy
    from a2d_sdk.robot import RobotDds as Robot, RobotController, Slam
    from examples.control_wheel_example import WheelController
    from servers.robot_info_server import create_robot_info_http_server, RobotInfo
    from pico_vr.pico_vr_server.server import DummyServer
    from real_world import HumanoidEnv, InferenceController
    from real_world.sim_backend import SimEnv
    _HW_IMPORT_ERROR = None
except Exception as _e:                       # missing SDK / ROS / zmq / pybullet
    rclpy = None
    Robot = RobotController = Slam = WheelController = None
    create_robot_info_http_server = RobotInfo = DummyServer = None
    HumanoidEnv = InferenceController = SimEnv = None
    _HW_IMPORT_ERROR = _e


class RobotControlGUI(StyleMixin, CameraMixin, InferenceMixin, DetectorTuningMixin,
                      VRMixin, DataCollectionMixin, EvalMixin):
    def __init__(self, root, camera_mode="all", demo=False):
        self.root = root
        self.demo = demo
        self.root.title("Mujin Humanoid Control Console" + ("  —  DEMO" if demo else ""))
        self.root.geometry("1600x950")
        self.root.minsize(1180, 720)
        self._setup_styles()

        if demo:
            self._init_demo_backend(camera_mode)
        else:
            self._init_real_backend(camera_mode)

    # ------------------------------------------------------------------ #
    #  Backend construction (real hardware vs. hardware-free demo)         #
    # ------------------------------------------------------------------ #
    def _init_real_backend(self, camera_mode):

        # 机器人（相机由 HumanoidEnv 持有，见下方 self.env）
        self.robot = Robot()
        self.robot_controller = RobotController()

        # Slam 必须在进程内实例化，否则 arm/head/waist 的 joint_states 会一直冻结，
        # 而 VR 执行线程每拍都读取这三者（并 assert 非空）。此处仅创建以解冻关节状态，
        # 不切换底盘导航模式，避免遥操过程中底盘被自动导航带动。
        try:
            self.slam = Slam()
            print("SLAM initialized (joint states unfrozen)")
        except Exception as e:
            self.slam = None
            print(f"⚠️ SLAM init failed; joint states needed for VR teleop may be unavailable: {e}")

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

        # ---- VR data channel ----
        # Uplink (:5556): headset -> _handle_vr_joints (the only place VR poses reach the robot).
        # Downlink (:5555): start_vr_stream_thread composites the camera view and pushes it back.
        # use_default_image=False: wait for real camera frames rather than a placeholder.
        self.dummy_server = DummyServer(
            on_joints=self._handle_vr_joints,
            use_default_image=False,
        )
        self._finish_init(start_vr_stream=True)

    def _init_demo_backend(self, camera_mode):
        """Hardware-free backend: every collaborator is a synthetic stand-in from
        gui.demo_backend, so the exact same UI runs on any laptop for recorded demos."""
        from gui.demo_backend import (
            DemoRobot, DemoRobotController, DemoSlam, DemoWheelController,
            DemoDummyServer, DemoEnv, DemoInference, DemoEvalWriter)
        from gui.eval_panel import EVAL_DIR

        self.camera_mode = camera_mode
        self.robot = DemoRobot()
        self.robot_controller = DemoRobotController()
        self.slam = DemoSlam()
        self.wheel_controller = DemoWheelController()
        self.robot_info = None
        self.camera_images = {}
        self.camera_tile_size = (320, 240)

        self.env = DemoEnv(output_dir="recordings")
        self._sim_thread = None
        self._sim_stop = threading.Event()

        # A synthetic eval session that fills the dashboard live during an auto run.
        self._demo_eval = DemoEvalWriter(EVAL_DIR / "demo_session.jsonl")
        self.inference = DemoInference(self.env, eval_writer=self._demo_eval)
        self.dummy_server = DemoDummyServer()

        # In demo mode "Start sim preview" just flags the sim ready (no PyBullet needed),
        # so auto-run proceeds without hardware or a display.
        def _demo_launch_sim():
            self.env.sim = object()
            self.status_text.set("Sim preview ready (demo)")
        self.launch_sim = _demo_launch_sim
        self._stop_sim = lambda: None

        self._finish_init(start_vr_stream=False)

    def _finish_init(self, start_vr_stream):
        """Shared tail for both backends: restore tuning, init shared state, build the
        UI, and start the (already-constructed) collaborators."""
        # Restore persisted tuning before the panel is built so widgets show live values.
        self._load_and_apply_tuning()

        # Left-gripper state (right gripper is not exposed in the UI; it stays put).
        self.left_gripper_pos = 0.0
        self.right_gripper_pos = 0.0

        # ---- VR teleop shared state (read/written by _handle_vr_joints / exec thread) ----
        self.is_vr_control = False
        self.vr_execution_thread = None
        self.vr_actions = []
        self.previous_vr_positions = []
        self.last_joint_update_timestamp = 0.0
        self.vr_buttons_pressed = set()
        self.vr_axes = []

        # Status bar.
        self.status_text = tk.StringVar()
        self.status_text.set("Ready" + ("  ·  demo mode" if self.demo else ""))
        self.status_label = None

        self.setup_ui()
        self.env.start()           # camera capture + exec threads (GUI owns env lifecycle)
        self.inference.start()
        self.start_camera_thread()

        self.dummy_server.start()
        if start_vr_stream:
            self.start_vr_stream_thread()

    def setup_ui(self):
        """Branded top bar + tabbed workspace (Console / VR teleop / Evaluation) + status bar."""
        # ===== Top app bar =====
        header = ttk.Frame(self.root, style="Header.TFrame")
        header.pack(fill=tk.X)
        hinner = ttk.Frame(header, style="Header.TFrame")
        hinner.pack(fill=tk.X, padx=20, pady=12)
        # Brand mark + product name.
        ttk.Label(hinner, text="MUJIN", style="Brand.TLabel").pack(side=tk.LEFT)
        ttk.Label(hinner, text="Humanoid Control Console",
                  style="Title.TLabel").pack(side=tk.LEFT, padx=(12, 0))
        ttk.Label(hinner, text="G1 dual-arm  ·  policy inference & teleop",
                  style="Subtitle.TLabel").pack(side=tk.LEFT, padx=14, pady=(6, 0))
        if self.demo:
            ttk.Label(hinner, text="DEMO MODE  ·  no hardware",
                      style="DemoPill.TLabel").pack(side=tk.RIGHT)
        else:
            ttk.Label(hinner, text="LIVE  ·  hardware connected",
                      style="Pill.TLabel").pack(side=tk.RIGHT)
        ttk.Separator(self.root, orient="horizontal").pack(fill=tk.X)

        # ===== Status bar (built first, pinned to the bottom) =====
        status_bar = ttk.Frame(self.root)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X, padx=12, pady=(4, 10))
        self.status_label = ttk.Label(
            status_bar, textvariable=self.status_text,
            style="Status.TLabel", anchor=tk.W)
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(status_bar, text="Clear", style="Muted.TButton",
                   command=lambda: self.status_text.set("Ready")
                   ).pack(side=tk.RIGHT, padx=(8, 0))

        # ===== Workspace: top-level notebook =====
        tabs = ttk.Notebook(self.root)
        tabs.pack(fill=tk.BOTH, expand=True, padx=12, pady=8)

        # ---- Tab 1: Console (camera views + inference control) ----
        body = ttk.Frame(tabs)
        tabs.add(body, text="   Console   ")

        left_frame = ttk.Frame(body)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))

        # The inference control column is fixed-width (its inner canvas reserves the width it
        # needs); the camera view takes the remaining space and stretches with the window.
        right_frame = ttk.LabelFrame(body, text="  Inference control  ")
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, expand=False, padx=(8, 0))

        self.setup_camera_panel(left_frame)         # left: camera views
        self.setup_inference_panel(right_frame)     # right: grippers + inference + substep monitor

        # ---- Tab 2: Detector tuning (live boxed head-cam view + LabelGate thresholds) ----
        tune_tab = ttk.Frame(tabs)
        tabs.add(tune_tab, text="   Detector tuning   ")
        self.setup_detector_tuning_panel(tune_tab)

        # ---- Tab 3: VR teleop (toggle + sensitivity + data collection) ----
        vr_tab = ttk.Frame(tabs)
        tabs.add(vr_tab, text="   VR teleop   ")
        self.setup_vr_panel(vr_tab)

        # ---- Tab 4: Evaluation dashboard (live success-rate KPIs) ----
        eval_tab = ttk.Frame(tabs)
        tabs.add(eval_tab, text="   Evaluation   ")
        self.setup_eval_panel(eval_tab)

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
            top, text="Start VR teleop", style="Primary.TButton",
            command=self._toggle_vr,
        )
        self.tip(self.vr_toggle_btn,
                 "Arm/disarm VR teleoperation. When ON, hold L_Y or R_B on the controllers to "
                 "drive the arms. The headset must be connected first (uplink :5556).")
        self.vr_toggle_btn.pack(side=tk.LEFT)

        # 灵敏度参数面板需要一个 Notebook 容器（setup_vr_params_panel 调 parent.add）。
        nb = ttk.Notebook(left)
        nb.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
        self.setup_vr_params_panel(nb)

        # Right: data collection (record controls + live EE pose), fixed width.
        right = ttk.LabelFrame(parent, text="  Data collection  ")
        right.pack(side=tk.RIGHT, fill=tk.Y, expand=False, padx=(8, 0))
        self.setup_data_collection_panel(right)

    def _toggle_vr(self):
        """翻转 VR 遥操总开关并同步按钮/状态栏文案。"""
        self.is_vr_control = not self.is_vr_control
        if self.is_vr_control:
            self.vr_toggle_btn.config(text="Stop VR teleop")
            self.status_text.set("VR teleop: ON  (hold L_Y / R_B to move the arms)")
        else:
            self.vr_toggle_btn.config(text="Start VR teleop")
            self.status_text.set("VR teleop: OFF")

    # ===================== sim preview (in-process PyBullet) =====================
    def launch_sim(self):
        """Start the PyBullet preview on its own thread (idempotent)."""
        if self._sim_thread is not None and self._sim_thread.is_alive():
            self.status_text.set("Sim preview already running")
            return
        self._sim_stop.clear()
        self._sim_thread = threading.Thread(target=self._sim_loop, daemon=True)
        self._sim_thread.start()
        self.status_text.set("Sim preview started")

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
        """Release each resource in order on window close, then force-exit the process.

        Each resource is torn down under its own try/except so one failure can't block the
        rest; os._exit() is the backstop for an interpreter wedged on a native DDS/rclpy thread.
        """
        if getattr(self, "_is_closing", False):     # signal + window-close may both fire
            return
        self._is_closing = True

        # Flip the VR master switch off first so the exec thread exits, then close the services.
        self.is_vr_control = False

        steps = [
            ("inference controller", self.inference.stop),
            ("VR service", self.dummy_server.stop),
            ("sim preview", self._stop_sim),     # stop the sim-stepping thread before the env
            ("environment", self.env.stop),      # stop capture/exec threads before its cameras
            ("robot", self.robot.shutdown),
            ("ROS node", self.wheel_controller.destroy_node),
        ]
        if getattr(self, "_demo_eval", None) is not None:
            steps.insert(0, ("demo eval", self._demo_eval.stop))
        if rclpy is not None and not self.demo:
            steps.append(("rclpy", rclpy.shutdown))

        for label, fn in steps:
            try:
                fn()
            except Exception as e:
                print(f"error closing {label}: {e}")

        try:
            self.root.destroy()
        except Exception as e:
            print(f"error destroying window: {e}")
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
    # Console AND a persistent file (infer_logs/gui.log) so offline tools can read the run — e.g.
    # scripts/eval_trials.py tails it to auto-count "[recovery] missed grasp" retreats per trial.
    _log_handlers = [logging.StreamHandler()]
    try:
        from real_world.timing import TRACE_DIR
        TRACE_DIR.mkdir(parents=True, exist_ok=True)
        _log_handlers.append(logging.FileHandler(TRACE_DIR / "gui.log"))
    except Exception as _e:      # never let logging setup block launch
        print(f"[startup] could not open gui.log file handler: {_e}")
    logging.basicConfig(
        level=os.environ.get("HUMANOID_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=_log_handlers,
    )

    parser = argparse.ArgumentParser(description="Mujin Humanoid Control Console")
    parser.add_argument("--data", action="store_true",
                        help="data-collection camera mode (3 cameras always on)")
    parser.add_argument("--demo", action="store_true",
                        help="hardware-free demo mode: synthetic robot/cameras/eval, "
                             "no SDK/ROS/robot required (for recorded demos & UI work)")
    args = parser.parse_args()

    camera_mode = "data" if args.data else "all"

    if not args.demo and _HW_IMPORT_ERROR is not None:
        print(f"\n*** Hardware stack unavailable ({type(_HW_IMPORT_ERROR).__name__}: "
              f"{_HW_IMPORT_ERROR}).\n*** This machine can only run the console in demo mode. "
              f"Launch with:  python robot_control_gui.py --demo\n")
        sys.exit(1)

    if not args.demo:
        _safety_preflight()    # block launch if the safety invariants regressed
    else:
        print("[startup] DEMO MODE — no hardware; synthetic robot/cameras/eval.\n")

    root = tk.Tk()
    app = RobotControlGUI(root, camera_mode, demo=args.demo)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)

    # Ctrl+C: route SIGINT to the main thread's on_closing.
    signal.signal(signal.SIGINT, lambda *_: app.on_closing())

    # Heartbeat: periodically hand control back to the Python interpreter, else Tk's C
    # event loop won't service a pending SIGINT while idle.
    def _tick():
        root.after(200, _tick)
    root.after(200, _tick)

    try:
        root.mainloop()
    except KeyboardInterrupt:
        app.on_closing()


if __name__ == "__main__":
    main()
