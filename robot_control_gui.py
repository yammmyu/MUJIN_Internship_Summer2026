"""智元 G1 机器人控制 GUI 主入口。

功能按域拆分到 gui/ 包下的多个 Mixin，本文件只负责：
  - 组装 RobotControlGUI（多继承各 Mixin）
  - __init__：初始化机器人/相机/服务器与共享状态，搭建界面、启动后台线程
  - setup_ui：顶层布局
  - on_closing / main：生命周期与退出清理


测试顺序
Stage A — Automated checks (any machine with sim deps; no robot, no policy)
A1. SDK-free import

cd ~/Documents/Humanoid/humanoid
.venv/bin/python -c "import real_world.humanoid_env, real_world.sim_backend, real_world.inference_controller; print('import OK')"
Pass: prints import OK (proves the guarded SDK import works on a machine without a2d_sdk).

A2. Safety-invariant suite (the fakes test — no hardware). Save it and run:


.venv/bin/python /tmp/safety_suite.py   # the script from the last verification step
Pass: FULL SAFETY SUITE: ALL PASS (C1 C2 C3 C4 C5 H1).
This is your regression gate — re-run it after any change to humanoid_env.py/sim_backend.py. It checks: no-sim refusal, validate→release reaches robot, step ≤ cap, one-shot, E-stop latch+hold+refuse+reset, no-zero-right-arm.


Stage B — Sim-only runner, no policy (no robot)

.venv/bin/python scripts/sim_infer_eval.py --source replay \
  --recording recording021 --recordings ~/Downloads/recordings
Pass: PyBullet opens, the left arm tracks the recorded trajectory, exits clean. Watch for: arm starts at the recording's pose, motion is smooth, no IK unreachable spam.


Stage C — Policy round-trip (needs the policy server reachable)

python3 ping_inference_server.py --host 10.12.11.144 --port 9001   # reachability first
.venv/bin/python scripts/sim_infer_eval.py --source policy \
  --recording recording021 --recordings ~/Downloads/recordings \
  --host 10.12.11.144 --port 9001
Pass: predictions arrive, the sim arm follows the policy's output (not the recording). This validates the full inference→IK→sim path with zero hardware risk.


Stage D — Tune the safety limits BEFORE the GUI (critical, do not skip)
The defaults are guesses and must match your robot/workspace:

Workspace envelope — confirm your real left-EE poses fall inside WORKSPACE_AABB (humanoid_env.py):

.venv/bin/python -c "
import numpy as np; from real_world.sim_backend import load_trajectory
p,_,_,_,_ = load_trajectory('$HOME/Downloads/recordings','recording021')
print('x',p[:,0].min(),p[:,0].max()); print('y',p[:,1].min(),p[:,1].max()); print('z',p[:,2].min(),p[:,2].max())"
Set WORKSPACE_AABB to enclose real reachable space with margin. Pass: replaying a recording shows no "outside workspace" skips for known-good poses.
SELF_COLLISION_PENETRATION and (optional) calibrate_collisions() — confirm safe recordings validate cleanly and an obvious folded pose is rejected.
MAX_JOINT_STEP — start conservative (current 0.05 rad/tick ≈ 86°/s ceiling); only raise after a successful slow run.


Stage E — On the robot machine, GUI, sim preview only (never release)
Robot powered, physical E-stop in hand, arm workspace clear.

python robot_control_gui.py

启动仿真预览 → PyBullet opens and the sim matches the real arm's current pose (both arms, grippers, torso pitch). Pass: poses visibly match.
推理一次 → a prediction is produced (no motion).
仿真验证(整条) → status bar shows ✅ 仿真验证通过 (or ❌ … reason). Pass: the sim plays the trajectory; the real robot does NOT move.
Confirm the real arm stayed still throughout. Do not press 释放到真机 yet.


Stage F — Gated hardware release (physical E-stop in hand, finger ready)

Only after E passes. Start with the arm in open space.

推理一次 → 仿真验证(整条) → verify ✅ and watch the sim path is safe.
🚀 释放到真机 → the real arm should ramp from its current pose (no jump) and follow the validated path slowly.
Mid-motion, press ⛔ 急停 → arm must stop and hold immediately; status/log shows latch. Verify 复位急停 is required before another release works.
Verify re-pressing 释放到真机 without a new 仿真验证 is refused (no snap-back).
Verify a deliberately bad case: unplug/freeze a camera → 推理一次 should refuse with a stale-obs message.
Abort criteria at any hardware stage: any unexpected motion, a jump at release start, E-stop not holding, or the right arm twitching → hit the physical E-stop, stop, and recheck.

Two things I'd confirm before Stage F specifically:

The active-hold stop assumes move_arm(current_joints) actually holds; verify the firmware behavior on a slow motion first (Stage F step 3 is exactly that test).
The self-collision gate is best-effort (coarse meshes) — Stage F step 1's sim preview is your real visual check; don't rely on the gate alone.
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

from a2d_sdk.robot import RobotDds as Robot, CosineCamera as Camera, RobotController
from control_wheel_example import WheelController
from robot_info_server import create_robot_info_http_server, RobotInfo
from constants import *
from pico_vr.pico_vr_server.server import DummyServer

from kinematics import RobotCoordinateTransformer
from gui import (
    StyleMixin,
    CameraMixin,
    StatusMixin,
    CoordinateMixin,
    PickPlaceMixin,
    ManualControlMixin,
    VRMixin,
    MotionPlanningMixin,
)
from real_world import HumanoidEnv, InferenceController
from real_world.humanoid_env import RECORD_HZ
from real_world.sim_backend import SimEnv


class RobotControlGUI(
    StyleMixin,
    CameraMixin,
    StatusMixin,
    CoordinateMixin,
    PickPlaceMixin,
    ManualControlMixin,
    VRMixin,
    MotionPlanningMixin,
):
    def __init__(self, root, camera_mode="all"):
        self.root = root
        self.root.title("智元G1机器人控制界面")
        self.root.geometry("1500x950")
        self.root.minsize(1200, 800)
        self._setup_styles()

        # 初始化机器人（相机由 HumanoidEnv 持有，见下方 self.env）
        self.robot = Robot()
        self.robot_controller = RobotController()

        # Disable cameras during data_collection mode
        if camera_mode == "data":
            camera_names = ["hand_left", "hand_right", "head"]
        else:
            camera_names = []
        self.camera_mode = camera_mode

        # Slam
        rclpy.init(args=None)
        self.wheel_controller = WheelController()
        wheel_thread = threading.Thread(target=rclpy.spin, args=(self.wheel_controller,), daemon=True)
        wheel_thread.start()

        # 等待初始化
        time.sleep(1.0)

        # 相机图像缓存（兼容镜像）：唯一抓取者是 self.env；显示线程把 env 拉到的帧
        # 镜像到此处，供尚未切换到 env 的消费者（VR/坐标/运动规划）继续读取。
        self.camera_images = {
            # "hand_left": None,  # cache latest two images for inference
            # "hand_right": None,  # cache latest two images for inference
            "head": None,
            # "head_depth": None,  # cache latest one image
            # "head_center_fisheye": None
        }
        self.last_two_left_arm_joint_values = []
        # 最近两帧左臂末端位姿 state ([pos(3), quat xyzw(4), grip(1)])，供 EE 策略推理
        self.last_two_left_ee_states = []
        # 相机显示缩放尺寸（由 _rebuild_camera_display 根据排版动态更新）
        self.camera_tile_size = (320, 240)

        # 相机内参缓存
        self.camera_intrinsics = {
            # "hand_left": None,
            # "hand_right": None,
            "head": None,
            # "head_depth": None,
            # "head_center_fisheye": None
        }

        # update robot_info
        self.robot_info = RobotInfo()
        create_robot_info_http_server(self.robot_info)

        # HumanoidEnv：相机的唯一持有者与抓取者。每路相机一个 CosineCamera，按需订阅、
        # RECORD_HZ 抓取、空闲自动退订（关流）。cameras= 为「常开」相机（数据采集模式用），
        # GUI 普通模式传 [] -> 启动时不订阅任何相机、无视频流带宽。robot/controller 仍共享。
        # real=True: enables the release pipeline. Actions always run in the sim preview first;
        # 仿真验证 accumulates sim-validated substeps, and only 释放(单步)/释放(剩余)
        # (release_next_substep / release_remaining_substeps) drive them onto the robot.
        self.env = HumanoidEnv(
            robot=self.robot,
            robot_controller=self.robot_controller,
            cameras=camera_names,
            frequency=RECORD_HZ,
            real=True,
        )

        # In-process PyBullet preview. Built lazily on its own thread (it owns all p.* calls)
        # when the user presses "启动仿真预览"; attached to self.env.sim so the exec loop
        # drives it. None until launched.
        self._sim_thread = None
        self._sim_stop = threading.Event()

        # left_arm_ee_image policy inference（使用注入的 env，不再自建）
        self.inference = InferenceController(self.env, self.robot_info)

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

        # auto inference (thread/state now owned by self.inference)
        self.is_grabbing_target: bool = False

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
            'waist': ['腰部俯仰', '腰部升降'],
            'gripper': ['左夹爪', '右夹爪']
        }

        self.setup_ui()
        self.env.start()           # 启动相机采集 + 执行线程（GUI 持有 env 生命周期）
        self.inference.start()
        self.start_camera_thread()
        self.start_status_thread()
        self.start_vr_stream_thread()
        if camera_mode != "data":
            self.start_motion_planning_thread()

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
            # Match the sim to the physical robot on load: both arms, both grippers, torso
            # pitch. (URDF has no head/waist-lift joints, so those aren't mirrored.)
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
        """窗口关闭时的处理：依次释放各资源，最后强制退出进程。

        每个资源单独 try/except，避免前一个清理失败阻断后续；
        最后用 os._exit() 兜底，绕过被 DDS/rclpy 原生线程卡住的解释器退出。
        """
        # 防止重复进入（信号 + 窗口关闭可能同时触发）
        if getattr(self, "_is_closing", False):
            return
        self._is_closing = True

        for label, fn in [
            ("Motion Planning TCP", self.stop_motion_planning),
            ("VR 串流服务器", self.dummy_server.stop),
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
    app = RobotControlGUI(root, camera_mode)  # 样式由 _setup_styles 统一配置
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
