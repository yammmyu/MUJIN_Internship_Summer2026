"""智元 G1 机器人控制 GUI 主入口。

功能按域拆分到 gui/ 包下的多个 Mixin，本文件只负责：
  - 组装 RobotControlGUI（多继承各 Mixin）
  - __init__：初始化机器人/相机/服务器与共享状态，搭建界面、启动后台线程
  - setup_ui：顶层布局
  - on_closing / main：生命周期与退出清理
"""

import os
import signal
import threading
import time
import tkinter as tk
from tkinter import ttk

import rclpy

from a2d_sdk.robot import RobotDds as Robot, CosineCamera as Camera, RobotController
from control_wheel_example import WheelController
from robot_info_server import create_robot_info_http_server, RobotInfo
from constants import *
from pico_vr.pico_vr_server.server import DummyServer

from kinematics import RobotCoordinateTransformer
from MDM_data_collection.robot_data_collect import RobotDataCollector
from gui import (
    StyleMixin,
    CameraMixin,
    StatusMixin,
    CoordinateMixin,
    PickPlaceMixin,
    InferenceMixin,
    ManualControlMixin,
    VRMixin,
    DataCollectionMixin,
)


class RobotControlGUI(
    StyleMixin,
    CameraMixin,
    StatusMixin,
    CoordinateMixin,
    PickPlaceMixin,
    InferenceMixin,
    ManualControlMixin,
    VRMixin,
    DataCollectionMixin,
):
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

        # 相机图像缓存
        self.camera_images = {
            "hand_left": None,  # cache latest two images for inference
            "hand_right": None,  # cache latest two images for inference
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
            "hand_right": None,
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

        # 数据采集器（共享已有的 robot / camera 实例，不重复初始化硬件）
        # get_camera_frame reads from camera_images — the single SDK reader is
        # start_camera_thread(); the recorder must not compete with a second reader.
        def _get_camera_frame(name):
            img = self.camera_images.get(name)
            if isinstance(img, list) and img:
                return img[-1]
            return None

        self.data_collector = RobotDataCollector(
            output_dir="recordings",
            robot=self.robot,
            robot_controller=self.robot_controller,
            get_camera_frame=_get_camera_frame,
        )

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
        self.setup_data_collection_panel(right_notebook)

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
