import math
import os
import threading
import time
import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageTk


class CameraMixin:
    """相机视图：勾选显示、动态网格排版、图像更新线程、保存与内参。"""

    def setup_camera_panel(self, parent):
        """设置相机图像面板：RGBD 控制条 + 相机勾选条 + 动态网格显示区。"""
        camera_frame = ttk.LabelFrame(parent, text="  📷  相机视图  ")
        camera_frame.pack(fill=tk.BOTH, expand=True)

        # 可显示的相机及其中文名（须为 env 构造 Camera() 时声明的相机；
        # data 模式下只有 3 路，深度/鱼眼不可用，故按 env.cameras 过滤）
        self.available_cameras = [
            name for name in
            ["head", "head_depth", "hand_left", "hand_right", "head_center_fisheye"]
            if name in self.env.cameras]
        self.camera_titles = {
            "head": "头部相机",
            "head_depth": "深度相机",
            "hand_left": "左手相机",
            "hand_right": "右手相机",
            "head_center_fisheye": "头部鱼眼相机",
        }

        # ---- RGBD 工具条 ----
        toolbar = ttk.Frame(camera_frame)
        toolbar.pack(fill=tk.X, padx=10, pady=(8, 6))

        self.rgbd_enabled_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(toolbar, text="启用 3D 坐标点击",
                        variable=self.rgbd_enabled_var,
                        command=self.toggle_rgbd_mode).pack(side=tk.LEFT)

        ttk.Label(toolbar, text="目标相机:").pack(side=tk.LEFT, padx=(16, 4))
        self.target_camera_var = tk.StringVar(value="head")
        camera_combo = ttk.Combobox(
            toolbar, textvariable=self.target_camera_var,
            values=[n for n in ["head", "head_depth", "hand_left", "hand_right"]
                    if n in self.available_cameras],
            state="readonly", width=14)
        camera_combo.pack(side=tk.LEFT)
        camera_combo.bind('<<ComboboxSelected>>', self.on_camera_selection_change)

        ttk.Button(toolbar, text="清除坐标记录",
                   style="Muted.TButton",
                   command=self.clear_coordinate_records
                   ).pack(side=tk.RIGHT)

        # ---- 相机勾选条：选择要显示的相机 ----
        select_bar = ttk.Frame(camera_frame)
        select_bar.pack(fill=tk.X, padx=10, pady=(0, 6))
        ttk.Label(select_bar, text="显示相机:").pack(side=tk.LEFT, padx=(0, 4))
        # 默认显示当前缓存中已有的相机（head / head_depth）
        self.camera_display_vars = {
            name: tk.BooleanVar(value=name in self.camera_images)
            for name in self.available_cameras
        }
        for name in self.available_cameras:
            ttk.Checkbutton(
                select_bar, text=self.camera_titles[name],
                variable=self.camera_display_vars[name],
                command=self._rebuild_camera_display).pack(side=tk.LEFT, padx=4)

        # ---- 相机显示区（动态网格） ----
        self.camera_labels = {}
        self.camera_display_area = ttk.Frame(camera_frame)
        self.camera_display_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        self._rebuild_camera_display()

    def _compute_camera_tile_size(self, ncols, nrows):
        """根据可用面板尺寸与网格规模，计算保持 4:3 的单格图像尺寸。"""
        try:
            self.root.update_idletasks()
            win_w = self.root.winfo_width()
            win_h = self.root.winfo_height()
        except Exception:
            win_w = win_h = 0
        if win_w < 100:
            win_w = 1500
        if win_h < 100:
            win_h = 950
        # 左侧相机面板约占窗口一半宽；高度扣除标题/工具条/状态栏
        panel_w = max(360, win_w // 2 - 60)
        panel_h = max(360, win_h - 220)
        # 每格扣除标题栏(~44)与内边距
        cell_w = max(160, panel_w // ncols - 16)
        cell_h = max(120, panel_h // nrows - 44)
        # 保持 4:3 比例，取受限的一边
        tile_w = min(cell_w, int(cell_h * 4 / 3))
        tile_h = int(tile_w * 3 / 4)
        return max(120, tile_w), max(90, tile_h)

    def _rebuild_camera_display(self):
        """根据用户勾选的相机重建显示区：动态网格排版 + 自适应缩放尺寸。"""
        # 当前选中的相机（按固定顺序）
        selected = [name for name in self.available_cameras
                    if self.camera_display_vars[name].get()]

        # 同步缓存字典的键：新增的置 None，取消的移除（停止抓取）
        for name in selected:
            self.camera_images.setdefault(name, None)
            self.camera_intrinsics.setdefault(name, None)
        for name in list(self.camera_images.keys()):
            if name not in selected:
                self.camera_images.pop(name, None)
                self.camera_intrinsics.pop(name, None)

        # 清空旧控件并重置标签字典
        for child in self.camera_display_area.winfo_children():
            child.destroy()
        self.camera_labels = {}

        if not selected:
            self.camera_tile_size = (320, 240)
            return

        # 计算网格列数（1→1，2~4→2，5→3）与每格缩放尺寸
        n = len(selected)
        ncols = 1 if n == 1 else 2 if n <= 4 else 3
        nrows = math.ceil(n / ncols)
        self.camera_tile_size = self._compute_camera_tile_size(ncols, nrows)

        # 让各行各列均分空间
        for c in range(ncols):
            self.camera_display_area.columnconfigure(c, weight=1, uniform="cam")
        for r in range(nrows):
            self.camera_display_area.rowconfigure(r, weight=1, uniform="cam")
        # 取消旧的多余行列权重
        for c in range(ncols, ncols + 3):
            self.camera_display_area.columnconfigure(c, weight=0, uniform="")
        for r in range(nrows, nrows + 5):
            self.camera_display_area.rowconfigure(r, weight=0, uniform="")

        for idx, name in enumerate(selected):
            r, c = divmod(idx, ncols)
            card = ttk.Frame(self.camera_display_area)
            card.grid(row=r, column=c, sticky="nsew", padx=4, pady=4)

            header = ttk.Frame(card)
            header.pack(fill=tk.X, pady=(0, 4))
            ttk.Label(header, text=f"📷  {self.camera_titles.get(name, name)}",
                      style="Section.TLabel").pack(side=tk.LEFT, anchor=tk.W)
            ttk.Button(header, text="💾 保存图片",
                       style="Primary.TButton",
                       command=lambda n=name: self.save_camera_frame(n)
                       ).pack(side=tk.RIGHT)

            label = ttk.Label(card, borderwidth=1, relief="solid",
                              background="#222", anchor=tk.CENTER)
            label.pack(fill=tk.BOTH, expand=True)
            self.camera_labels[name] = label
            self._bind_camera_click(name)

        # 重新应用鼠标样式
        self.toggle_rgbd_mode()

    def _selected_display_cameras(self):
        """当前勾选要显示的相机（按 available_cameras 固定顺序）。"""
        return [name for name in self.available_cameras
                if self.camera_display_vars[name].get()]

    def start_camera_thread(self):
        """启动相机显示线程：纯消费者。

        唯一抓取者是 self.env（按需开关、统一 RECORD_HZ 抓取）。本线程对每个勾选
        的相机调用 env.get_frame(name)（即「请求」该相机，使其保活并返回最新帧），
        渲染到界面，并把帧/内参镜像回 self.camera_images / self.camera_intrinsics，
        供尚未切换到 env 的消费者（VR/坐标/运动规划）继续读取。
        """
        def update_camera_images():
            while True:
                try:
                    for camera_name in self._selected_display_cameras():
                        # 向 env 请求该相机（保活）并取最新帧；首帧或刚开机时可能为 None
                        image = self.env.get_frame(camera_name)
                        if image is not None:
                            # 镜像最近两帧（list 结构），供 VR / 保存图片 / 3D 点击消费
                            prev = self.camera_images.get(camera_name)
                            if isinstance(prev, list) and len(prev) == 2:
                                self.camera_images[camera_name] = [prev[-1], image.copy()]
                            else:
                                self.camera_images[camera_name] = [image.copy(), image.copy()]

                            self.update_camera_display(camera_name, image)

                        # 内参镜像（env 懒加载并缓存，这里取一次填入兼容字典）
                        if self.camera_intrinsics.get(camera_name) is None:
                            self.camera_intrinsics[camera_name] = self.env.get_intrinsics(camera_name)

                    time.sleep(0.033)  # ~30 Hz, matches recording rate
                except Exception as e:
                    print(f"相机更新错误: {e}")
                    time.sleep(1)

        camera_thread = threading.Thread(target=update_camera_images, daemon=True)
        camera_thread.start()

    def save_camera_frame(self, camera_name):
        """将指定相机的当前帧保存为本地 jpg 图片。"""
        try:
            image = self.camera_images.get(camera_name)
            # 部分相机缓存的是最近两帧的列表，取最新一帧
            if isinstance(image, (list, tuple)):
                image = image[-1] if image else None
            if image is None or not isinstance(image, np.ndarray) or image.size == 0:
                self.status_text.set(f"保存失败：{camera_name} 相机暂无图像")
                return

            # 转换为可保存的 RGB PIL 图像（与显示逻辑保持一致）
            if len(image.shape) == 3 and image.shape[2] == 3:
                arr = image if image.dtype == np.uint8 else image.astype(np.uint8)
                if hasattr(self, "bgr_cameras") and camera_name in self.bgr_cameras:
                    arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(arr)
            else:
                # 深度 / 灰度图：归一化并做伪彩色映射后保存
                depth_2d = image[:, :, 0] if len(image.shape) == 3 else image
                if depth_2d.dtype == np.uint16:
                    valid_pixels = depth_2d[depth_2d > 0]
                    if valid_pixels.size > 0:
                        min_val, max_val = np.min(valid_pixels), np.max(valid_pixels)
                    else:
                        min_val, max_val = 0, 1
                    if max_val > min_val:
                        depth_norm = ((depth_2d.astype(np.float32) - min_val) /
                                      (max_val - min_val) * 255).astype(np.uint8)
                    else:
                        depth_norm = np.zeros_like(depth_2d, dtype=np.uint8)
                else:
                    depth_norm = depth_2d.astype(np.uint8)
                depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_TURBO)
                depth_rgb = cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(depth_rgb)

            save_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "saved_images")
            os.makedirs(save_dir, exist_ok=True)
            filename = f"{camera_name}_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
            filepath = os.path.join(save_dir, filename)
            pil_image.convert("RGB").save(filepath, "JPEG", quality=95)
            self.status_text.set(f"已保存图片: {filepath}")
            print(f"已保存图片: {filepath}")
        except Exception as e:
            self.status_text.set(f"保存图片失败: {e}")
            print(f"保存图片失败: {e}")

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
                # 原始深度数据的缓存改由 start_camera_thread 统一以 list 结构维护

            else:
                print(f"不支持shape: {image.shape}")
                return

            # ================= resize =================
            # 使用按排版动态计算的尺寸，避免多相机时画面超出显示区
            tile_w, tile_h = getattr(self, "camera_tile_size", (320, 240))
            pil_image = pil_image.resize((tile_w, tile_h), Image.Resampling.LANCZOS)


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
        try:
            label.config(image=photo)
            label.image = photo  # 保持引用
        except tk.TclError:
            # 标签可能在重建显示区时已被销毁，忽略这次延迟回调
            pass

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

    def _latest_camera_frame(self, camera_name):
        """从 camera_images 取指定相机的最新一帧。

        camera_images 内所有相机统一为「最近两帧」的 list 结构，
        此处返回最新一帧（list[-1]）；兼容历史的单帧 ndarray。
        """
        # 未来切换到 env 单一数据源（按需开关相机）时改为：
        # return self.env.get_frame(camera_name)
        entry = self.camera_images.get(camera_name)
        if isinstance(entry, (list, tuple)):
            return entry[-1] if entry else None
        return entry

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
