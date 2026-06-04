"""Standalone data-collection GUI.

Entry point (run from the humanoid/ project root):
    python -m MDM_data_collection.data_collection_gui
    python MDM_data_collection/data_collection_gui.py
"""

import math
import os
import shutil
import signal
import threading
import time
import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from real_world.humanoid_env import HumanoidEnv, RECORD_CAMERAS as CAMERA_NAMES

_CAMERA_TITLES = {
    "hand_left":  "左手相机",
    "hand_right": "右手相机",
    "head":       "头部相机",
}


# ── Styles ────────────────────────────────────────────────────────────────────

def _apply_styles(root: tk.Tk) -> None:
    style = ttk.Style()
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    PRIMARY       = "#1976d2"
    PRIMARY_HOVER = "#1565c0"
    SUCCESS       = "#2e7d32"
    SUCCESS_HOVER = "#1b5e20"
    WARN          = "#ef6c00"
    WARN_HOVER    = "#e65100"
    DANGER        = "#c62828"
    DANGER_HOVER  = "#8e0000"
    MUTED         = "#546e7a"
    MUTED_HOVER   = "#37474f"
    TEXT          = "#1a1a1a"
    BG            = "#f5f6f8"

    root.configure(bg=BG)
    style.configure("TFrame",    background=BG)
    style.configure("TLabel",    background=BG, foreground=TEXT,    font=("Microsoft YaHei", 10))
    style.configure("Title.TLabel",    background=BG, foreground=PRIMARY, font=("Microsoft YaHei", 18, "bold"))
    style.configure("Subtitle.TLabel", background=BG, foreground="#666",  font=("Microsoft YaHei", 10))
    style.configure("Section.TLabel",  background=BG, foreground=TEXT,    font=("Microsoft YaHei", 11, "bold"))
    style.configure("Value.TLabel",    background=BG, foreground="#0d47a1", font=("Consolas", 10))
    style.configure("Status.TLabel",   background="#263238", foreground="#e0f7fa",
                    font=("Consolas", 10), padding=(8, 4))
    style.configure("TLabelframe",       background=BG, borderwidth=1, relief="solid")
    style.configure("TLabelframe.Label", background=BG, foreground=PRIMARY,
                    font=("Microsoft YaHei", 10, "bold"))

    base_btn = dict(font=("Microsoft YaHei", 10), padding=(10, 6), borderwidth=0)
    style.configure("TButton", **base_btn)

    def _btn(name, base, hover, fg="white"):
        style.configure(name, background=base, foreground=fg, **base_btn)
        style.map(name, background=[("active", hover), ("pressed", hover)],
                  foreground=[("active", fg)])

    _btn("Primary.TButton", PRIMARY,  PRIMARY_HOVER)
    _btn("Success.TButton", SUCCESS,  SUCCESS_HOVER)
    _btn("Warn.TButton",    WARN,     WARN_HOVER)
    _btn("Danger.TButton",  DANGER,   DANGER_HOVER)
    _btn("Muted.TButton",   MUTED,    MUTED_HOVER)


# ── Application ───────────────────────────────────────────────────────────────

class DataCollectionGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("数据采集控制台")
        self.root.geometry("1280x800")
        self.root.minsize(900, 600)

        _apply_styles(root)

        # HumanoidEnv is the single SDK owner; collect-only (no exec thread) for
        # data collection. Pin the record cameras so previews + recording stay live.
        self.collector = HumanoidEnv(cameras=CAMERA_NAMES, output_dir="recordings")
        self.collector.start(run_exec=False)

        # Recording state
        self._is_recording = False
        self._is_stopped   = False
        self._current_name = tk.StringVar(value="—")
        self._status_var   = tk.StringVar(value="空闲")
        self._status_bar   = tk.StringVar(value="就绪")

        # Camera display
        self.camera_labels:   dict[str, ttk.Label] = {}
        self.camera_tile_size = (320, 240)

        # EE position display vars
        self._left_pos_var   = tk.StringVar(value="—")
        self._left_quat_var  = tk.StringVar(value="—")
        self._right_pos_var  = tk.StringVar(value="—")
        self._right_quat_var = tk.StringVar(value="—")

        self._build_ui()
        self._start_camera_thread()
        self._start_ee_pos_thread()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        header = ttk.Frame(self.root)
        header.pack(fill=tk.X, padx=16, pady=(12, 4))
        ttk.Label(header, text="数据采集控制台",
                  style="Title.TLabel").pack(side=tk.LEFT)
        ttk.Label(header, text="Robot Data Collection Console",
                  style="Subtitle.TLabel").pack(side=tk.LEFT, padx=12, pady=(8, 0))

        status_bar = ttk.Frame(self.root)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X, padx=12, pady=(4, 10))
        ttk.Label(status_bar, textvariable=self._status_bar,
                  style="Status.TLabel", anchor=tk.W).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(status_bar, text="清除", style="Muted.TButton",
                   command=lambda: self._status_bar.set("就绪")).pack(side=tk.RIGHT, padx=(8, 0))

        body = ttk.Frame(self.root)
        body.pack(fill=tk.BOTH, expand=True, padx=12, pady=4)

        left_panel = ttk.Frame(body)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))

        right_panel = ttk.Frame(body)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False, padx=(8, 0))

        self._build_camera_panel(left_panel)
        self._build_control_panel(right_panel)

        self.root.bind("<r>",      lambda _: self.recording_start())
        self.root.bind("<R>",      lambda _: self.recording_start())
        self.root.bind("<s>",      lambda _: self.recording_stop())
        self.root.bind("<S>",      lambda _: self.recording_stop())
        self.root.bind("<Return>", lambda _: self.recording_save())
        self.root.bind("<d>",      lambda _: self.recording_delete())
        self.root.bind("<D>",      lambda _: self.recording_delete())

    def _build_camera_panel(self, parent: ttk.Frame) -> None:
        cam_frame = ttk.LabelFrame(parent, text="  📷  相机视图  ")
        cam_frame.pack(fill=tk.BOTH, expand=True)

        self._camera_area = ttk.Frame(cam_frame)
        self._camera_area.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self._rebuild_camera_grid()

    def _rebuild_camera_grid(self) -> None:
        for child in self._camera_area.winfo_children():
            child.destroy()
        self.camera_labels = {}

        n     = len(CAMERA_NAMES)
        ncols = 1 if n == 1 else 2 if n <= 4 else 3
        nrows = math.ceil(n / ncols)

        try:
            self.root.update_idletasks()
            win_w = max(900, self.root.winfo_width())
            win_h = max(600, self.root.winfo_height())
        except Exception:
            win_w, win_h = 1280, 800

        panel_w = max(360, win_w * 2 // 3 - 60)
        panel_h = max(360, win_h - 180)
        cell_w  = max(160, panel_w // ncols - 16)
        cell_h  = max(120, panel_h // nrows - 44)
        tile_w  = min(cell_w, int(cell_h * 4 / 3))
        tile_h  = int(tile_w * 3 / 4)
        self.camera_tile_size = (max(120, tile_w), max(90, tile_h))

        for c in range(ncols):
            self._camera_area.columnconfigure(c, weight=1, uniform="cam")
        for r in range(nrows):
            self._camera_area.rowconfigure(r, weight=1, uniform="cam")

        for idx, name in enumerate(CAMERA_NAMES):
            r, c = divmod(idx, ncols)
            card = ttk.Frame(self._camera_area)
            card.grid(row=r, column=c, sticky="nsew", padx=4, pady=4)

            ttk.Label(card, text=f"📷  {_CAMERA_TITLES.get(name, name)}",
                      style="Section.TLabel").pack(anchor=tk.W, pady=(0, 4))

            lbl = ttk.Label(card, borderwidth=1, relief="solid",
                            background="#222", anchor=tk.CENTER)
            lbl.pack(fill=tk.BOTH, expand=True)
            self.camera_labels[name] = lbl

    def _build_control_panel(self, parent: ttk.Frame) -> None:
        canvas = tk.Canvas(parent, highlightthickness=0, bg="#f5f6f8", width=380)
        sb     = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        body   = ttk.Frame(canvas)
        body.bind("<Configure>",
                  lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=body, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        # ── Recording status ──
        sec_info = ttk.LabelFrame(body, text="  ⏺  录制状态  ")
        sec_info.pack(fill=tk.X, padx=10, pady=(10, 6))

        info_grid = ttk.Frame(sec_info)
        info_grid.pack(fill=tk.X, padx=8, pady=8)
        info_grid.columnconfigure(1, weight=1)

        ttk.Label(info_grid, text="当前录制:").grid(row=0, column=0, sticky="w", pady=3)
        ttk.Label(info_grid, textvariable=self._current_name,
                  style="Value.TLabel").grid(row=0, column=1, sticky="w", padx=8)

        ttk.Label(info_grid, text="状态:").grid(row=1, column=0, sticky="w", pady=3)
        ttk.Label(info_grid, textvariable=self._status_var,
                  style="Value.TLabel").grid(row=1, column=1, sticky="w", padx=8)

        # ── Controls ──
        sec_ctrl = ttk.LabelFrame(body, text="  🎛  控制  ")
        sec_ctrl.pack(fill=tk.X, padx=10, pady=6)

        main_row = ttk.Frame(sec_ctrl)
        main_row.pack(fill=tk.X, padx=8, pady=8)

        self._btn_start = ttk.Button(main_row, text="⏺  开始录制",
                                     style="Danger.TButton",
                                     command=self.recording_start)
        self._btn_start.pack(side=tk.LEFT, padx=(0, 6))

        self._btn_stop = ttk.Button(main_row, text="⏹  停止录制",
                                    style="Muted.TButton",
                                    command=self.recording_stop,
                                    state=tk.DISABLED)
        self._btn_stop.pack(side=tk.LEFT, padx=6)

        self._post_stop_frame = ttk.Frame(sec_ctrl)
        ttk.Label(self._post_stop_frame,
                  text="录制已停止，请选择:").pack(side=tk.LEFT, padx=(8, 12))
        self._btn_save = ttk.Button(self._post_stop_frame, text="💾  保存",
                                    style="Success.TButton",
                                    command=self.recording_save)
        self._btn_save.pack(side=tk.LEFT, padx=(0, 6))
        self._btn_delete = ttk.Button(self._post_stop_frame, text="🗑  删除",
                                      style="Warn.TButton",
                                      command=self.recording_delete)
        self._btn_delete.pack(side=tk.LEFT, padx=6)

        # ── Keyboard shortcuts reference ──
        sec_keys = ttk.LabelFrame(body, text="  ⌨  键盘快捷键  ")
        sec_keys.pack(fill=tk.X, padx=10, pady=6)

        keys_body = ttk.Frame(sec_keys)
        keys_body.pack(fill=tk.X, padx=8, pady=8)
        keys_body.columnconfigure(1, weight=1)

        for i, (key, action) in enumerate([
            ("[R]",     "开始录制"),
            ("[S]",     "停止录制"),
            ("[Enter]", "保存"),
            ("[D]",     "删除"),
        ]):
            ttk.Label(keys_body, text=key, style="Value.TLabel").grid(
                row=i, column=0, sticky="w", pady=2)
            ttk.Label(keys_body, text=action).grid(
                row=i, column=1, sticky="w", padx=8, pady=2)

        # ── EE position ──
        sec_pos = ttk.LabelFrame(body, text="  📡  末端位置实时  ")
        sec_pos.pack(fill=tk.X, padx=10, pady=6)

        pos_grid = ttk.Frame(sec_pos)
        pos_grid.pack(fill=tk.X, padx=8, pady=8)
        pos_grid.columnconfigure(1, weight=1)

        ttk.Label(pos_grid, text="左手 pos (x,y,z):").grid(
            row=0, column=0, sticky="w", pady=2)
        ttk.Label(pos_grid, textvariable=self._left_pos_var,
                  style="Value.TLabel").grid(row=0, column=1, sticky="w", padx=8)

        ttk.Label(pos_grid, text="左手 quat (x,y,z,w):").grid(
            row=1, column=0, sticky="w", pady=2)
        ttk.Label(pos_grid, textvariable=self._left_quat_var,
                  style="Value.TLabel").grid(row=1, column=1, sticky="w", padx=8)

        ttk.Separator(pos_grid, orient="horizontal").grid(
            row=2, column=0, columnspan=2, sticky="ew", pady=4)

        ttk.Label(pos_grid, text="右手 pos (x,y,z):").grid(
            row=3, column=0, sticky="w", pady=2)
        ttk.Label(pos_grid, textvariable=self._right_pos_var,
                  style="Value.TLabel").grid(row=3, column=1, sticky="w", padx=8)

        ttk.Label(pos_grid, text="右手 quat (x,y,z,w):").grid(
            row=4, column=0, sticky="w", pady=2)
        ttk.Label(pos_grid, textvariable=self._right_quat_var,
                  style="Value.TLabel").grid(row=4, column=1, sticky="w", padx=8)

    # ── Recording API ─────────────────────────────────────────────────────────

    def _next_name(self) -> str:
        out   = self.collector.output_dir
        max_n = 0
        if out.exists():
            for p in out.iterdir():
                name = p.name
                if name.startswith("recording") and len(name) == 12:
                    try:
                        max_n = max(max_n, int(name[9:]))
                    except ValueError:
                        pass
        return f"recording{max_n + 1:03d}"

    def recording_start(self) -> None:
        if self._is_recording or self._is_stopped:
            return
        name = self._next_name()
        self._current_name.set(name)
        self._is_recording = True
        self._status_var.set("🔴  录制中…")
        self._btn_start.config(state=tk.DISABLED)
        self._btn_stop.config(state=tk.NORMAL)
        self._post_stop_frame.pack_forget()
        self.collector.start_recording(episode_name=name)
        self._show_status(f"开始录制: {name}")

    def recording_stop(self) -> None:
        if not self._is_recording:
            return
        self._is_recording = False
        self._is_stopped   = True
        self._status_var.set("⏸  正在停止…")
        self._btn_stop.config(state=tk.DISABLED)
        self._show_status(f"正在停止录制: {self._current_name.get()}…")
        threading.Thread(target=self._flush_and_prompt, daemon=True).start()

    def _flush_and_prompt(self) -> None:
        self.collector.stop_recording()
        self.root.after(0, self._show_post_stop)

    def _show_post_stop(self) -> None:
        self._status_var.set("⏸  已停止 — 请保存或删除")
        self._post_stop_frame.pack(fill=tk.X, padx=8, pady=(0, 8))
        self._show_status(f"录制已停止: {self._current_name.get()}，请保存或删除")

    def recording_save(self) -> None:
        if not self._is_stopped:
            return
        name = self._current_name.get()
        self._is_stopped = False
        self._post_stop_frame.pack_forget()
        self._status_var.set(f"✅  已保存: {name}")
        self._btn_start.config(state=tk.NORMAL)
        self._show_status(f"已保存录制: {name}")

    def recording_delete(self) -> None:
        if not self._is_stopped:
            return
        name        = self._current_name.get()
        episode_dir = self.collector.output_dir / name
        if episode_dir.exists():
            shutil.rmtree(episode_dir)
        self._is_stopped = False
        self._post_stop_frame.pack_forget()
        self._current_name.set("—")
        self._status_var.set("🗑  已删除")
        self._btn_start.config(state=tk.NORMAL)
        self._show_status(f"已删除录制: {name}")

    def _show_status(self, msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        self._status_bar.set(f"[{ts}] {msg}")
        print(f"[INFO] {msg}")

    # ── Background threads ────────────────────────────────────────────────────

    def _start_camera_thread(self) -> None:
        """Display loop — renders the frames streamed by the collector at 30 Hz.

        It does not touch the SDK itself; the collector's single stream loop is
        the only place camera frames are pulled.
        """
        def _loop():
            while True:
                try:
                    for name in CAMERA_NAMES:
                        frame = self.collector.get_frame(name)
                        if frame is not None and name in self.camera_labels:
                            self._push_camera_frame(name, frame)
                    time.sleep(0.033)
                except Exception as e:
                    print(f"[camera thread] {e}")
                    time.sleep(1)

        threading.Thread(target=_loop, daemon=True).start()

    def _push_camera_frame(self, name: str, image: np.ndarray) -> None:
        try:
            if image.dtype != np.uint8:
                image = image.astype(np.uint8)

            if len(image.shape) == 3 and image.shape[2] == 3:
                pil_img = Image.fromarray(image)
            else:
                depth_2d = image[:, :, 0] if len(image.shape) == 3 else image
                if depth_2d.dtype == np.uint16:
                    valid = depth_2d[depth_2d > 0]
                    mn, mx = (np.min(valid), np.max(valid)) if valid.size else (0, 1)
                    norm = ((depth_2d.astype(np.float32) - mn) / max(mx - mn, 1) * 255).astype(np.uint8)
                else:
                    norm = depth_2d.astype(np.uint8)
                colored = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
                pil_img = Image.fromarray(cv2.cvtColor(colored, cv2.COLOR_BGR2RGB))

            tw, th  = self.camera_tile_size
            pil_img = pil_img.resize((tw, th), Image.Resampling.LANCZOS)
            photo   = ImageTk.PhotoImage(pil_img)

            def _set(lbl=self.camera_labels[name], img=photo):
                try:
                    lbl.config(image=img)
                    lbl.image = img
                except tk.TclError:
                    pass

            self.root.after(0, _set)
        except Exception as e:
            print(f"[camera display] {name}: {e}")

    def _start_ee_pos_thread(self) -> None:
        """Display loop — reads the EE poses the collector streams at 30 Hz.

        It never calls get_motion_status itself; the collector's single stream
        loop keeps latest_*_pos/quat fresh whether or not we are recording.
        """
        def _loop():
            while True:
                try:
                    lp = self.collector.latest_left_pos
                    rp = self.collector.latest_right_pos
                    lq = self.collector.latest_left_quat
                    rq = self.collector.latest_right_quat
                    if lp is None or rp is None:
                        time.sleep(0.05)
                        continue
                    lp_t = f"[{lp[0]:.3f},  {lp[1]:.3f},  {lp[2]:.3f}]"
                    rp_t = f"[{rp[0]:.3f},  {rp[1]:.3f},  {rp[2]:.3f}]"
                    lq_t = (f"[{lq[0]:.3f},  {lq[1]:.3f},  {lq[2]:.3f},  {lq[3]:.3f}]"
                            if lq is not None else "—")
                    rq_t = (f"[{rq[0]:.3f},  {rq[1]:.3f},  {rq[2]:.3f},  {rq[3]:.3f}]"
                            if rq is not None else "—")

                    self.root.after(0, lambda a=lp_t, b=rp_t, c=lq_t, d=rq_t: (
                        self._left_pos_var.set(a),
                        self._right_pos_var.set(b),
                        self._left_quat_var.set(c),
                        self._right_quat_var.set(d),
                    ))
                except Exception as e:
                    print(f"[ee pos loop] {e}")
                time.sleep(0.033)

        threading.Thread(target=_loop, daemon=True).start()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def on_closing(self) -> None:
        if getattr(self, "_closing", False):
            return
        self._closing = True

        # Destroy the window first so the UI is immediately gone.
        try:
            self.root.destroy()
        except Exception:
            pass

        # Run shutdown in a daemon thread so the process force-exits via
        # os._exit after 5 s even if an SDK call hangs.
        def _do_shutdown():
            try:
                self.collector.stop()
            except Exception as e:
                print(f"关闭数据采集器出错: {e}")

        t = threading.Thread(target=_do_shutdown, daemon=True)
        t.start()
        t.join(timeout=5.0)
        os._exit(0)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    root = tk.Tk()
    app  = DataCollectionGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    signal.signal(signal.SIGINT, lambda *_: app.on_closing())

    def _tick():
        root.after(200, _tick)
    root.after(200, _tick)

    try:
        root.mainloop()
    except KeyboardInterrupt:
        app.on_closing()


if __name__ == "__main__":
    main()
