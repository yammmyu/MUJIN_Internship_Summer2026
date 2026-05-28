import shutil
import threading
import tkinter as tk
from tkinter import ttk


class DataCollectionMixin:
    """数据采集面板：录制启停、自动命名、保存/删除、键盘与 VR 按键绑定。

    依赖主类在 __init__ 中提供：
      self.data_collector  — RobotDataCollector 实例
      self.root            — Tk 根窗口
      self.show_status()   — 状态栏消息方法（来自 StatusMixin）
    """

    # ── 命名辅助 ────────────────────────────────────────────────────────

    def _next_recording_name(self) -> str:
        """扫描录制目录，返回下一个未使用的名称（recording001、002…）。"""
        output_dir = self.data_collector.output_dir
        max_num = 0
        if output_dir.exists():
            for p in output_dir.iterdir():
                name = p.name
                if name.startswith("recording") and len(name) == 12:
                    try:
                        max_num = max(max_num, int(name[9:]))
                    except ValueError:
                        pass
        return f"recording{max_num + 1:03d}"

    # ── 面板构建 ─────────────────────────────────────────────────────────

    def setup_data_collection_panel(self, parent):
        """在 parent Notebook 中追加"数据采集"标签页。"""
        tab = ttk.Frame(parent)
        parent.add(tab, text="⏺  数据采集")

        canvas = tk.Canvas(tab, highlightthickness=0, bg="#f5f6f8")
        sb = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        body = ttk.Frame(canvas)
        body.bind("<Configure>",
                  lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=body, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        # Internal state (all UI-only; no hardware state)
        self._dc_is_recording = False
        self._dc_is_stopped = False
        self._dc_current_name = tk.StringVar(value="—")
        self._dc_status_var = tk.StringVar(value="空闲")

        # ── 状态卡片 ──────────────────────────────────────────────────
        sec_info = ttk.LabelFrame(body, text="  ⏺  录制状态  ")
        sec_info.pack(fill=tk.X, padx=10, pady=(10, 6))

        info_grid = ttk.Frame(sec_info)
        info_grid.pack(fill=tk.X, padx=8, pady=8)
        info_grid.columnconfigure(1, weight=1)

        ttk.Label(info_grid, text="当前录制:").grid(
            row=0, column=0, sticky="w", pady=3)
        ttk.Label(info_grid, textvariable=self._dc_current_name,
                  style="Value.TLabel").grid(
            row=0, column=1, sticky="w", padx=8)

        ttk.Label(info_grid, text="状态:").grid(
            row=1, column=0, sticky="w", pady=3)
        ttk.Label(info_grid, textvariable=self._dc_status_var,
                  style="Value.TLabel").grid(
            row=1, column=1, sticky="w", padx=8)

        # ── 主控制按钮 ─────────────────────────────────────────────────
        sec_ctrl = ttk.LabelFrame(body, text="  🎛  控制  ")
        sec_ctrl.pack(fill=tk.X, padx=10, pady=6)

        main_row = ttk.Frame(sec_ctrl)
        main_row.pack(fill=tk.X, padx=8, pady=8)

        self._dc_btn_start = ttk.Button(
            main_row, text="⏺  开始录制",
            style="Danger.TButton",
            command=self.recording_start)
        self._dc_btn_start.pack(side=tk.LEFT, padx=(0, 6))

        self._dc_btn_stop = ttk.Button(
            main_row, text="⏹  停止录制",
            style="Muted.TButton",
            command=self.recording_stop,
            state=tk.DISABLED)
        self._dc_btn_stop.pack(side=tk.LEFT, padx=6)

        # Save / Delete row — only shown after a recording is stopped
        self._dc_post_stop_frame = ttk.Frame(sec_ctrl)
        # (not packed yet — hidden by default)

        ttk.Label(self._dc_post_stop_frame,
                  text="录制已停止，请选择:").pack(side=tk.LEFT, padx=(8, 12))

        self._dc_btn_save = ttk.Button(
            self._dc_post_stop_frame, text="💾  保存",
            style="Success.TButton",
            command=self.recording_save)
        self._dc_btn_save.pack(side=tk.LEFT, padx=(0, 6))

        self._dc_btn_delete = ttk.Button(
            self._dc_post_stop_frame, text="🗑  删除",
            style="Warn.TButton",
            command=self.recording_delete)
        self._dc_btn_delete.pack(side=tk.LEFT, padx=6)

        # ── 快捷键说明 ─────────────────────────────────────────────────
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

        # ── 键盘绑定（挂在根窗口，全局生效） ──────────────────────────
        self.root.bind("<r>",      lambda e: self.recording_start())
        self.root.bind("<R>",      lambda e: self.recording_start())
        self.root.bind("<s>",      lambda e: self.recording_stop())
        self.root.bind("<S>",      lambda e: self.recording_stop())
        self.root.bind("<Return>", lambda e: self.recording_save())
        self.root.bind("<d>",      lambda e: self.recording_delete())
        self.root.bind("<D>",      lambda e: self.recording_delete())

    # ── 录制 API ─────────────────────────────────────────────────────────

    def recording_start(self):
        """开始一段新录制（空闲状态下有效）。"""
        if not hasattr(self, '_dc_is_recording'):
            return
        if self._dc_is_recording or self._dc_is_stopped:
            return

        name = self._next_recording_name()
        self._dc_current_name.set(name)
        self._dc_is_recording = True
        self._dc_status_var.set("🔴  录制中…")
        self._dc_btn_start.config(state=tk.DISABLED)
        self._dc_btn_stop.config(state=tk.NORMAL)
        self._dc_post_stop_frame.pack_forget()

        # start() is non-blocking — it spawns its own threads
        self.data_collector.start(episode_name=name)
        self.show_status(f"开始录制: {name}", "info")

    def recording_stop(self):
        """停止当前录制并等待数据落盘（非阻塞，在后台线程完成）。"""
        if not hasattr(self, '_dc_is_recording'):
            return
        if not self._dc_is_recording:
            return

        self._dc_is_recording = False
        self._dc_is_stopped = True
        self._dc_status_var.set("⏸  正在停止…")
        self._dc_btn_stop.config(state=tk.DISABLED)
        self.show_status(f"正在停止录制: {self._dc_current_name.get()}…", "info")

        # stop() joins the record thread; run in background to keep UI responsive
        threading.Thread(target=self._dc_flush_and_prompt, daemon=True).start()

    def _dc_flush_and_prompt(self):
        """后台：等待数据落盘，然后在主线程显示保存/删除按钮。"""
        self.data_collector.stop()
        self.root.after(0, self._dc_show_post_stop)

    def _dc_show_post_stop(self):
        self._dc_status_var.set("⏸  已停止 — 请保存或删除")
        self._dc_post_stop_frame.pack(fill=tk.X, padx=8, pady=(0, 8))
        self.show_status(
            f"录制已停止: {self._dc_current_name.get()}，请保存或删除", "info")

    def recording_save(self):
        """确认保存当前录制，计数器向前推进。"""
        if not hasattr(self, '_dc_is_stopped') or not self._dc_is_stopped:
            return
        name = self._dc_current_name.get()
        self._dc_is_stopped = False
        self._dc_post_stop_frame.pack_forget()
        self._dc_status_var.set(f"✅  已保存: {name}")
        self._dc_btn_start.config(state=tk.NORMAL)
        self.show_status(f"已保存录制: {name}", "info")

    def recording_delete(self):
        """删除当前录制的磁盘数据，计数器不推进（可重用同编号）。"""
        if not hasattr(self, '_dc_is_stopped') or not self._dc_is_stopped:
            return
        name = self._dc_current_name.get()
        episode_dir = self.data_collector.output_dir / name
        if episode_dir.exists():
            shutil.rmtree(episode_dir)
        self._dc_is_stopped = False
        self._dc_post_stop_frame.pack_forget()
        self._dc_current_name.set("—")
        self._dc_status_var.set("🗑  已删除")
        self._dc_btn_start.config(state=tk.NORMAL)
        self.show_status(f"已删除录制: {name}", "info")

