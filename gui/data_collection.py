"""数据采集 Mixin：把独立的数据采集控制台并入 VR 遥操标签页。

录制管线全部走主类已持有的 self.env（HumanoidEnv）——它是相机与 SDK 的唯一持有者，
并自带录制 API：start_recording / stop_recording 会把 RECORD_CAMERAS 钉为常开
（即便控制台以无相机模式启动也能录），on_closing 时 env.stop() 会兜底 finalize
任何进行中的录制。本 Mixin 只负责 UI（开始/停止/保存/删除 + 末端位姿实时显示 +
键盘快捷键），不直接触碰相机或 SDK。

末端位姿用 self.robot_controller.get_motion_status() 读 arm_left/right_link7，与
HumanoidEnv._record_tick 写入录制的来源完全一致。
"""

import shutil
import threading
import time
import tkinter as tk
from tkinter import ttk


class DataCollectionMixin:
    """录制控制 + 末端位姿显示，挂在 VR 遥操标签页上。"""

    # ── 面板搭建 ────────────────────────────────────────────────────────────

    def setup_data_collection_panel(self, parent):
        """在 `parent`（VR 标签页里的一个容器）内搭建录制控制与末端位姿显示。"""
        # ---- 录制状态（GUI 侧，与 env._is_recording 区分开）----
        # 三态：idle → recording → finalizing → stopped（待保存/删除）→ idle。
        # finalizing 期间 stop_recording 正在后台写盘并撤销相机钉固，此时禁止
        # 开始新录制，否则新录制刚钉上的相机会被这次收尾撤掉（见 R_A 逻辑）。
        self._dc_is_recording = False
        self._dc_finalizing   = False
        self._dc_is_stopped   = False
        self._dc_current_name = tk.StringVar(value="—")
        self._dc_status_var   = tk.StringVar(value="Idle")

        # ---- 末端位姿显示变量 ----
        self._dc_left_pos_var   = tk.StringVar(value="—")
        self._dc_left_quat_var  = tk.StringVar(value="—")
        self._dc_right_pos_var  = tk.StringVar(value="—")
        self._dc_right_quat_var = tk.StringVar(value="—")

        # ── 大号录制指示灯：录制中=绿，未录制=红（一眼可辨）──
        # 用原生 tk.Label（而非 ttk）直接设 bg，确保各主题下背景色都生效。
        # 颜色/文案由 _dc_update_indicator() 按 self._dc_is_recording 驱动。
        self._dc_indicator = tk.Label(
            parent, text="■   NOT RECORDING",
            font=self.theme.ui(18, "bold"), bg=self.theme.DANGER, fg="white",
            height=2, anchor="center", relief="flat", bd=0,
        )
        self._dc_indicator.pack(fill=tk.X, padx=8, pady=(8, 4))

        # ── Recording status ──
        sec_info = ttk.LabelFrame(parent, text="  Recording status  ")
        sec_info.pack(fill=tk.X, padx=8, pady=(8, 6))
        info_grid = ttk.Frame(sec_info)
        info_grid.pack(fill=tk.X, padx=8, pady=8)
        info_grid.columnconfigure(1, weight=1)
        ttk.Label(info_grid, text="Episode").grid(row=0, column=0, sticky="w", pady=3)
        ttk.Label(info_grid, textvariable=self._dc_current_name,
                  style="Value.TLabel").grid(row=0, column=1, sticky="w", padx=8)
        ttk.Label(info_grid, text="State").grid(row=1, column=0, sticky="w", pady=3)
        ttk.Label(info_grid, textvariable=self._dc_status_var,
                  style="Value.TLabel").grid(row=1, column=1, sticky="w", padx=8)

        # ── Controls ──
        sec_ctrl = ttk.LabelFrame(parent, text="  Data collection  ")
        sec_ctrl.pack(fill=tk.X, padx=8, pady=6)
        main_row = ttk.Frame(sec_ctrl)
        main_row.pack(fill=tk.X, padx=8, pady=8)
        self._dc_btn_start = ttk.Button(main_row, text="⏺  Start recording",
                                        style="Danger.TButton",
                                        command=self.recording_start)
        self.tip(self._dc_btn_start,
                 "Begin recording a new episode (joints, end-effector poses and the live cameras). "
                 "Shortcut: R  ·  VR: R_A.")
        self._dc_btn_start.pack(side=tk.LEFT, padx=(0, 6))
        self._dc_btn_stop = ttk.Button(main_row, text="⏹  Stop recording",
                                       style="Muted.TButton",
                                       command=self.recording_stop,
                                       state=tk.DISABLED)
        self.tip(self._dc_btn_stop,
                 "Stop recording. You'll then choose Save (keep) or Delete (discard). Shortcut: S.")
        self._dc_btn_stop.pack(side=tk.LEFT, padx=6)

        self._dc_post_stop_frame = ttk.Frame(sec_ctrl)
        ttk.Label(self._dc_post_stop_frame,
                  text="Recording stopped — choose:").pack(side=tk.LEFT, padx=(8, 12))
        self._dc_btn_save = ttk.Button(self._dc_post_stop_frame, text="Save",
                                       style="Success.TButton",
                                       command=self.recording_save)
        self.tip(self._dc_btn_save, "Keep this episode and write it to disk. Shortcut: Enter.")
        self._dc_btn_save.pack(side=tk.LEFT, padx=(0, 6))
        self._dc_btn_delete = ttk.Button(self._dc_post_stop_frame, text="Delete",
                                         style="Warn.TButton",
                                         command=self.recording_delete)
        self.tip(self._dc_btn_delete, "Discard this episode without saving. Shortcut: D.")
        self._dc_btn_delete.pack(side=tk.LEFT, padx=6)

        # ── Shortcuts (keyboard + VR controller) ──
        sec_keys = ttk.LabelFrame(parent, text="  Shortcuts   ·   active when no text field is focused  ")
        sec_keys.pack(fill=tk.X, padx=8, pady=6)
        keys_body = ttk.Frame(sec_keys)
        keys_body.pack(fill=tk.X, padx=8, pady=8)
        keys_body.columnconfigure(1, weight=1)
        for i, (key, act) in enumerate([
            ("[R]", "Start recording"), ("[S]", "Stop recording"),
            ("[Enter]", "Save"), ("[D]", "Delete"),
            ("[VR R_A]", "Record / stop / save & start next"),
        ]):
            ttk.Label(keys_body, text=key, style="Value.TLabel").grid(
                row=i, column=0, sticky="w", pady=2)
            ttk.Label(keys_body, text=act).grid(
                row=i, column=1, sticky="w", padx=8, pady=2)

        # ── Live end-effector pose ──
        sec_pos = ttk.LabelFrame(parent, text="  End-effector pose (live)  ")
        sec_pos.pack(fill=tk.X, padx=8, pady=6)
        pos_grid = ttk.Frame(sec_pos)
        pos_grid.pack(fill=tk.X, padx=8, pady=8)
        pos_grid.columnconfigure(1, weight=1)
        ttk.Label(pos_grid, text="Left pos (x,y,z)").grid(row=0, column=0, sticky="w", pady=2)
        ttk.Label(pos_grid, textvariable=self._dc_left_pos_var,
                  style="Value.TLabel").grid(row=0, column=1, sticky="w", padx=8)
        ttk.Label(pos_grid, text="Left quat (x,y,z,w)").grid(row=1, column=0, sticky="w", pady=2)
        ttk.Label(pos_grid, textvariable=self._dc_left_quat_var,
                  style="Value.TLabel").grid(row=1, column=1, sticky="w", padx=8)
        ttk.Separator(pos_grid, orient="horizontal").grid(
            row=2, column=0, columnspan=2, sticky="ew", pady=4)
        ttk.Label(pos_grid, text="Right pos (x,y,z)").grid(row=3, column=0, sticky="w", pady=2)
        ttk.Label(pos_grid, textvariable=self._dc_right_pos_var,
                  style="Value.TLabel").grid(row=3, column=1, sticky="w", padx=8)
        ttk.Label(pos_grid, text="Right quat (x,y,z,w)").grid(row=4, column=0, sticky="w", pady=2)
        ttk.Label(pos_grid, textvariable=self._dc_right_quat_var,
                  style="Value.TLabel").grid(row=4, column=1, sticky="w", padx=8)

        self._dc_update_indicator()
        self._dc_bind_shortcuts()
        self._dc_start_ee_pos_thread()
        self._dc_start_vr_button_poll()

    def _dc_update_indicator(self):
        """Repaint the big recording box: solid GREEN while recording, solid RED otherwise."""
        ind = getattr(self, "_dc_indicator", None)
        if ind is None:
            return
        if self._dc_is_recording:
            ind.config(text="●   RECORDING", bg=self.theme.SUCCESS, fg="white")
        else:
            ind.config(text="■   NOT RECORDING", bg=self.theme.DANGER, fg="white")

    # ── 键盘快捷键（焦点守卫，避免在 VR 参数输入框里误触发）──────────────────

    def _dc_bind_shortcuts(self):
        def guarded(fn):
            def _h(_evt=None):
                # 焦点在输入框时（如 VR 灵敏度参数 Entry），让按键归输入框处理。
                w = self.root.focus_get()
                if isinstance(w, (ttk.Entry, tk.Entry)):
                    return
                fn()
            return _h

        self.root.bind("<r>",      guarded(self.recording_start))
        self.root.bind("<R>",      guarded(self.recording_start))
        self.root.bind("<s>",      guarded(self.recording_stop))
        self.root.bind("<S>",      guarded(self.recording_stop))
        self.root.bind("<Return>", guarded(self.recording_save))
        self.root.bind("<d>",      guarded(self.recording_delete))
        self.root.bind("<D>",      guarded(self.recording_delete))

    # ── 录制 API（驱动 self.env）─────────────────────────────────────────────

    def _dc_next_name(self) -> str:
        """下一个 recordingNNN 名（与 build_dataset.py 的 `recording*` 通配一致）。"""
        out   = self.env.output_dir
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

    def recording_start(self):
        if self._dc_is_recording or self._dc_finalizing or self._dc_is_stopped:
            return
        name = self._dc_next_name()
        self._dc_current_name.set(name)
        self._dc_is_recording = True
        self._dc_update_indicator()                    # → green "RECORDING"
        self._dc_status_var.set("● Recording…")
        self._dc_btn_start.config(state=tk.DISABLED)
        self._dc_btn_stop.config(state=tk.NORMAL)
        self._dc_post_stop_frame.pack_forget()
        self.env.start_recording(episode_name=name)
        self._dc_show_status(f"Started recording: {name}")

    def recording_stop(self):
        if not self._dc_is_recording:
            return
        self._dc_is_recording = False
        self._dc_update_indicator()                    # → 红色「未录制」
        self._dc_finalizing   = True   # finalizing: block a new recording until the write completes
        self._dc_status_var.set("⏸ Stopping…")
        self._dc_btn_stop.config(state=tk.DISABLED)
        self._dc_show_status(f"Stopping recording: {self._dc_current_name.get()}…")
        # stop_recording 会 finalize（写盘）—— 放后台线程，避免阻塞 UI。
        threading.Thread(target=self._dc_flush_and_prompt, daemon=True).start()

    def _dc_flush_and_prompt(self):
        try:
            self.env.stop_recording()
        except Exception as e:
            print(f"[data-collection] stop_recording: {e}")
        self.root.after(0, self._dc_show_post_stop)

    def _dc_show_post_stop(self):
        # 写盘完成 → 进入「待保存/删除」态。
        self._dc_finalizing = False
        self._dc_is_stopped = True
        self._dc_status_var.set("⏸ Stopped — save or delete")
        self._dc_post_stop_frame.pack(fill=tk.X, padx=8, pady=(0, 8))
        self._dc_show_status(f"Recording stopped: {self._dc_current_name.get()} — save or delete")

    def recording_save(self):
        if not self._dc_is_stopped:
            return
        name = self._dc_current_name.get()
        self._dc_is_stopped = False
        self._dc_post_stop_frame.pack_forget()
        self._dc_status_var.set(f"✓ Saved: {name}")
        self._dc_btn_start.config(state=tk.NORMAL)
        self._dc_show_status(f"Saved recording: {name}")

    def recording_delete(self):
        if not self._dc_is_stopped:
            return
        name        = self._dc_current_name.get()
        episode_dir = self.env.output_dir / name
        if episode_dir.exists():
            shutil.rmtree(episode_dir)
        self._dc_is_stopped = False
        self._dc_post_stop_frame.pack_forget()
        self._dc_current_name.set("—")
        self._dc_status_var.set("Deleted")
        self._dc_btn_start.config(state=tk.NORMAL)
        self._dc_show_status(f"Deleted recording: {name}")

    def _dc_show_status(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        # 复用主窗口底部状态栏。
        if getattr(self, "status_text", None) is not None:
            try:
                self.status_text.set(f"[{ts}] {msg}")
            except Exception:
                pass
        print(f"[INFO] {msg}")

    # ── VR 手柄 R_A 录制控制（单键循环）─────────────────────────────────────────

    # R_A（右手 A 键）是空闲的：VR 回调只消费 L_grab/R_grab（夹爪）与 L_Y/R_B
    # （臂部移动闸门）。详见 pico_vr 协议 triggers 索引表。
    _DC_VR_RECORD_BTN = "R_A"
    _DC_VR_STALE_S    = 0.2   # VR 超过此秒数无更新视为断连 → 按键当作松开

    def _dc_start_vr_button_poll(self):
        """在 Tk 主循环里以 ~20Hz 边沿检测 R_A，保证录制操作都在主线程跑（Tk 安全）。"""
        self._dc_vr_prev_down = False
        self._dc_vr_poll()

    def _dc_vr_button_down(self, label: str) -> bool:
        """label 当前是否按下。VR 断连（last_joint_update_timestamp 过期）时一律视为松开，
        否则一次冻结的断连会把键卡成「一直按下」。vr_buttons_pressed 由 VR 回调每拍
        以新 set 整体替换，读引用是原子的，无需加锁。"""
        last = getattr(self, "last_joint_update_timestamp", 0.0)
        if time.time() - last > self._DC_VR_STALE_S:
            return False
        return label in getattr(self, "vr_buttons_pressed", ())

    def _dc_vr_poll(self):
        try:
            down = self._dc_vr_button_down(self._DC_VR_RECORD_BTN)
            if down and not self._dc_vr_prev_down:      # 上升沿 = 按下一次
                self._dc_on_vr_record_button()
            self._dc_vr_prev_down = down
        except Exception as e:
            print(f"[data-collection vr-btn] {e}")
        if not getattr(self, "_is_closing", False):
            self.root.after(50, self._dc_vr_poll)

    def _dc_on_vr_record_button(self):
        """R_A 单键循环：录制中→停止；待保存→保存并立即开始下一段；空闲→开始。

        收尾写盘（finalizing）期间忽略，避免新录制的相机钉固被这次收尾撤掉，
        也与界面一致——保存/删除按钮在写盘完成后才出现，此时再按才会保存并续录。
        删除仍只走界面按钮，R_A 不删除。
        """
        if self._dc_is_recording:
            self.recording_stop()
        elif self._dc_is_stopped:
            self.recording_save()    # 保存上一段（清掉 stopped 态）
            self.recording_start()   # 立即开始下一段
        elif not self._dc_finalizing:
            self.recording_start()
        # finalizing：忽略，等写盘完成

    # ── 末端位姿显示线程 ──────────────────────────────────────────────────────

    def _dc_start_ee_pos_thread(self):
        """~10Hz 读 get_motion_status 的两臂 link7 位姿，刷到界面。只读，独立于录制。"""
        def _extract(frame):
            pos  = frame['position']
            quat = frame['orientation']['quaternion']
            return ((pos['x'], pos['y'], pos['z']),
                    (quat['x'], quat['y'], quat['z'], quat['w']))

        def _loop():
            while not getattr(self, "_is_closing", False):
                try:
                    status = self.robot_controller.get_motion_status()
                    frames = (status or {}).get('frames')
                    if not frames:
                        time.sleep(0.1)
                        continue
                    lp, lq = _extract(frames['arm_left_link7'])
                    rp, rq = _extract(frames['arm_right_link7'])
                    lp_t = f"[{lp[0]:.3f},  {lp[1]:.3f},  {lp[2]:.3f}]"
                    rp_t = f"[{rp[0]:.3f},  {rp[1]:.3f},  {rp[2]:.3f}]"
                    lq_t = f"[{lq[0]:.3f},  {lq[1]:.3f},  {lq[2]:.3f},  {lq[3]:.3f}]"
                    rq_t = f"[{rq[0]:.3f},  {rq[1]:.3f},  {rq[2]:.3f},  {rq[3]:.3f}]"
                    self.root.after(0, lambda a=lp_t, b=rp_t, c=lq_t, d=rq_t: (
                        self._dc_left_pos_var.set(a),
                        self._dc_right_pos_var.set(b),
                        self._dc_left_quat_var.set(c),
                        self._dc_right_quat_var.set(d),
                    ))
                except Exception as e:
                    print(f"[data-collection ee pos] {e}")
                time.sleep(0.1)

        threading.Thread(target=_loop, daemon=True).start()
