"""推理控制面板：左夹爪、策略推理（手动单步 / 自动运行）、仿真预览与真机释放、子步监视。

只保留推理相关结构与左夹爪控制，原抓取/坐标/VR/手动调试已移除。
"""
import json
import pathlib
import threading
import time
import tkinter as tk
from tkinter import ttk

import numpy as np

# Persisted tuning values live next to the app (humanoid/tuning_config.json). Loaded on startup and
# rewritten whenever the operator changes a knob, so tuning survives a UI restart.
TUNING_CONFIG_PATH = pathlib.Path(__file__).resolve().parent.parent / "tuning_config.json"


def _grip_label(grip):
    """Binary {0,1} gripper command -> readable label for the substep monitor. Accepts a scalar
    (single gripper) or a [gl, gr] pair (dual-arm). ● = closed, ○ = open."""
    if isinstance(grip, (list, tuple, np.ndarray)):
        return f"L{'●' if float(grip[0]) >= 0.5 else '○'} R{'●' if float(grip[1]) >= 0.5 else '○'}"
    return "● closed" if float(grip) >= 0.5 else "○ open"


class InferenceMixin:
    """策略推理 + 左夹爪 + 仿真验证/真机释放 + 子步监视。"""

    @property
    def left_arm_joint_values(self):
        return self.robot.arm_joint_states()[0][:7]

    @property
    def both_arm_joint_values(self):
        """Live 14 arm joints [left7, right7] for the dual-arm substep monitor."""
        return self.robot.arm_joint_states()[0][:14]

    # ------------------------------------------------------------------ #
    #  左夹爪                                                              #
    # ------------------------------------------------------------------ #
    def move_gripper(self, side, position):
        """控制夹爪（左 / 右 各自独立；side='left'|'right'）。未指定的一侧保持当前位置。"""
        try:
            pos = float(np.clip(position, 0.0, 1.0))
            if side == "left":
                self.left_gripper_pos = pos
            else:
                self.right_gripper_pos = pos
            self.robot.move_gripper([self.left_gripper_pos, self.right_gripper_pos])
            self.status_text.set(f"{'Left' if side == 'left' else 'Right'} gripper → "
                                 f"{'closed' if pos >= 0.5 else 'open'}")
        except Exception as e:
            self.status_text.set(f"Gripper command failed: {e}")

    # ------------------------------------------------------------------ #
    #  手动单步推理                                                        #
    # ------------------------------------------------------------------ #
    def _run_inference_once(self):
        """推理一次, off the Tk thread (the server round-trip can block for seconds). On
        completion a fresh chunk exists, so re-enable the step-through buttons."""
        def worker():
            try:
                ok = self.inference.inference_once()
                remaining = self.inference.steps_remaining()
                msg = (f"✓ Inference complete — {remaining} steps" if ok else "✗ Inference failed")
            except Exception as e:
                msg = f"✗ Inference error: {e}"
            def done():
                self.status_text.set(msg)
                self._refresh_validation_buttons()
                self._set_manual_busy(False)       # idle again -> tuning unlocked (ALWAYS runs)
            self.root.after(0, done)
        self.status_text.set("Running inference…")
        self._set_manual_busy(True)                # lock tuning while inferencing
        threading.Thread(target=worker, daemon=True).start()

    def _run_validation(self, once=False):
        """Step the last prediction through the sim off the Tk thread (it can take a few seconds
        while the sim plays the trajectory) and report the result to the status bar.

        once=True validates+stages the NEXT unexecuted action row; once=False the REMAINING
        rows. Both become no-ops once the chunk is fully consumed (see _refresh_validation_buttons).
        """
        def worker():
            ok, reason = self.inference.execute_inference_result(once=once)
            remaining = self.inference.steps_remaining()
            msg = (f"✓ Sim-validated ({remaining} steps left) — ready to release" if ok
                   else f"✗ Sim validation failed: {reason}")
            def done():
                self.status_text.set(msg)
                self._refresh_validation_buttons()
                self._refresh_release_buttons()    # a successful validate stages new substeps
                self._set_manual_busy(False)       # idle again -> tuning unlocked
            self.root.after(0, done)
        self.status_text.set("Validating in sim…")
        self._set_manual_busy(True)                # lock tuning while validating
        threading.Thread(target=worker, daemon=True).start()

    def _refresh_validation_buttons(self):
        """Enable the 单步/整条 buttons only while the current chunk has unexecuted steps; grey
        them out (and they no-op anyway) once it's consumed or there's no prediction."""
        try:
            remaining = self.inference.steps_remaining()
        except Exception:
            remaining = 0
        flag = "!disabled" if remaining > 0 else "disabled"
        for name in ("_btn_validate_step", "_btn_validate_rest"):
            btn = getattr(self, name, None)
            if btn is not None:
                btn.state([flag])

    # ------------------------------------------------------------------ #
    #  自动运行（推理 -> 仿真验证 -> 真机）                                  #
    # ------------------------------------------------------------------ #
    def _start_auto_inference(self):
        """Begin auto-inference -> robot. Validation runs through the preview sim, so launch it
        first if needed, then wait (non-blocking) until env.sim is ready before starting the loop."""
        if getattr(self.env, "sim", None) is None:
            self.launch_sim()
        self.status_text.set("Starting sim, preparing auto-run…")
        self._auto_pending = True            # lock tuning from the moment auto-run is requested
        self._refresh_tuning_state()
        self._await_sim_then_auto()

    def _await_sim_then_auto(self, tries=0):
        """Poll (via root.after, non-blocking) for the preview sim to come up, then start auto."""
        if getattr(self.env, "sim", None) is not None:
            self.inference.auto_inference()
            self._auto_pending = False       # is_auto_inference now carries the lock
            self.status_text.set("● Auto-run active  (infer → sim-validate → robot; E-stop to halt)")
            self._refresh_tuning_state()     # is_auto_inference now True -> stays locked
        elif tries < 50:                     # ~5s budget for the sim thread to build
            self.root.after(100, lambda: self._await_sim_then_auto(tries + 1))
        else:
            self._auto_pending = False
            self.status_text.set("⚠ Sim start timed out — cannot begin auto-run")
            self._refresh_tuning_state()     # auto never started -> unlock

    def _stop_auto_inference(self):
        """Stop feeding new chunks; the release loop drains whatever is queued (use 急停 to halt)."""
        self.inference.auto_inference(stop=True)
        self.status_text.set("■ Auto-run stopped (queue drains naturally; E-stop halts immediately)")
        self._refresh_tuning_state()         # idle again -> unlock tuning

    def _update_no_flip_indicator(self):
        """Repaint the no-flip barcode pill from the macro's live status (~4 Hz), and push a one-time
        status-bar message on the two key edges (first detection, and the 6 s lock):
          * grey  — off / not loaded / watching (no marker in view)
          * amber — armed but the marker detector (cv2.aruco) is unavailable, or no camera frame
          * blue  — marker DETECTED (fires immediately), counting up toward the lock (s / commit_s)
          * green (solid) — LOCKED: held commit_s; the next place is no-flip
        """
        ind = getattr(self, "_no_flip_ind", None)
        if ind is None:
            return
        th = self.theme
        # While auto-run is OFF, actively drive one scan tick so the indicator responds to a barcode
        # held under a camera even when idle. During auto the loop already scans, so skip (avoid double
        # work). Failures are surfaced (not swallowed) so a broken scan is visible while debugging.
        scan_err = None
        try:
            if not getattr(self.inference, "is_auto_inference", False):
                self.inference.no_flip_place_scan()
        except Exception as e:
            scan_err = e
        try:
            st = self.inference.no_flip_place_status()
        except Exception as e:
            st = None
            scan_err = scan_err or e

        # Classify into one discrete state, then paint the pill AND (on state CHANGES) push a one-time
        # status-bar message — so there are two distinct events: "barcode detected" the instant it is
        # first seen, and "LOCKED" when it has been held commit_s. seen_within's hold debounces flicker,
        # so a steadily-held code fires each message once, not repeatedly.
        commit_s = float(st.get("commit_s", 0.0)) if st else 0.0
        held = float(st.get("commit_progress", 0.0)) * commit_s if st else 0.0
        txt = st.get("barcode_text") if st else None
        if not st or not st.get("has_detector"):
            state, label, bg, fg = "na", "○  no-flip: n/a", th.APP_BG, th.DOT_OFF
        elif not st.get("enabled"):
            state, label, bg, fg = "off", "○  no-flip: off", th.APP_BG, th.DOT_OFF
        elif st.get("available") is False:
            state, label, bg, fg = "missing", "▲  reader missing", th.WARN_SOFT, th.WARN_DARK
        elif st.get("committed"):
            state = "locked"
            label = "●  LOCKED — next place: no-flip" + (f"  ({txt})" if txt else "")
            bg, fg = th.SUCCESS, "white"
        elif st.get("barcode_seen"):
            state = "seen"
            label = f"◉  marker detected  {held:.1f} / {commit_s:.0f}s"
            bg, fg = th.BRAND_SOFT, th.BRAND_INK
        elif st.get("frames_ok") is False:              # scanning, but no camera frame is arriving
            state, label, bg, fg = "nocam", "◌  waiting for camera…", th.WARN_SOFT, th.WARN_DARK
        else:
            state, label, bg, fg = "watching", "◌  watching for marker", th.APP_BG, th.DOT_OFF

        try:
            ind.config(text=label, bg=bg, fg=fg)
        except tk.TclError:
            return                                          # widget destroyed (window closed) -> stop

        prev = getattr(self, "_no_flip_prev_state", None)
        if state != prev:
            if state == "seen":
                self.status_text.set("◉ Marker detected — hold it in view to lock no-flip "
                                     f"({commit_s:.0f}s)…")
            elif state == "locked":
                self.status_text.set("● Marker held — LOCKED: the next item will be placed flat "
                                     "(no flip)" + (f"  [{txt}]" if txt else ""))
            self._no_flip_prev_state = state

        # Throttled (~1 Hz) UI-side diagnostic: proves the refresh loop is alive and shows what status
        # it got + what it painted. Compare with the "[no-flip scan]" backend line: if these print but
        # the pill doesn't move -> display issue; if status never flips barcode_seen -> code/detection.
        now = time.monotonic()
        if now - getattr(self, "_no_flip_diag_t", 0.0) >= 1.0:
            self._no_flip_diag_t = now
            auto = getattr(self.inference, "is_auto_inference", False)
            if scan_err is not None:
                print(f"[no-flip UI] refresh alive, auto={auto}, SCAN ERROR: {scan_err!r}")
            else:
                keys = ("available", "frames_ok", "frame_cams", "barcode_seen", "commit_progress",
                        "committed")
                brief = {k: st.get(k) for k in keys} if st else None
                print(f"[no-flip UI] refresh alive, auto={auto}, state={state!r}, status={brief}")

        self.root.after(250, self._update_no_flip_indicator)

    # ------------------------------------------------------------------ #
    #  真机释放（手动单步路径）                                             #
    # ------------------------------------------------------------------ #
    def _run_release_substeps(self, remaining=False, count=1):
        """释放 staged substeps to the real robot. remaining=True streams ALL staged substeps;
        otherwise releases the next `count` (1 or 10). Releases whatever is left if fewer remain.
        No-op (with a status note) when nothing is staged."""
        if remaining:
            n = self.env.release_remaining_substeps()
        else:
            n = self.env.release_n_substeps(count)
        staged = self.env.staged_substeps
        if n > 0:
            self.status_text.set(f"▶ Sent {n} commands to the robot ({staged} substeps still staged)")
        else:
            self.status_text.set("⚠ No staged substeps to release (validate in sim first)")
        self._refresh_release_buttons()            # staged buffer shrank (maybe now empty)

    def _refresh_release_buttons(self):
        """Enable the 释放(单步)/释放(剩余) buttons only while substeps are staged for release;
        grey them out (and they no-op anyway) once the staged buffer is empty."""
        try:
            staged = self.env.staged_substeps
        except Exception:
            staged = 0
        flag = "!disabled" if staged > 0 else "disabled"
        for name in ("_btn_release_step", "_btn_release_ten", "_btn_release_rest"):
            btn = getattr(self, name, None)
            if btn is not None:
                btn.state([flag])

    # ------------------------------------------------------------------ #
    #  平滑参数（实时可调）                                                 #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _bind_spinbox_apply(spinbox, apply_fn):
        """ttk.Spinbox 的 command 只在点箭头时触发，键入数值不会。补绑回车/失焦，让手动键入的值
        也能应用（否则输入后无反应，像“按钮/控件失灵”）。"""
        spinbox.bind("<Return>", lambda _e: apply_fn())
        spinbox.bind("<FocusOut>", lambda _e: apply_fn())

    def _apply_smoothing(self, *_, announce=True):
        """Read the smoothing widgets (radius/σ/m/buffer) and push them to the inference controller
        live. Tolerates a partially-typed Spinbox value (TclError) by ignoring that update.
        Refused while inference is running (announce=True path) — tuning is idle-only."""
        if announce and self._tuning_locked():
            return
        try:
            radius = int(self._sm_radius_var.get())
            sigma = float(self._sm_sigma_var.get())
            m = float(self._sm_m_var.get())
            buflen = int(self._sm_buflen_var.get())
        except (tk.TclError, ValueError):
            return
        radius, sigma, m = self.inference.set_smoothing(radius=radius, sigma=sigma, m=m)
        buflen = self.inference.set_buffer_len(buflen)
        self._sm_readout_var.set(f"current:  radius={radius}   σ={sigma:.2f}   m={m:.3f}   buffer={buflen}")
        if announce:
            self.status_text.set(f"Smoothing updated:  radius={radius}  σ={sigma:.2f}  m={m:.3f}  buffer={buflen}")
            self._save_tuning()

    def _apply_exec_knobs(self, *_, announce=True):
        """Read the execution widgets (speed_scale / append_ahead) and push them to the env live.
        Affects only chunks validated after the change. Refused while inference is running."""
        if announce and self._tuning_locked():
            return
        try:
            speed = float(self._ex_speed_var.get())
            ahead = int(self._ex_ahead_var.get())
        except (tk.TclError, ValueError):
            return
        speed, sub = self.env.set_speed_scale(speed)
        self.env.append_ahead_rows = max(1, ahead)
        self._ex_readout_var.set(
            f"current:  speed={speed:.2f}  (substeps/row={sub})   append_ahead={self.env.append_ahead_rows}")
        if announce:
            self.status_text.set(
                f"Execution updated:  speed={speed:.2f}  substeps/row={sub}  append_ahead={self.env.append_ahead_rows}")
            self._save_tuning()

    # ------------------------------------------------------------------ #
    #  调参锁（仅空闲时可调）+ 持久化                                        #
    # ------------------------------------------------------------------ #
    def _tuning_locked(self):
        """Tuning is allowed only when NOTHING is inferencing: no auto-inference loop (or one being
        started), and no manual inference/validation worker in flight."""
        return bool(getattr(self.inference, "is_auto_inference", False)
                    or getattr(self, "_manual_busy", False)
                    or getattr(self, "_auto_pending", False))

    def _refresh_tuning_state(self):
        """Enable the tuning widgets only while idle; grey them out + show a hint while running."""
        locked = self._tuning_locked()
        for w in getattr(self, "_tuning_widgets", []):
            try:
                w.state(["disabled"] if locked else ["!disabled"])
            except tk.TclError:
                pass
        if getattr(self, "_tuning_hint_var", None) is not None:
            self._tuning_hint_var.set("⛔ Running — stop inference/auto-run to edit tuning" if locked
                                      else "✓ Idle — tuning editable (saved on change)")

    def _set_manual_busy(self, busy):
        self._manual_busy = busy
        self._refresh_tuning_state()

    def _load_and_apply_tuning(self):
        """Load persisted tuning (if any) and apply it to the controller/env. Called once at startup
        BEFORE the panel is built, so the widgets initialize from the restored values."""
        try:
            cfg = json.loads(TUNING_CONFIG_PATH.read_text())
        except FileNotFoundError:
            return
        except Exception as e:
            print(f"[tuning] load failed ({e}); using defaults")
            return
        try:
            self.inference.set_smoothing(radius=cfg.get("te_radius"),
                                         sigma=cfg.get("te_sigma"), m=cfg.get("te_m"))
            if cfg.get("te_buffer_len") is not None:
                self.inference.set_buffer_len(cfg["te_buffer_len"])
            if cfg.get("speed_scale") is not None:
                self.env.set_speed_scale(cfg["speed_scale"])
            if cfg.get("append_ahead_rows") is not None:
                self.env.append_ahead_rows = max(1, int(cfg["append_ahead_rows"]))
            print(f"[tuning] restored from {TUNING_CONFIG_PATH}")
        except Exception as e:
            print(f"[tuning] apply failed: {e}")

    def _save_tuning(self):
        """Persist the current tuning to disk (called whenever the operator changes a knob)."""
        cfg = {
            "_note": ("Live tuning overrides for robot_control_gui. Loaded ONCE at GUI startup; the "
                      "code constants (TE_* in inference_controller.py, APPEND_AHEAD_ROWS in "
                      "humanoid_env.py, SPEED_SCALE in timing.py) are only defaults when a key is "
                      "absent here. Hand-edits take effect on the next launch. Delete this file to "
                      "reset to code defaults."),
            "te_radius": self.inference.te_radius,        # id-axis Gaussian half-width / frozen rows
            "te_sigma": self.inference.te_sigma,          # id-axis Gaussian sigma
            "te_m": self.inference.te_m,                  # cross-chunk recency decay
            "te_buffer_len": self.inference.te_buffer_len,  # # recent raw chunks averaged
            "speed_scale": self.env.speed_scale,          # fraction of demo speed
            "append_ahead_rows": self.env.append_ahead_rows,  # rows queued ahead of the clock
        }
        try:
            TUNING_CONFIG_PATH.write_text(json.dumps(cfg, indent=2))
        except Exception as e:
            print(f"[tuning] save failed: {e}")

    # ------------------------------------------------------------------ #
    #  子步监视：实时双臂 14 关节 + 待释放子步滚动表（左/右两张表）           #
    # ------------------------------------------------------------------ #
    def _update_one_arm_monitor(self, tree, sl, live14, live_grips, arm, subs):
        """Fill ONE arm's 7-joint substep table. sl = slice(0,7) left / slice(7,14) right; arm =
        0/1 selects that side's gripper. Both tables are driven by the SAME 14-joint subs, sliced
        per arm, so left and right stay row-aligned (same master-id substep on each side)."""
        # live row = that arm's 7 measured joints + its gripper; also the delta ref for substep #0.
        if live14 is not None and len(live14) >= 14:
            a7 = np.asarray(live14, dtype=np.float64)[sl]
            vals = [f"{a7[k]:+.3f}" for k in range(7)]
            gtxt = _grip_label(live_grips[arm]) if live_grips is not None else "—"
            tree.item("live", values=("LIVE", *vals, gtxt))
            prev = a7
        else:
            tree.item("live", values=("LIVE", *["—"] * 8))
            prev = None
        for i in range(self._monitor_rows):
            iid = f"subrow{i}"
            if i < len(subs):
                q, grip = subs[i]
                q = np.asarray(q, dtype=np.float64)[sl]          # this arm's 7 joints of the 14-vec
                if prev is not None:
                    d = q - prev
                    vals = [f"{q[k]:+.3f} Δ{d[k]:+.3f}" for k in range(7)]
                else:
                    vals = [f"{q[k]:+.3f}" for k in range(7)]
                g = grip[arm] if isinstance(grip, (list, tuple, np.ndarray)) else grip
                gtxt = _grip_label(g) if g is not None else "—"
                tree.item(iid, values=(i, *vals, gtxt))
                prev = q
            else:
                tree.item(iid, values=(i, *[""] * 8))

    def _update_substep_monitor(self):
        """Refresh the live joints readout + the rolling next-10 staged-substep tables for BOTH arms
        (each cell shows the joint value and its delta vs the previous row / the live pose).
        Reschedules itself ~5 Hz; the staged buffer draining in real time produces the rolling effect."""
        try:
            # --- live actual BOTH-arm joints + both grippers -> the pinned top "实时" rows ---
            try:
                live14 = list(self.both_arm_joint_values)
            except Exception:
                live14 = None
            try:
                live_grips = list(self.robot.gripper_states()[0][:2])  # [left, right] {0,1}
            except Exception:
                live_grips = None
            # --- upcoming robot commands (14-joint substeps), shared by both arm tables ---
            # Manual path stages into _staged_release (shown pre-release); the AUTO path appends
            # straight onto _robot_q and never stages, so fall back to robot_q_preview when nothing
            # is staged. Either way these are absolute (q14, [gl,gr]) targets; each table slices its side.
            try:
                subs = self.env.staged_preview(self._monitor_rows)
                source = "staged (manual validation)"
                if not subs:
                    subs = self.env.robot_q_preview(self._monitor_rows)
                    source = "robot queue (auto / released)"
            except Exception:
                subs = []
                source = "—"
            self._monitor_src_var.set(f"queue source:  {source}")
            self._update_one_arm_monitor(self._monitor_tree_l, slice(0, 7), live14, live_grips, 0, subs)
            self._update_one_arm_monitor(self._monitor_tree_r, slice(7, 14), live14, live_grips, 1, subs)
        except tk.TclError:
            self._monitor_after_id = None       # widget destroyed (window closed) -> stop
            return
        finally:
            if getattr(self, "_monitor_after_id", None) is not None:
                self._monitor_after_id = self.root.after(200, self._update_substep_monitor)

    # ------------------------------------------------------------------ #
    #  面板布局                                                            #
    # ------------------------------------------------------------------ #
    def setup_inference_panel(self, parent):
        """推理控制面板：左夹爪 / 仿真预览 / 自动运行 / 手动单步 / 真机释放 / 子步监视。"""
        # ===== Pinned E-STOP bar (never scrolls out of reach) =====
        # The single most safety-critical control lives OUTSIDE the scroll canvas, pinned to the
        # top of the column, so it is always one click away no matter how far the operator has
        # scrolled through tuning / the substep monitor. E-STOP is latched: it drops pending
        # commands and actively holds; Reset re-arms once the arm is confirmed safe.
        estop_bar = ttk.Frame(parent, style="Toolbar.TFrame")
        estop_bar.pack(side=tk.TOP, fill=tk.X, padx=0, pady=(0, 6))
        inner = ttk.Frame(estop_bar, style="Toolbar.TFrame")
        inner.pack(fill=tk.X, padx=8, pady=8)
        self.tip(
            ttk.Button(inner, text="⛔  E-STOP", style="EStop.TButton",
                       command=lambda: (self.env.lock_robot(), self._refresh_release_buttons())),
            "Immediately halt and hold both arms. Latched — drops all queued motion. "
            "Press Reset only once you've confirmed the arm is in a safe position."
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        self.tip(
            ttk.Button(inner, text="Reset", style="Muted.TButton",
                       command=lambda: self.env.reset_estop()),
            "Clear the latched E-stop and re-arm the robot. Do this only when the arm is safe."
        ).pack(side=tk.LEFT)
        ttk.Separator(parent, orient="horizontal").pack(side=tk.TOP, fill=tk.X, pady=(0, 4))

        # 用 Canvas+滚动条 包裹，内容多时可滚动。
        # width= 预留足够横向空间，使整面板（含较宽的子步监视表）在默认窗口尺寸下即可完整显示；
        # 下面的 <Configure> 绑定再把内层 frame 钉到 canvas 视口宽度，窗口变窄时内容自适应回流
        # （纵向溢出交给竖直滚动条），而不是被裁切到右边缘之外、只能靠放大整窗才能看到。
        canvas = tk.Canvas(parent, highlightthickness=0, bg=self.theme.APP_BG, width=760)
        sb = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview, style="Vertical.TScrollbar")
        body = ttk.Frame(canvas)
        body.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        body_window = canvas.create_window((0, 0), window=body, anchor="nw")
        # 内层 frame 始终与 canvas 视口同宽：不再横向裁切；内容用 fill=X / grid 权重 / 可拉伸列
        # 回流到给定宽度。
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(body_window, width=e.width))
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        # 鼠标滚轮滚动（仅当指针在面板内时生效，避免抢占其他控件）。跨平台：
        # Windows/macOS 用 <MouseWheel>(event.delta)，X11/Linux 用 <Button-4>/<Button-5>。
        def _on_wheel(event):
            delta = 1 if getattr(event, "num", None) == 5 else -1 if getattr(event, "num", None) == 4 \
                else (-1 if event.delta > 0 else 1)
            canvas.yview_scroll(delta, "units")
        def _bind_wheel(_):
            canvas.bind_all("<MouseWheel>", _on_wheel)
            canvas.bind_all("<Button-4>", _on_wheel)
            canvas.bind_all("<Button-5>", _on_wheel)
        def _unbind_wheel(_):
            canvas.unbind_all("<MouseWheel>")
            canvas.unbind_all("<Button-4>")
            canvas.unbind_all("<Button-5>")
        canvas.bind("<Enter>", _bind_wheel)
        canvas.bind("<Leave>", _unbind_wheel)

        # ===== Guided-workflow hint (orients a first-time operator to the happy path) =====
        guide = ttk.Frame(body)
        guide.pack(fill=tk.X, padx=10, pady=(10, 2))
        ttk.Label(guide, text="How to run the policy", style="CardTitle.TLabel").pack(anchor="w")
        ttk.Label(
            guide, style="Caption.TLabel", justify="left",
            text=("Typical run:  ①  Start sim preview   →   ②  Start auto-run   "
                  "(infer → sim-validate → robot).\nPrefer to go step by step? Use "
                  "Manual step-through below. Hit ⛔ E-STOP (top) to halt at any time."),
        ).pack(anchor="w", pady=(2, 0))

        # ===== Grippers (left / right, independent) =====
        sec_grip = ttk.LabelFrame(body, text="  Grippers  ")
        sec_grip.pack(fill=tk.X, padx=10, pady=(10, 6))
        for side, label in (("left", "Left"), ("right", "Right")):
            row = ttk.Frame(sec_grip)
            row.pack(fill=tk.X, padx=8, pady=(8, 4))
            ttk.Label(row, text=label, width=6, style="CardTitle.TLabel").pack(side=tk.LEFT, padx=(0, 6))
            self.tip(ttk.Button(row, text="Open", style="Success.TButton",
                     command=lambda s=side: self.move_gripper(s, 0.0)),
                     f"Fully open the {label.lower()} gripper.").pack(side=tk.LEFT, padx=(0, 6))
            self.tip(ttk.Button(row, text="Close", style="Warn.TButton",
                     command=lambda s=side: self.move_gripper(s, 1.0)),
                     f"Fully close the {label.lower()} gripper.").pack(side=tk.LEFT, padx=6)

        # ===== Sim preview (required before auto-run / sim-validate) =====
        sec_sim = ttk.LabelFrame(body, text="  Sim preview  ")
        sec_sim.pack(fill=tk.X, padx=10, pady=6)
        ttk.Label(sec_sim, text="Start the PyBullet preview before auto-run or sim validation.",
                  style="Caption.TLabel").pack(anchor="w", padx=8, pady=(2, 0))
        row = ttk.Frame(sec_sim)
        row.pack(fill=tk.X, padx=8, pady=8)
        self.tip(ttk.Button(row, text="Start sim preview", style="Primary.TButton",
                 command=lambda: self.launch_sim()),
                 "Open the PyBullet preview window and match it to the robot's current pose. "
                 "Required before auto-run or sim validation — the trajectory is checked here first."
                 ).pack(side=tk.LEFT, padx=(0, 6))

        # ===== Auto-run (infer -> sim-validate -> robot) =====
        sec_auto = ttk.LabelFrame(body, text="  Auto-run   ·   infer → sim-validate → robot  ")
        sec_auto.pack(fill=tk.X, padx=10, pady=6)
        row = ttk.Frame(sec_auto)
        row.pack(fill=tk.X, padx=8, pady=8)
        self.tip(ttk.Button(row, text="▶  Start auto-run", style="Danger.TButton",
                 command=lambda: self._start_auto_inference()),
                 "Continuously run the policy: infer → validate in sim → send to the robot, on a loop. "
                 "Starts the sim preview automatically if it isn't up yet. Halt with E-STOP (top)."
                 ).pack(side=tk.LEFT, padx=(0, 6))
        self.tip(ttk.Button(row, text="■  Stop auto-run", style="Muted.TButton",
                 command=lambda: self._stop_auto_inference()),
                 "Stop feeding new predictions. Motion already queued drains naturally — "
                 "use E-STOP instead if you need to halt immediately."
                 ).pack(side=tk.LEFT, padx=6)

        # Post-flip release macro toggle: after the policy's grab+lift+flip, run the scripted release
        # (move out, open gripper, come back). Live-toggleable even mid-run (it just gates the loop
        # hook). Disabled + greyed when the baked release path isn't loaded (macro is None).
        fp_row = ttk.Frame(sec_auto)
        fp_row.pack(fill=tk.X, padx=8, pady=(0, 8))
        self._flip_place_var = tk.BooleanVar(value=self.inference.flip_place_enabled)
        self._flip_place_cb = ttk.Checkbutton(
            fp_row, text="Run post-flip release macro", variable=self._flip_place_var,
            command=lambda: setattr(self.inference, "flip_place_enabled", self._flip_place_var.get()))
        self._flip_place_cb.pack(side=tk.LEFT)
        if getattr(self.inference, "flip_place", None) is None:
            self._flip_place_cb.state(["disabled"])
            ttk.Label(fp_row, text="(macro not loaded)", style="Caption.TLabel").pack(side=tk.LEFT, padx=6)
        self.tip(self._flip_place_cb,
                 "After the policy grabs, lifts and flips, automatically run the scripted release "
                 "(move out, open gripper, return). Safe to toggle mid-run.")

        # No-flip release macro: when the head camera sees an ArUco MARKER on the grasped package it is
        # right-side-up and must NOT be flipped, so place it as-is. Checkbox arms it; the pill to its
        # right is a LIVE indicator (repainted ~4 Hz by _update_no_flip_indicator) of marker detection.
        nfp_row = ttk.Frame(sec_auto)
        nfp_row.pack(fill=tk.X, padx=8, pady=(0, 8))
        self._no_flip_place_var = tk.BooleanVar(value=bool(self.inference.no_flip_place_enabled))
        self._no_flip_place_cb = ttk.Checkbutton(
            nfp_row, text="Place flat when a marker is seen (skip flip)",
            variable=self._no_flip_place_var,
            command=lambda: setattr(self.inference, "no_flip_place_enabled", self._no_flip_place_var.get()))
        self._no_flip_place_cb.pack(side=tk.LEFT)
        if getattr(self.inference, "no_flip_place", None) is None:
            self._no_flip_place_cb.state(["disabled"])
        self.tip(self._no_flip_place_cb,
                 "Scan the head camera continuously; once an ArUco marker is held ~6 s the next place "
                 "locks to no-flip (place as-is via the scripted release), and the lock clears after it "
                 "fires. Safe to toggle mid-run.")
        # Live marker indicator pill (green when a marker is currently seen).
        self._no_flip_ind = tk.Label(nfp_row, text="●  marker", font=self.theme.ui(9, "bold"),
                                     bg=self.theme.APP_BG, fg=self.theme.DOT_OFF, padx=8, pady=2)
        self._no_flip_ind.pack(side=tk.RIGHT)
        self._update_no_flip_indicator()          # paint once now + start the ~4 Hz refresh

        # (Emergency stop lives in the pinned bar at the top of this column — always reachable.)

        # ===== Manual step-through (infer once -> validate -> release) =====
        sec_manual = ttk.LabelFrame(body, text="  Manual step-through  ")
        sec_manual.pack(fill=tk.X, padx=10, pady=6)

        manual_row = ttk.Frame(sec_manual)
        manual_row.pack(fill=tk.X, padx=8, pady=(8, 4))
        ttk.Label(manual_row, text="① Infer", style="Section.TLabel").pack(side=tk.LEFT, padx=(0, 6))
        self.tip(ttk.Button(manual_row, text="Infer once", style="Primary.TButton",
                 command=lambda: self._run_inference_once()),
                 "Run the policy a single time to produce a fresh action chunk. "
                 "Nothing moves yet — validate it next.").pack(side=tk.LEFT, padx=4)
        # Validate-in-sim (step + self-collision + readback); stages the sim-validated
        # trajectory but does NOT touch the robot. Run off the Tk thread so the GUI doesn't freeze.
        self._btn_validate_step = ttk.Button(manual_row, text="② Validate (step)", style="Ghost.TButton",
                   command=lambda: self._run_validation(once=True))
        self.tip(self._btn_validate_step,
                 "Play the NEXT action through the sim (collision + reachability check) and stage it "
                 "for release. The real robot does not move.")
        self._btn_validate_step.pack(side=tk.LEFT, padx=4)
        self._btn_validate_rest = ttk.Button(manual_row, text="② Validate (all)", style="Ghost.TButton",
                   command=lambda: self._run_validation(once=False))
        self.tip(self._btn_validate_rest,
                 "Validate ALL remaining actions of this chunk through the sim and stage them. "
                 "The real robot does not move.")
        self._btn_validate_rest.pack(side=tk.LEFT, padx=4)
        self._refresh_validation_buttons()       # no prediction yet -> disabled until first inference

        release_row = ttk.Frame(sec_manual)
        release_row.pack(fill=tk.X, padx=8, pady=(4, 8))
        ttk.Label(release_row, text="③ Release", style="Section.TLabel").pack(side=tk.LEFT, padx=(0, 6))
        self._btn_release_step = ttk.Button(release_row, text="▶ Release (1)", style="Danger.TButton",
                   command=lambda: self._run_release_substeps(remaining=False, count=1))
        self.tip(self._btn_release_step, "Send the next single sim-validated substep to the real robot.")
        self._btn_release_step.pack(side=tk.LEFT, padx=4)
        self._btn_release_ten = ttk.Button(release_row, text="▶ Release (10)", style="Danger.TButton",
                   command=lambda: self._run_release_substeps(remaining=False, count=10))
        self.tip(self._btn_release_ten, "Send the next 10 sim-validated substeps to the real robot.")
        self._btn_release_ten.pack(side=tk.LEFT, padx=4)
        self._btn_release_rest = ttk.Button(release_row, text="▶ Release (rest)", style="Danger.TButton",
                   command=lambda: self._run_release_substeps(remaining=True))
        self.tip(self._btn_release_rest, "Stream ALL remaining staged substeps to the real robot.")
        self._btn_release_rest.pack(side=tk.LEFT, padx=4)
        self._refresh_release_buttons()          # nothing staged yet -> disabled until validation

        # 调参锁状态：仅在空闲（无手动/自动推理）时可调；下列控件按状态启用/禁用。
        self._manual_busy = False
        self._auto_pending = False
        self._tuning_widgets = []

        # ===== Smoothing (editable when idle; saved on change) =====
        sec_smooth = ttk.LabelFrame(body, text="  Smoothing   ·   editable when idle  ")
        sec_smooth.pack(fill=tk.X, padx=10, pady=6)
        grid = ttk.Frame(sec_smooth)
        grid.pack(fill=tk.X, padx=8, pady=8)
        grid.columnconfigure(1, weight=1)

        self._tuning_hint_var = tk.StringVar()
        ttk.Label(grid, textvariable=self._tuning_hint_var,
                  style="Section.TLabel").grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 4))

        # 当前值取自推理控制器/env（已应用持久化的设置），保证界面与运行时一致
        self._sm_radius_var = tk.IntVar(value=self.inference.te_radius)
        self._sm_sigma_var = tk.DoubleVar(value=round(self.inference.te_sigma, 2))
        self._sm_m_var = tk.DoubleVar(value=round(self.inference.te_m, 3))

        # TE radius: id-axis Gaussian half-width = frozen context rows. Larger = smoother, more lag.
        lbl = ttk.Label(grid, text="Smoothness (TE radius)", anchor=tk.W)
        lbl.grid(row=1, column=0, sticky="w", pady=3)
        self.tip(lbl, "How many neighbouring prediction rows are blended together. "
                      "Higher = smoother motion, but a touch more lag.")
        self._tuning_widgets.append(ttk.Spinbox(grid, from_=0, to=8, increment=1, width=6,
                    textvariable=self._sm_radius_var, command=self._apply_smoothing))
        self._bind_spinbox_apply(self._tuning_widgets[-1], self._apply_smoothing)  # 键入回车/失焦也应用
        self._tuning_widgets[-1].grid(row=1, column=2, sticky="e", padx=4)

        # TE σ: Gaussian shape within the window. Larger = stronger in-window smoothing (keep <= radius).
        lbl = ttk.Label(grid, text="Blend strength (σ)", anchor=tk.W)
        lbl.grid(row=2, column=0, sticky="w", pady=3)
        self.tip(lbl, "How strongly rows inside the smoothing window are blended. "
                      "Higher = stronger smoothing. Keep it at or below the smoothness radius.")
        self._tuning_widgets.append(ttk.Scale(grid, from_=0.3, to=4.0, orient=tk.HORIZONTAL,
                  variable=self._sm_sigma_var, command=lambda _v: self._apply_smoothing()))
        self._tuning_widgets[-1].grid(row=2, column=1, sticky="ew", padx=8)

        # TE decay m: recency decay across chunks at the same id. Larger = trust newest chunk more.
        lbl = ttk.Label(grid, text="Recency (decay m)", anchor=tk.W)
        lbl.grid(row=3, column=0, sticky="w", pady=3)
        self.tip(lbl, "How much the newest prediction outweighs older ones for the same step. "
                      "Higher = react faster to the latest prediction (less averaging).")
        self._tuning_widgets.append(ttk.Scale(grid, from_=0.0, to=0.5, orient=tk.HORIZONTAL,
                  variable=self._sm_m_var, command=lambda _v: self._apply_smoothing()))
        self._tuning_widgets[-1].grid(row=3, column=1, sticky="ew", padx=8)

        # TE buffer length: # of recent chunks averaged (overlap depth). Larger = deeper, smoother.
        lbl = ttk.Label(grid, text="Overlap depth (buffer)", anchor=tk.W)
        lbl.grid(row=4, column=0, sticky="w", pady=3)
        self.tip(lbl, "How many recent prediction chunks are averaged together. "
                      "Higher = deeper overlap and smoother motion, at a little more lag.")
        self._sm_buflen_var = tk.IntVar(value=self.inference.te_buffer_len)
        self._tuning_widgets.append(ttk.Spinbox(grid, from_=1, to=16, increment=1, width=6,
                    textvariable=self._sm_buflen_var, command=self._apply_smoothing))
        self._bind_spinbox_apply(self._tuning_widgets[-1], self._apply_smoothing)  # 键入回车/失焦也应用
        self._tuning_widgets[-1].grid(row=4, column=2, sticky="e", padx=4)

        self._sm_readout_var = tk.StringVar()
        ttk.Label(grid, textvariable=self._sm_readout_var,
                  style="Value.TLabel").grid(row=5, column=0, columnspan=3, sticky="w", pady=(6, 0))

        # ===== Execution (speed / queue-ahead; applies to later inferences) =====
        sec_exec = ttk.LabelFrame(body, text="  Execution   ·   editable when idle  ")
        sec_exec.pack(fill=tk.X, padx=10, pady=6)
        egrid = ttk.Frame(sec_exec)
        egrid.pack(fill=tk.X, padx=8, pady=8)
        egrid.columnconfigure(1, weight=1)

        # Speed: fraction of demo speed. Lower = slower, more substeps/row (smoother), longer chunk.
        lbl = ttk.Label(egrid, text="Speed (× demo)", anchor=tk.W)
        lbl.grid(row=0, column=0, sticky="w", pady=3)
        self.tip(lbl, "Playback speed as a fraction of the recorded demo. "
                      "Lower = slower and smoother (more substeps per row); 1.0 = demo speed.")
        self._ex_speed_var = tk.DoubleVar(value=round(self.env.speed_scale, 2))
        self._tuning_widgets.append(ttk.Scale(egrid, from_=0.05, to=3.0, orient=tk.HORIZONTAL,
                  variable=self._ex_speed_var, command=lambda _v: self._apply_exec_knobs()))
        self._tuning_widgets[-1].grid(row=0, column=1, sticky="ew", padx=8)

        # Queue-ahead: rows queued ahead of the master clock = commit distance. Lower = later freeze.
        lbl = ttk.Label(egrid, text="Look-ahead (rows)", anchor=tk.W)
        lbl.grid(row=1, column=0, sticky="w", pady=3)
        self.tip(lbl, "How many rows are committed ahead of real time. "
                      "Lower = commits later, so the robot reacts sooner to a stop; higher = smoother buffering.")
        self._ex_ahead_var = tk.IntVar(value=self.env.append_ahead_rows)
        self._tuning_widgets.append(ttk.Spinbox(egrid, from_=1, to=20, increment=1, width=6,
                    textvariable=self._ex_ahead_var, command=self._apply_exec_knobs))
        self._bind_spinbox_apply(self._tuning_widgets[-1], self._apply_exec_knobs)  # 键入回车/失焦也应用
        self._tuning_widgets[-1].grid(row=1, column=2, sticky="e", padx=4)

        self._ex_readout_var = tk.StringVar()
        ttk.Label(egrid, textvariable=self._ex_readout_var,
                  style="Value.TLabel").grid(row=2, column=0, columnspan=3, sticky="w", pady=(6, 0))

        self._apply_smoothing(announce=False)    # initialize the readout labels (no save/lock-check)
        self._apply_exec_knobs(announce=False)
        self._refresh_tuning_state()             # set initial enabled/disabled + hint

        # ===== Substep monitor: live dual-arm 14 joints + upcoming substeps (per-arm, with deltas) =====
        sec_monitor = ttk.LabelFrame(body, text="  Substep monitor   ·   dual-arm 14 joints (rad)  ")
        sec_monitor.pack(fill=tk.X, padx=10, pady=6)

        # 实时关节作为表内置顶高亮行（#＝实时），其下为接下来要下发到真机的子步；左右两张表按同一
        # 子步行号（master id）对齐，便于把“当前位姿 → 接下来要执行的子步”双臂一起看。Δ＝相对上一行增量。
        # 子步来源：手动验证时取待释放缓冲（_staged_release）；自动运行/已释放时取真机队列
        # （_robot_q，即 append_actions 直接喂入的实时指令）。左右两表共用同一 subs，各取自己 7 关节。
        cap_row = ttk.Frame(sec_monitor)
        cap_row.pack(fill=tk.X, padx=8, pady=(6, 4))
        ttk.Label(cap_row,
                  text="Top row = LIVE joints;  #0 = next substep to send,  Δ = change vs previous row.  "
                       "Grip: ● closed / ○ open.",
                  style="Caption.TLabel", anchor=tk.W).pack(side=tk.LEFT)
        self._monitor_src_var = tk.StringVar(value="queue source:  —")
        ttk.Label(cap_row, textvariable=self._monitor_src_var,
                  style="Value.TLabel", anchor=tk.E).pack(side=tk.RIGHT)

        self._monitor_rows = 10

        def _build_arm_table(arm_label):
            """Build one arm's 7-joint substep Treeview (live row + _monitor_rows substep rows)."""
            grp = ttk.LabelFrame(sec_monitor, text=f"  {arm_label}  ")
            grp.pack(fill=tk.X, padx=8, pady=(2, 6))
            cols = ("idx", "j1", "j2", "j3", "j4", "j5", "j6", "j7", "grip")
            t = ttk.Treeview(grp, columns=cols, show="headings", height=11, style="Monitor.Treeview")
            t.heading("idx", text="#")
            t.column("idx", width=42, anchor="center", stretch=False)
            for k, c in enumerate(cols[1:8], start=1):
                t.heading(c, text=f"J{k}")
                t.column(c, width=90, minwidth=64, anchor="center", stretch=True)
            t.heading("grip", text="grip")
            t.column("grip", width=64, anchor="center", stretch=False)
            th = self.theme
            t.tag_configure("live", background=th.BRAND_SOFT, foreground=th.BRAND_INK,
                            font=th.mono(9, "bold"))
            t.tag_configure("odd", background=th.SURFACE)
            t.tag_configure("even", background=th.SURFACE_ALT)
            t.pack(fill=tk.X, padx=6, pady=(0, 6))
            t.insert("", "end", iid="live", tags=("live",), values=("LIVE", *["—"] * 8))
            for i in range(self._monitor_rows):
                t.insert("", "end", iid=f"subrow{i}",
                         tags=("even" if i % 2 else "odd",), values=(i, *[""] * 8))
            return t

        self._monitor_tree_l = _build_arm_table("Left arm (L)")
        self._monitor_tree_r = _build_arm_table("Right arm (R)")

        # Start the periodic refresh (idempotent — only schedules once).
        if getattr(self, "_monitor_after_id", None) is None:
            self._monitor_after_id = self.root.after(200, self._update_substep_monitor)
