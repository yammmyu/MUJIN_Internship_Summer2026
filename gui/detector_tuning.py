"""Detector-tuning page: a live head-camera view with YOLO detection boxes drawn, next to the YoloGate
controls — so you can watch the no-flip detector and adjust its confidence/IoU in real time.

Boxes: GREEN = an accepted detection, labelled with its class + confidence. (Until the trained model
is added the view just shows the raw frame — see the status line.) Adjust the sliders on the right so
the target you WANT is detected and nothing else is.

The view runs only while this tab is visible (winfo_viewable), and it runs the SAME YoloGate instance
the macro uses (via inference.no_flip_detector()), so edits here apply to the live robot immediately.
"""
import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageTk


class DetectorTuningMixin:
    """Live label-detector tuning tab (view + threshold controls)."""

    # Rejected candidates smaller than this fraction of the frame are NOT drawn (declutters the sea of
    # tiny fragments from fabric texture). Accepted (green) blocks are always drawn regardless.
    _REJECT_DRAW_MIN_FRAC = 0.004

    def setup_detector_tuning_panel(self, parent):
        self._tune_view_w = 760            # rendered view width (px); height follows the frame aspect

        outer = ttk.Frame(parent)
        outer.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # ---- left: live annotated camera view (head OR right-wrist, per the source selector) ----
        left = ttk.LabelFrame(outer, text="  Live detection  ")
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))
        self._tune_view = tk.Label(left, bg="#111418")
        self._tune_view.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self._tune_stats_var = tk.StringVar(value="starting…")
        ttk.Label(left, textvariable=self._tune_stats_var, style="Value.TLabel").pack(
            anchor="w", padx=8, pady=(0, 6))
        ttk.Label(left, style="Caption.TLabel", justify="left",
                  text="GREEN = accepted detection (class + confidence).   RED = below the confidence "
                       "threshold (near-miss). Pick which model/camera to watch on the right.").pack(
            anchor="w", padx=8, pady=(0, 6))

        # ---- right: threshold controls (same live tunables as the macro) ----
        right = ttk.LabelFrame(outer, text="  Thresholds · live  ")
        right.pack(side=tk.RIGHT, fill=tk.Y, expand=False, padx=(8, 0))

        # Live source: which model + camera the annotated view runs. "head" = the no-flip label YoloGate
        # on the head camera (with its full threshold set below); "gripper" = the grasp-recovery
        # right-wrist YOLO (open / closed-gripped / closed-empty), tuned by its own confidence spinbox.
        src = ttk.Frame(right)
        src.pack(fill=tk.X, padx=10, pady=(10, 0))
        ttk.Label(src, text="Live source", style="Caption.TLabel").pack(anchor="w")
        self._tune_source_var = tk.StringVar(value="head")
        for val, text in (("head", "Head · no-flip label"), ("gripper", "Right wrist · gripper")):
            ttk.Radiobutton(src, text=text, value=val, variable=self._tune_source_var).pack(anchor="w")

        grid = ttk.Frame(right)
        grid.pack(fill=tk.X, padx=10, pady=10)
        grid.columnconfigure(1, weight=1)

        ttk.Label(grid, style="Caption.TLabel", justify="left", wraplength=240,
                  text="Turn a threshold UP to reject more (stricter); DOWN to accept more (looser). "
                       "Changes apply on the next frame.").grid(row=0, column=0, columnspan=3,
                                                                sticky="w", pady=(0, 8))
        cur = (self.inference.no_flip_detector_params() or {}) if hasattr(self, "inference") else {}
        no_macro = getattr(getattr(self, "inference", None), "no_flip_place", None) is None
        self._tune_param_vars = {}
        for i, (attr, text, frm, to, step, is_int, tip) in enumerate(self._LABEL_PARAM_SPEC, start=1):
            lbl = ttk.Label(grid, text=text, anchor="w")
            lbl.grid(row=i, column=0, sticky="w", pady=3)
            self.tip(lbl, tip)
            var = (tk.IntVar if is_int else tk.DoubleVar)(value=cur.get(attr, 2 if is_int else 0.1))
            self._tune_param_vars[attr] = var
            sb = ttk.Spinbox(grid, from_=frm, to=to, increment=step, width=8, textvariable=var,
                             command=lambda a=attr, v=var, ii=is_int: self._apply_label_param(a, v, ii))
            self._bind_spinbox_apply(
                sb, lambda a=attr, v=var, ii=is_int: self._apply_label_param(a, v, ii))
            sb.grid(row=i, column=2, sticky="e", padx=4)
            if no_macro:
                sb.state(["disabled"])
        if no_macro:
            ttk.Label(grid, text="(no-flip macro not loaded)", style="Caption.TLabel").grid(
                row=len(self._LABEL_PARAM_SPEC) + 1, column=0, columnspan=3, sticky="w", pady=(6, 0))

        # Gripper (right-wrist) detector: a single confidence threshold for the grasp-recovery YOLO.
        # Applies to the live recovery check too (same detector instance). Disabled if it isn't loaded.
        ttk.Separator(right, orient="horizontal").pack(fill=tk.X, padx=10, pady=(2, 6))
        gframe = ttk.Frame(right)
        gframe.pack(fill=tk.X, padx=10, pady=(0, 10))
        gp = (self.inference.gripper_detector_params() if hasattr(self, "inference") else None) or None
        glbl = ttk.Label(gframe, text="Gripper conf", anchor="w")
        glbl.grid(row=0, column=0, sticky="w")
        self.tip(glbl, "Min confidence for the right-wrist grasp-recovery detector (open / "
                       "closed-gripped / closed-empty). Higher = stricter. Applies to recovery live.")
        gframe.columnconfigure(1, weight=1)
        self._tune_grip_conf = tk.DoubleVar(value=(gp or {}).get("conf", 0.40))
        gsb = ttk.Spinbox(gframe, from_=0.05, to=0.95, increment=0.05, width=8,
                          textvariable=self._tune_grip_conf, command=self._apply_gripper_conf)
        self._bind_spinbox_apply(gsb, self._apply_gripper_conf)
        gsb.grid(row=0, column=2, sticky="e", padx=4)
        if gp is None:
            gsb.state(["disabled"])
            ttk.Label(gframe, text="(gripper detector not loaded)", style="Caption.TLabel").grid(
                row=1, column=0, columnspan=3, sticky="w", pady=(6, 0))

        self._detector_tuning_refresh()    # start the ~10 Hz live view (self-reschedules)

    def _apply_gripper_conf(self, *_):
        """Push the gripper-detector confidence to the live recovery detector (takes effect next scan)."""
        try:
            self.inference.set_gripper_param("conf", float(self._tune_grip_conf.get()))
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    def _detector_tuning_refresh(self):
        """~10 Hz: while this tab is visible, grab the head frame, run the live detector's analyze(),
        draw green/red boxes, and render. Skips the heavy work when the tab is hidden."""
        lbl = getattr(self, "_tune_view", None)
        if lbl is None:
            return
        try:
            viewable = bool(lbl.winfo_viewable())
        except tk.TclError:
            return                          # widget destroyed
        if viewable:
            try:
                self._render_detector_tuning()
            except Exception as e:
                try:
                    self._tune_stats_var.set(f"error: {e}")
                except tk.TclError:
                    return
        self.root.after(160, self._detector_tuning_refresh)

    def _render_detector_tuning(self):
        # Source selector: "gripper" -> right-wrist camera + grasp-recovery YOLO; else head + no-flip.
        source = getattr(self, "_tune_source_var", None)
        source = source.get() if source is not None else "head"
        cam = "hand_right" if source == "gripper" else "head"
        frame = None
        try:
            frame = self.env.get_frame(cam)      # also keeps that camera ON while this tab is up
        except Exception:
            frame = None
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            self._tune_stats_var.set(f"waiting for {cam} camera…")
            return
        img = frame
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)

        if hasattr(self, "inference"):
            det = (self.inference.gripper_detector() if source == "gripper"
                   else self.inference.no_flip_detector())
        else:
            det = None
        # Show/analyze only what the detector sees: crop off the same noisy top strip it ignores, so the
        # view matches detection and the drawn boxes are in the cropped frame's coordinates.
        if det is not None and hasattr(det, "crop_roi"):
            img = det.crop_roi(img)
        vis = np.ascontiguousarray(img)          # RGB working copy for drawing

        accepted = drawn = 0
        if det is not None and hasattr(det, "analyze"):
            cands = det.analyze(vis)                  # YoloGate: predict on the RGB frame -> box dicts
            # Accepted detections (reason is None) are drawn green with their label+confidence; any box
            # the detector reports as a reject is drawn red only if big enough on screen to be worth
            # looking at. Keys other than x/y/w/h are read defensively so any detector's dicts render.
            reject_floor = self._REJECT_DRAW_MIN_FRAC * (vis.shape[0] * vis.shape[1])
            for c in cands:
                x, y, w, h = c["x"], c["y"], c["w"], c["h"]
                if c.get("reason") is None:
                    col = (0, 200, 0)
                    txt = c.get("text") or f"{w}x{h}"
                    accepted += 1
                elif (w * h) < reject_floor:
                    continue                         # too small on screen to be useful -> skip
                else:
                    col, txt = (235, 70, 70), c.get("reason")
                cv2.rectangle(vis, (x, y), (x + w, y + h), col, 2)
                cv2.putText(vis, txt, (x, max(14, y - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
                drawn += 1
            avail = getattr(det, "available", None)
            note = "" if avail is not False else "  ·  model not loaded (add weights)"
            self._tune_stats_var.set(
                f"[{cam}]  detections {accepted}  ·  {drawn} box(es) shown{note}")
        else:
            self._tune_stats_var.set(f"[{cam}]  no YOLO detector loaded (showing raw frame)")

        # resize to the view width, keep aspect, render (PhotoImage kept referenced to avoid GC)
        pil = Image.fromarray(vis)
        tw = self._tune_view_w
        if pil.width != tw:
            pil = pil.resize((tw, max(1, round(pil.height * tw / pil.width))), Image.Resampling.BILINEAR)
        photo = ImageTk.PhotoImage(pil)
        try:
            self._tune_view.config(image=photo)
            self._tune_view.image = photo        # keep a reference
        except tk.TclError:
            pass
