"""Detector-tuning page: a live head-camera view with detection boxes drawn, next to the LabelGate
threshold controls — so you can tune the no-flip label detector and watch the effect in real time.

Boxes: GREEN = an accepted label block (labelled WxH + char count); RED = a rejected candidate,
labelled with the FIRST gate it failed ('small' / 'fill' / 'aspect' / 'density' / 'chars'). Adjust the
sliders on the right until the things you WANT are green and everything else is red/absent.

The view runs only while this tab is visible (winfo_viewable), and it runs the SAME LabelGate instance
the macro uses (via inference.no_flip_detector()), so edits here apply to the live robot immediately.
"""
import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageTk


class DetectorTuningMixin:
    """Live label-detector tuning tab (view + threshold controls)."""

    def setup_detector_tuning_panel(self, parent):
        self._tune_view_w = 760            # rendered view width (px); height follows the frame aspect

        outer = ttk.Frame(parent)
        outer.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # ---- left: live annotated head-camera view ----
        left = ttk.LabelFrame(outer, text="  Head camera · live detection  ")
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))
        self._tune_view = tk.Label(left, bg="#111418")
        self._tune_view.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self._tune_stats_var = tk.StringVar(value="starting…")
        ttk.Label(left, textvariable=self._tune_stats_var, style="Value.TLabel").pack(
            anchor="w", padx=8, pady=(0, 6))
        ttk.Label(left, style="Caption.TLabel", justify="left",
                  text="GREEN = accepted label block (WxH + #chars).   RED = rejected — the tag is the "
                       "first failed gate (small / fill / aspect / density / chars).").pack(
            anchor="w", padx=8, pady=(0, 6))

        # ---- right: threshold controls (same live tunables as the macro) ----
        right = ttk.LabelFrame(outer, text="  Thresholds · live  ")
        right.pack(side=tk.RIGHT, fill=tk.Y, expand=False, padx=(8, 0))
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

        self._detector_tuning_refresh()    # start the ~10 Hz live view (self-reschedules)

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
        self.root.after(100, self._detector_tuning_refresh)

    def _render_detector_tuning(self):
        frame = None
        try:
            frame = self.env.get_frame("head")   # also keeps the head camera ON while this tab is up
        except Exception:
            frame = None
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            self._tune_stats_var.set("waiting for head camera…")
            return
        img = frame
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        vis = np.ascontiguousarray(img)          # RGB working copy for drawing

        det = self.inference.no_flip_detector() if hasattr(self, "inference") else None
        accepted = total = 0
        if det is not None and hasattr(det, "analyze"):
            gray = cv2.cvtColor(vis, cv2.COLOR_RGB2GRAY)
            cands = det.analyze(gray)
            total = len(cands)
            for c in cands:
                x, y, w, h = c["x"], c["y"], c["w"], c["h"]
                if c["reason"] is None:
                    col, txt = (0, 200, 0), f"{w}x{h} {c['comps']}ch"
                    accepted += 1
                else:
                    col, txt = (235, 70, 70), c["reason"]
                cv2.rectangle(vis, (x, y), (x + w, y + h), col, 2)
                cv2.putText(vis, txt, (x, max(14, y - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
            self._tune_stats_var.set(f"accepted {accepted} / {total} candidate block(s)")
        else:
            self._tune_stats_var.set("no LabelGate detector loaded (showing raw frame)")

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
