"""Live evaluation dashboard.

Reads the append-only trial logs that ``scripts/eval_trials.py`` writes
(``infer_logs/eval/<session>.jsonl``) and renders them as a product-style KPI
board that refreshes ~1 Hz: a success-rate ring, headline metric tiles, and an
outcome histogram. It only *reads* files, so it stays in perfect sync with a
running eval session (or the demo's synthetic session) without touching the
robot or the eval tool.

The small stat helpers below are intentionally a self-contained copy of the
pure functions in ``scripts/eval_trials.py`` (``load_trials`` / Wilson interval)
so the GUI has no import dependency on that CLI script.
"""

import json
import math
import pathlib
import tkinter as tk
from tkinter import ttk

EVAL_DIR = pathlib.Path(__file__).resolve().parent.parent / "infer_logs" / "eval"

# Canonical outcome order; index 0 is the success category. Mirrors eval_trials.
CATEGORIES = ["success", "grasp-miss", "dropped", "misplaced", "collision", "estop", "other"]


def _load_trials(path):
    """Read an append-only session file into a list of trial dicts (honoring the
    undo / note tombstones the eval tool writes)."""
    if not path or not path.exists():
        return []
    trials = []
    try:
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            op = rec.get("_op")
            if op == "undo":
                if trials:
                    trials.pop()
            elif op == "note":
                if trials:
                    trials[-1]["note"] = rec.get("note", "")
            else:
                trials.append(rec)
    except (OSError, json.JSONDecodeError):
        return trials
    return trials


def _wilson(successes, n, z=1.96):
    """95% Wilson score interval — the right CI for a handful of trials."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _stats(trials):
    """Fold trials into the numbers the dashboard shows."""
    n = len(trials)
    succ = sum(1 for t in trials if t.get("outcome") == "success")
    lo, hi = _wilson(succ, n)
    durs = [t["duration_s"] for t in trials if t.get("duration_s") is not None]
    mean = sum(durs) / len(durs) if durs else None
    std = (sum((d - mean) ** 2 for d in durs) / len(durs)) ** 0.5 if durs else None
    succ_durs = [t["duration_s"] for t in trials
                 if t.get("outcome") == "success" and t.get("duration_s") is not None]
    succ_mean = sum(succ_durs) / len(succ_durs) if succ_durs else None
    tracked = [t for t in trials if t.get("grasp_misses") is not None]
    first_try = sum(1 for t in tracked if t["grasp_misses"] == 0)
    first_rate = first_try / len(tracked) if tracked else None
    counts = {c: 0 for c in CATEGORIES}
    for t in trials:
        o = t.get("outcome", "other")
        counts[o] = counts.get(o, 0) + 1
    return dict(n=n, succ=succ, rate=(succ / n if n else 0.0), ci=(lo, hi),
                mean=mean, std=std, succ_mean=succ_mean,
                first_rate=first_rate, n_tracked=len(tracked), counts=counts)


class EvalMixin:
    """Live KPI dashboard tab, driven by the eval JSONL logs."""

    def setup_eval_panel(self, parent):
        t = self.theme
        wrap = ttk.Frame(parent)
        wrap.pack(fill=tk.BOTH, expand=True, padx=t.PAD_LG, pady=t.PAD_MD)

        # ---- toolbar: session picker + freshness ----
        bar = ttk.Frame(wrap)
        bar.pack(fill=tk.X, pady=(0, t.PAD_MD))
        ttk.Label(bar, text="Policy Evaluation", style="Section.TLabel").pack(side=tk.LEFT)
        self._ev_session_var = tk.StringVar(value="")
        self._ev_session = ttk.Combobox(bar, textvariable=self._ev_session_var,
                                        width=26, state="readonly")
        self._ev_session.pack(side=tk.RIGHT)
        self._ev_session.bind("<<ComboboxSelected>>", lambda _e: self._refresh_eval(force=True))
        ttk.Label(bar, text="Session", style="Caption.TLabel").pack(side=tk.RIGHT, padx=(0, 6))
        self._ev_fresh_var = tk.StringVar(value="—")
        ttk.Label(bar, textvariable=self._ev_fresh_var,
                  style="Caption.TLabel").pack(side=tk.RIGHT, padx=(0, 16))

        # ---- top row: success ring + metric tiles ----
        top = ttk.Frame(wrap)
        top.pack(fill=tk.X)

        ring_card = ttk.Labelframe(top, text="  Success rate  ")
        ring_card.pack(side=tk.LEFT, fill=tk.Y, padx=(0, t.PAD_MD))
        self._ev_ring = tk.Canvas(ring_card, width=190, height=190,
                                  highlightthickness=0, bg=t.APP_BG)
        self._ev_ring.pack(padx=t.PAD_MD, pady=t.PAD_MD)

        tiles = ttk.Frame(top)
        tiles.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._ev_tiles = {}
        specs = [("trials", "TRIALS"), ("mean", "MEAN TIME"),
                 ("first", "FIRST-TRY GRASP"), ("succ_mean", "SUCCESS TIME")]
        for i, (key, label) in enumerate(specs):
            r, c = divmod(i, 2)
            tiles.columnconfigure(c, weight=1, uniform="tile")
            tiles.rowconfigure(r, weight=1, uniform="tile")
            card = ttk.Frame(tiles, style="Card.TFrame")
            card.grid(row=r, column=c, sticky="nsew", padx=4, pady=4, ipadx=8, ipady=6)
            ttk.Label(card, text=label, style="MetricLabel.TLabel").pack(anchor="w", padx=12, pady=(10, 0))
            var = tk.StringVar(value="—")
            ttk.Label(card, textvariable=var, style="Metric.TLabel").pack(anchor="w", padx=12, pady=(0, 10))
            self._ev_tiles[key] = var

        # ---- outcome histogram ----
        hist_card = ttk.Labelframe(wrap, text="  Outcome breakdown  ")
        hist_card.pack(fill=tk.BOTH, expand=True, pady=(t.PAD_MD, 0))
        self._ev_hist = tk.Canvas(hist_card, height=210, highlightthickness=0, bg=t.APP_BG)
        self._ev_hist.pack(fill=tk.BOTH, expand=True, padx=t.PAD_MD, pady=t.PAD_MD)

        self._ev_last_path = None
        self._ev_after_id = None
        self._refresh_eval(force=True)
        self._ev_after_id = self.root.after(1000, self._eval_tick)

    # ------------------------------------------------------------------ #
    def _eval_sessions(self):
        if not EVAL_DIR.exists():
            return []
        return sorted(EVAL_DIR.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)

    def _eval_tick(self):
        try:
            self._refresh_eval()
        except tk.TclError:
            self._ev_after_id = None
            return
        self._ev_after_id = self.root.after(1000, self._eval_tick)

    def _refresh_eval(self, force=False):
        sessions = self._eval_sessions()
        names = [p.stem for p in sessions]
        if names != list(self._ev_session["values"]):
            self._ev_session["values"] = names
        # Resolve the active session: explicit selection, else newest.
        sel = self._ev_session_var.get()
        path = None
        if sel and sel in names:
            path = sessions[names.index(sel)]
        elif sessions:
            path = sessions[0]
            self._ev_session_var.set(path.stem)

        if path is None:
            self._ev_fresh_var.set("no eval sessions yet")
            self._draw_ring(None)
            self._draw_hist(None)
            for v in self._ev_tiles.values():
                v.set("—")
            return

        st = _stats(_load_trials(path))
        import time as _t
        age = _t.time() - path.stat().st_mtime
        self._ev_fresh_var.set("updated just now" if age < 3
                               else f"updated {int(age)}s ago")
        self._draw_ring(st)
        self._draw_hist(st)
        self._ev_tiles["trials"].set(str(st["n"]))
        self._ev_tiles["mean"].set(f"{st['mean']:.1f}s" if st["mean"] is not None else "—")
        self._ev_tiles["first"].set(
            f"{100 * st['first_rate']:.0f}%" if st["first_rate"] is not None else "—")
        self._ev_tiles["succ_mean"].set(
            f"{st['succ_mean']:.1f}s" if st["succ_mean"] is not None else "—")

    # ------------------------------------------------------------------ #
    def _draw_ring(self, st):
        c = self._ev_ring
        t = self.theme
        c.delete("all")
        W = int(c["width"]); H = int(c["height"])
        pad = 16
        rate = st["rate"] if st else 0.0
        # Track + progress arcs (thick, rounded look).
        c.create_arc(pad, pad, W - pad, H - pad, start=0, extent=359.9,
                     style="arc", width=16, outline=t.BORDER)
        col = t.SUCCESS if rate >= 0.6 else (t.WARN if rate >= 0.35 else t.DANGER)
        if rate > 0:
            c.create_arc(pad, pad, W - pad, H - pad, start=90, extent=-359.9 * rate,
                         style="arc", width=16, outline=col)
        pct = f"{100 * rate:.0f}%" if st and st["n"] else "—"
        c.create_text(W / 2, H / 2 - 8, text=pct, fill=t.TEXT, font=t.mono(30, "bold"))
        if st and st["n"]:
            lo, hi = st["ci"]
            c.create_text(W / 2, H / 2 + 22,
                          text=f"{st['succ']}/{st['n']}  ·  CI {100*lo:.0f}–{100*hi:.0f}%",
                          fill=t.TEXT_MUTED, font=t.ui(8))
        else:
            c.create_text(W / 2, H / 2 + 22, text="awaiting trials",
                          fill=t.TEXT_MUTED, font=t.ui(8))

    def _draw_hist(self, st):
        c = self._ev_hist
        t = self.theme
        c.delete("all")
        c.update_idletasks()
        W = c.winfo_width() or 700
        counts = st["counts"] if st else {k: 0 for k in CATEGORIES}
        cats = CATEGORIES
        mx = max(counts.values()) if counts and max(counts.values()) > 0 else 1
        left = 130                      # label gutter
        right = 48                      # count gutter
        top = 12
        row_h = 26
        bar_area = max(60, W - left - right)
        for i, cat in enumerate(cats):
            y = top + i * row_h
            n = counts.get(cat, 0)
            fill = t.SUCCESS if cat == "success" else (t.MUTED if n == 0 else t.DANGER)
            c.create_text(left - 12, y + row_h / 2 - 4, text=cat, anchor="e",
                          fill=t.TEXT if n else t.TEXT_FAINT, font=t.ui(9, "bold"))
            # track
            c.create_rectangle(left, y, left + bar_area, y + row_h - 8,
                               fill=t.SURFACE_ALT, outline="")
            w = int(bar_area * (n / mx))
            if w > 0:
                c.create_rectangle(left, y, left + w, y + row_h - 8,
                                   fill=fill, outline="")
            c.create_text(left + bar_area + 24, y + row_h / 2 - 4, text=str(n),
                          fill=t.TEXT if n else t.TEXT_FAINT, font=t.mono(9, "bold"))
