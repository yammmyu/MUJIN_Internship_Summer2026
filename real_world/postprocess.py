"""Post-inference action pipeline for the dual-arm policy: raw policy chunk -> robot-ready substeps.

The OUTPUT-side mirror of build_data.py (which builds obs -> /predict request). Everything that
happens to a policy action chunk AFTER the server returns it lives here, in one place, in execution
order, so the whole path reads top to bottom:

  1. GRIPPER BINARIZE        raw [0,~85] grip -> {0,1} per arm (col 9 left / col 19 right)
  2. TEMPORAL-ENSEMBLE MERGE cross-chunk recency-weighted mean + along-id Gaussian, master-id aligned
  (later stages absorb: 3. dual-arm IK, 4. sim validation, 5. master-id tag + seam ramp + subdivide
   + queue splice — see the plan; each is moved here behavior-preserving.)

A PostProcessor is the single owner of the merge buffer (and, in later stages, the robot queue); the
env owns the hardware, the sim, and the release-loop CONSUMER, and constructs one PostProcessor. The
inference controller hands each server chunk to PostProcessor.merge and (when streaming) the result
is spliced onto the robot queue.

Master-row-id alignment: every action row carries an absolute MASTER ROW ID (the robot's own
execution clock; see real_world/timing.py). Row j of a chunk anchored at S is at id S + j, so the
merge tracks the arm's real progress and never runs ahead of the arm when execution lags real-time.
"""

import json
import logging
import pathlib
import time
from collections import deque

import numpy as np

from real_world.ik import IKQuery, smooth_quat_step, decode_action_row_dual
from real_world.timing import MAX_JOINT_STEP, SUBSTEPS_PER_ROW, RAMP_JOINT_STEP

log = logging.getLogger(__name__)

# Raw gripper value at/above which we call the gripper CLOSED (-> 1). The [0,~85] signal is noisy
# (transient spikes), so only a (near-)fully-closed reading binarizes to 1. Single home for the
# pipeline; humanoid_env re-exports it for back-compat (scripts import it from there).
GRIPPER_CLOSE_THRESH = 50.0

# How many policy rows append_actions keeps queued AHEAD of the master clock (the streaming commit-
# ahead distance). Live-tunable via the pipeline. humanoid_env re-exports it for back-compat.
APPEND_AHEAD_ROWS = 2

# --- Temporal ensemble (ACT-style) DEFAULT SEEDS ---------------------------------------------------
# These four (TE_M, TE_BUFFER_LEN, TE_RADIUS, TE_SIGMA) are only DEFAULT SEEDS: the GUI restores the
# operator's saved values from tuning_config.json at startup and tunes them live (set_smoothing /
# set_buffer_len). Editing a literal here only changes the fallback when there is no saved JSON entry
# (e.g. a headless run). The live runtime values are pp.te_m / .te_buffer_len / .te_radius / .te_sigma.
USE_TEMPORAL_ENSEMBLE = True   # master on/off for ALL temporal-ensemble smoothing.
TE_M = 0.02            # recency decay of the per-id cross-chunk mean. LARGER -> trust the newest
                       # chunk more (less averaging, more reactive, rougher); SMALLER -> smoother.
TE_BUFFER_LEN = 8      # how many recent raw chunks are kept = max overlap depth to average over.
TE_RADIUS = 6          # Gaussian half-width (ids) == # of frozen rows retained as left-context.
TE_SIGMA = 1.5         # Gaussian sigma (id units); keep <= TE_RADIUS or the kernel is clipped.
# Smoothness guard: warn when a per-row position step in the buffer exceeds this (m). Diagnostic only
# (logs a warning, never clamps) — the seam between chunks is where roughness shows up.
SMOOTHNESS_WARN_DPOS = 0.03

# --- Stage 3: dual-arm IK (workspace envelopes + orientation smoothing) ----------------------------
# Orientation EMA factor toward the new target quaternion (H3; 0..1, 1 = no smoothing). The ONLY
# orientation smoother on the robot path (applied row-to-row in solve_chunk_ik). 1.0 -> no smoothing
# (snappiest, roughest); LOWER -> smoother orientation but more lag.
QUAT_ALPHA = 0.5
# Workspace envelope (firmware EE frame, metres) the LEFT policy target EE pos must lie in. A target
# outside is rejected (never sent to IK/robot). Generous box; tighten per workspace. (H4)
WORKSPACE_AABB = ((-0.20, 0.85), (-0.20, 1.10), (0.40, 1.30))   # (x_lo,x_hi),(y..),(z..) LEFT arm
# RIGHT-arm envelope (H4): the right arm lives at NEGATIVE y (recorded right EE y ~[-0.38, -0.17]), so
# its AABB is the LEFT box mirrored across y=0 (same x, z). Per-arm envelopes keep the bound tight.
WORKSPACE_AABB_RIGHT = ((-0.20, 0.85), (-1.10, 0.20), (0.40, 1.30))


def pos_in_workspace(pos, side="left"):
    """True if the target EE position is inside that arm's workspace AABB (H4). side="right" uses the
    mirrored right-arm envelope. Reads the module globals at call time so the safety suite can widen
    them to isolate the release-pipeline invariants."""
    (xl, xh), (yl, yh), (zl, zh) = WORKSPACE_AABB_RIGHT if side == "right" else WORKSPACE_AABB
    x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
    return xl <= x <= xh and yl <= y <= yh and zl <= z <= zh


# --- Runtime EE safe region (C7) -----------------------------------------------------------------
# A HARD spatial boundary (firmware EE frame, metres) checked at DISPATCH, per arm. Unlike the H4
# envelope above — a generous, hand-set gate that only REJECTS a planning target (skip the row) —
# this region is data-estimated and, when the actually-commanded EE leaves it, the release path
# LATCHES THE E-STOP. It is the spatial sibling of the C6 joint-jump watchdog: C6 catches a large
# per-command rotation, C7 catches the EE straying outside where the robot has ever safely operated
# (a slow drift, an accumulated excursion, or a bad target that slipped past validation).
#
# Estimated from MDM_data_collection/recordings (165 episodes, ~87.5k frames/arm): the full
# per-axis min/max of the demonstrated left/right EE positions, expanded by a 0.12 m margin (so
# 100% of demonstrated motion is inside, with headroom for legitimate policy extrapolation) and
# rounded OUTWARD to the cm. Re-estimate with scripts/estimate_ee_region.py if the task/workcell
# changes. TUNE: widen the margin to reduce nuisance trips, tighten it for a stricter cutoff.
EE_SAFE_REGION_LEFT  = ((0.23, 0.80), (0.04, 0.59), (0.37, 0.84))   # (x_lo,x_hi),(y..),(z..) LEFT
EE_SAFE_REGION_RIGHT = ((0.28, 0.83), (-0.56, 0.01), (0.36, 0.92))  # RIGHT arm (negative-y half)


def ee_in_safe_region(pos, side="left"):
    """True if the commanded EE position is inside that arm's data-estimated safe region (C7).
    side="right" uses the right-arm region. Reads the module globals at call time so the safety
    suite / a workcell config can override the box."""
    (xl, xh), (yl, yh), (zl, zh) = EE_SAFE_REGION_RIGHT if side == "right" else EE_SAFE_REGION_LEFT
    x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
    return xl <= x <= xh and yl <= y <= yh and zl <= z <= zh


def load_nominal_config():
    """Nominal per-arm training posture (real_world/config/nominal_arm_config.json) as a 14-vec
    [left7, right7] — the IK FALLBACK seed. Zeros if the file is absent (fallback disabled-ish)."""
    path = pathlib.Path(__file__).parent / "config" / "nominal_arm_config.json"
    try:
        d = json.loads(path.read_text())
        return np.asarray(list(d["left"]) + list(d["right"]), dtype=np.float64)
    except Exception as e:
        print(f"[PostProcessor] nominal_arm_config.json unavailable ({e}); IK fallback seed = zeros.")
        return np.zeros(14, dtype=np.float64)


def load_retreat_waypoints():
    """The pre-grasp APPROACH waypoints (real_world/config/retreat_waypoints.json) as an (n, 14)
    array [left7, right7] per row, ordered from the average start pose (row 0) to ~85% of the way to
    the grasp (last row). The recovery retreat homes to the nearest of these that isn't ahead of the
    arm's current approach phase; row 0 is also the auto-run start pose. Re-generate with
    scripts/estimate_retreat_waypoints.py. Returns None if the file is absent (callers fall back to a
    single fixed home pose)."""
    path = pathlib.Path(__file__).parent / "config" / "retreat_waypoints.json"
    try:
        d = json.loads(path.read_text())
        W = np.asarray(d["waypoints"], dtype=np.float64)
        if W.ndim != 2 or W.shape[1] != 14 or len(W) == 0:
            raise ValueError(f"expected (n,14) waypoints, got {W.shape}")
        return W
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"[PostProcessor] retreat_waypoints.json unusable ({e}); retreat falls back to home pose.")
        return None


def _limit_pinned(solver, q, margin=0.10):
    """True if any joint of q sits within `margin` rad of its limit — the signature of a CONTORTED IK
    solution (the redundant DOF shoved into a limit to reach the target)."""
    lo = np.asarray(solver.m.lower, dtype=np.float64)
    hi = np.asarray(solver.m.upper, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    return float(np.min(np.minimum(q - lo, hi - q))) < margin


class _NullLock:
    """No-op context manager so a standalone PostProcessor (the ensemble test builds one with no env)
    can run the queue paths lock-free. The env always injects its real shared lock."""
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False


class PostProcessor:
    """Owns the post-inference pipeline: temporal-ensemble merge + dual-arm IK + sim validation (and,
    in a later stage, the robot queue). References the env's LEFT/RIGHT IK solvers; owns the merge
    state and the nominal-posture IK fallback. Standalone-constructible: the ensemble tests build one
    with no solvers (solver/solver_r default None -> merge works, IK is unavailable)."""

    def __init__(self, solver=None, solver_r=None, nominal_q14=None,
                 lock=None, read_arm14=None, estopped=None, get_sim=None, real=False,
                 substeps_per_row=SUBSTEPS_PER_ROW, ramp_joint_step=RAMP_JOINT_STEP,
                 append_ahead_rows=APPEND_AHEAD_ROWS,
                 use_temporal_ensemble=USE_TEMPORAL_ENSEMBLE):
        # --- IK (stage 3): references to the env's solvers + the nominal-posture fallback seed ---
        self.solver = solver                       # LEFT-arm IK solver (None in the ensemble-only test)
        self.solver_r = solver_r                   # RIGHT-arm IK solver
        # Nominal per-arm joint posture (median of the training recordings): the IK FALLBACK warm-start
        # when the live/chained seed resolves to a contorted or unreachable config.
        self.nominal_q14 = (load_nominal_config() if nominal_q14 is None
                            else np.asarray(nominal_q14, dtype=np.float64))
        # --- injected env dependencies (stage 5: queue splice) ---------------------------------------
        # The queue is shared with the env's release loop (consumer) + manual-release path, so it is
        # guarded by the SAME lock the env uses (injected). read_arm14/estopped/get_sim are callables
        # into the env (live dual-arm pose read, E-stop predicate, current sim handle); `real` mirrors
        # env.real. A no-op lock lets a standalone PostProcessor (ensemble test) run lock-free.
        self._lock = lock if lock is not None else _NullLock()
        self.read_arm14 = read_arm14 if read_arm14 is not None else (lambda: None)
        self.estopped = estopped if estopped is not None else (lambda: False)
        self.get_sim = get_sim if get_sim is not None else (lambda: None)
        self.real = real
        self._log_ts = {}                          # key -> last monotonic print time (throttling)
        # --- LIVE-TUNABLE execution knobs (env.set_speed_scale updates these) ---
        self.substeps_per_row = substeps_per_row   # K: time-uniform substeps each policy row expands to
        self.ramp_joint_step = ramp_joint_step     # per-substep joint delta for seam ramps (SPEED_SCALE'd)
        self.append_ahead_rows = append_ahead_rows # rows kept queued ahead of the clock (streaming)
        # --- queue state (stage 5), guarded by self._lock ---
        self._robot_q = deque()                    # released, sim-validated cmds pending on robot
        self._current_row_id = 0                   # master row clock (advanced as substeps dispatch)
        self._queued_through = -1                  # highest master id in _robot_q; -1 = nothing queued
        self._last_q14 = None                      # dual IK warm-start seed [left7, right7]; env seeds it
        # --- merge (stage 1-2) ---
        self.use_temporal_ensemble = use_temporal_ensemble
        # Rolling buffer of recent RAW chunks (sid, action[N,W]), OLDEST -> NEWEST. Always averaged
        # from raw — never re-buffer ensembled output, or smoothing compounds.
        self._te_buffer = deque(maxlen=TE_BUFFER_LEN)
        # LIVE-TUNABLE smoothing config held as ONE immutable snapshot (radius, sigma, m, gauss) so the
        # inference thread reads a consistent set even while the GUI changes it (ref assignment is
        # atomic under the GIL). Scalar mirrors (te_radius/te_sigma/te_m) are display/readback only.
        self.te_radius = TE_RADIUS
        self.te_sigma = TE_SIGMA
        self.te_m = TE_M
        self._smooth = self._make_smoothing(TE_RADIUS, TE_SIGMA, TE_M)
        # The long smoothed buffer fed to the robot: master_id -> EE row [pos3, rot6d6, grip1].
        # Ids <= queued_through are FROZEN (committed, read-only context); ids above it are MUTABLE and
        # rebuilt every merge. Only the inference thread touches this -> no lock.
        self._buffer = {}
        # Last queued_through seen: a DROP (env re-anchored on E-stop -> -1) means the live queue was
        # cleared, so our buffer is stale and must be cleared too before rebuilding.
        self._last_queued_through = -1

    def reset_merge(self):
        """Clear the temporal-ensemble state (called on inference start / re-anchor)."""
        self._te_buffer.clear()
        self._buffer.clear()
        self._last_queued_through = -1

    # ------------------------------------------------------------------ #
    #  Stage 1: gripper binarize + temporal-ensemble merge                 #
    # ------------------------------------------------------------------ #
    @staticmethod
    def binarize_grippers(action):
        """Binarize the gripper column(s) IN PLACE as soon as the chunk arrives: col 9 = left grip,
        col 19 = right grip (dual). Only a (near-)fully-closed reading (>= GRIPPER_CLOSE_THRESH)
        becomes closed=1, else open=0, so downstream carries a clean {0,1} per arm and gripper spikes
        no longer pollute the next inference's state context. 10-col (left-only) rows touch col 9 only."""
        if action.ndim == 2:
            for gc in (9, 19):
                if action.shape[1] > gc:
                    action[:, gc] = (action[:, gc] >= GRIPPER_CLOSE_THRESH).astype(action.dtype)

    def merge(self, action, sid, cur, queued_through, submit):
        """Binarize grippers, run the temporal-ensemble merge, and return (base_id, buf, jerk).

        The dual_arm policy returns 20-col rows: L[pos3,rot6d6,grip1] ++ R[pos3,rot6d6,grip1] (10-col
        left-only rows pass through unchanged for a single-arm policy). `sid` is the chunk's anchor
        master id; `cur`/`queued_through` are the env's live clock + commit frontier (queue_status).

        AUTO/streaming (submit=True, TE on): append this raw chunk to the rolling buffer, then rebuild
        the long smoothed master buffer over its MUTABLE tail (id > queued_through) and materialize the
        contiguous run. The BUFFER — not this chunk — is what feeds the robot, so we keep one
        continuous id-to-id smooth sequence instead of re-emitting per-inference chunks.
        Manual one-shots (submit=False) / TE off / a narrow chunk bypass the buffer and pass the raw
        chunk through (base_id = sid), so manual step-through reads an un-smoothed chunk and the
        streaming buffer isn't polluted while queued_through sits at -1."""
        action = np.asarray(action, dtype=np.float32)
        self.binarize_grippers(action)                    # in-place: caller's chunk carries {0,1} grip
        use_buffer = (submit and self.use_temporal_ensemble
                      and action.ndim == 2 and action.shape[1] >= 10)
        if use_buffer:
            if queued_through < self._last_queued_through:     # env re-anchored (E-stop) -> stale
                self._buffer.clear()
                self._te_buffer.clear()
            self._last_queued_through = queued_through
            self._te_buffer.append((sid, action.copy()))       # raw chunk, keyed on master row id
            self._rebuild_buffer(queued_through)
            base_id, buf = self._materialize()
        else:                                                  # manual / TE off -> raw chunk direct
            base_id, buf = sid, action
        jerk = self._smoothness_guard(buf, base_id, cur, queued_through, use_buffer)
        return base_id, buf, jerk

    def _smoothness_guard(self, buf, base_id, cur, queued_through, use_buffer):
        """The buffer feeds the robot, so any id-to-id jump becomes a fast seam. Measure the executed
        sequence's roughness in EE space over the STILL-MUTABLE rows (frozen rows are committed):
        |Δpos| is the per-row position step and |Δ²pos| the bend (jerk proxy). Warn (log, never clamp)
        when a step is unusually large so a bad merge surfaces. Returns the jerk for the trace."""
        jerk = 0.0
        if use_buffer and buf.shape[0] >= 3:
            mut0 = max(0, queued_through + 1 - int(base_id))   # index of first mutable row in buf
            seg = buf[max(0, mut0 - 1):]                       # include one frozen anchor for the seam
            if seg.shape[0] >= 3:
                dpos = np.linalg.norm(np.diff(seg[:, :3], axis=0), axis=1)
                jerk = float(np.linalg.norm(np.diff(seg[:, :3], n=2, axis=0), axis=1).max())
                if dpos.max() > SMOOTHNESS_WARN_DPOS:
                    log.warning("SMOOTHNESS WARN: buffer |Δpos|max=%.4f |Δ²pos|max=%.4f m "
                                "near seam (clock=%s, qt=%s)", dpos.max(), jerk, cur, queued_through)
        return jerk

    @staticmethod
    def _make_smoothing(radius, sigma, m):
        """Build an immutable smoothing snapshot {radius, sigma, m, gauss} with the normalized
        symmetric Gaussian kernel precomputed. radius>=0 int, sigma>0, m>=0."""
        radius = max(0, int(radius))
        sigma = max(1e-3, float(sigma))
        ks = np.arange(-radius, radius + 1)
        g = np.exp(-(ks ** 2) / (2.0 * sigma ** 2))
        g = g / g.sum()
        return {"radius": radius, "sigma": sigma, "m": max(0.0, float(m)), "gauss": g}

    def set_buffer_len(self, n):
        """LIVE-tunable max number of recent raw chunks averaged (= overlap depth). Rebuilds the
        deque preserving the most-recent contents. Returns the applied maxlen."""
        n = max(1, int(n))
        self._te_buffer = deque(self._te_buffer, maxlen=n)   # keeps newest n (deque drops from left)
        log.info("te_buffer_len set: %d", n)
        return n

    @property
    def te_buffer_len(self):
        return self._te_buffer.maxlen

    def set_smoothing(self, radius=None, sigma=None, m=None):
        """LIVE-tunable smoothing update (safe to call from the GUI thread while inference runs).
        Any arg left None keeps its current value. Rebuilds the kernel and swaps the whole config in
        one atomic reference assignment, so _rebuild_buffer always reads a consistent (radius, kernel)
        pair. Returns the applied (radius, sigma, m)."""
        radius = self.te_radius if radius is None else radius
        sigma = self.te_sigma if sigma is None else sigma
        m = self.te_m if m is None else m
        snap = self._make_smoothing(radius, sigma, m)
        self._smooth = snap                               # atomic swap (single ref assignment)
        self.te_radius, self.te_sigma, self.te_m = snap["radius"], snap["sigma"], snap["m"]
        log.info("smoothing set: radius=%d sigma=%.2f m=%.3f",
                 self.te_radius, self.te_sigma, self.te_m)
        return self.te_radius, self.te_sigma, self.te_m

    def _rebuild_buffer(self, queued_through):
        r"""Rebuild the long smoothed master buffer (self._buffer: master_id -> EE row
        [pos3, rot6d6, grip1]) from the recent RAW chunks in self._te_buffer. Two smoothing
        dimensions, in EE-pose space, all aligned by MASTER ROW ID (row j of a chunk anchored at S
        is at id S + j):

          (a) ACROSS CHUNKS (per id): for each mutable id, gather every buffered chunk's row at the
              SAME absolute id and take a recency-weighted mean (w = exp(-TE_M * age), age = how many
              inferences old, newest = 0). Exact integer alignment. The ACT temporal ensemble.
          (b) ALONG THE ID AXIS (neighbors): low-pass the per-id estimate with a symmetric Gaussian
              of half-width TE_RADIUS so the sequence is smooth id-to-id, not just averaged per id.

        Split by the env's commit frontier:
          * FROZEN  (id <= queued_through): committed to the robot. Kept (the last TE_RADIUS) as
            READ-ONLY left-context for (b) so the seam to the rows already on the arm stays continuous.
          * MUTABLE (id >  queued_through): rebuilt here every merge from passes (a) then (b).

        pos (0:3) and rot6d (3:9) smooth linearly (rot6d re-orthonormalised downstream); gripper (9),
        already {0,1}, is NOT low-passed along id (that would blur open/close timing) — it carries the
        cross-chunk value, re-thresholded at 0.5. Assumes the new raw chunk is already in
        self._te_buffer (deque, OLDEST -> NEWEST), each element a (master_id, raw chunk (N,W)) tuple."""
        raw = list(self._te_buffer)                       # oldest -> newest
        if not raw:
            return
        s = self._smooth                                  # atomic snapshot (live-tunable)
        R, gauss, te_m = s["radius"], s["gauss"], s["m"]
        newest_idx = len(raw) - 1
        max_id = max(sid + a.shape[0] - 1 for sid, a in raw)
        min_sid = min(sid for sid, _ in raw)
        # First MUTABLE id (everything <= queued_through is frozen). Bounded below by the oldest
        # buffered chunk: no id below any chunk can have a cross-chunk estimate, so don't iterate there
        # (after an E-stop re-anchor queued_through+1 would otherwise be ~0 while max_id is the live
        # clock, spinning over thousands of empty ids).
        mut_lo = max(queued_through + 1, min_sid)
        n_ids = max_id - mut_lo + 1                        # mutable id i <-> master id (mut_lo + i)
        if n_ids <= 0:                                     # whole buffer already committed/frozen
            self._prune_frozen(queued_through, R)         # nothing mutable to rebuild; just prune
            return

        # Column layout: W = 10 (left-only) or 20 (dual L++R). Gripper cols {9, 19} are binary and NOT
        # low-passed along id (that would blur open/close timing); every other col (pos + rot6d, per
        # arm) is smoothed linearly. Derived from the row width so the same code handles both.
        W = raw[-1][1].shape[1]
        grip_cols = [c for c in (9, 19) if c < W]
        smooth_cols = [c for c in range(W) if c not in grip_cols]
        ns = len(smooth_cols)

        # (a) cross-chunk recency-weighted mean per mutable id (the ACT temporal ensemble). Each chunk
        # overlaps a CONTIGUOUS span of mutable ids, so it scatter-adds in one vectorised slice
        # (w*rows) instead of an id-by-id Python loop; per-id weight = sum of contributors.
        acc = np.zeros((n_ids, W), dtype=np.float64)
        wsum = np.zeros(n_ids, dtype=np.float64)
        for idx, (sid, a) in enumerate(raw):
            lo, hi = max(mut_lo, sid), min(max_id, sid + a.shape[0] - 1)
            if lo > hi:                                    # chunk doesn't reach the mutable region
                continue
            wc = np.exp(-te_m * (newest_idx - idx))        # newest chunk -> age 0
            acc[lo - mut_lo:hi - mut_lo + 1] += wc * a[lo - sid:hi - sid + 1]
            wsum[lo - mut_lo:hi - mut_lo + 1] += wc
        present = wsum > 0.0                               # ids that any chunk covered
        tent = acc / np.where(present, wsum, 1.0)[:, None]  # normalized weighted mean

        # (b) symmetric Gaussian along id over the SMOOTH (pos+rot6d, both arms) columns; frozen rows
        # (<= queued_through) are fixed left-context. Read from a dense padded context array (2R wider
        # than the mutable span) so the convolution is 2R+1 vectorised shifted adds.
        ext = n_ids + 2 * R
        ctxv = np.zeros((ext, ns), dtype=np.float64)       # neighbor smooth-col values
        pres = np.zeros(ext, dtype=np.float64)             # 1 where that neighbor exists
        ctxv[R:R + n_ids] = tent[:, smooth_cols]           # mutable estimates (pre-smoothing)
        pres[R:R + n_ids] = present
        for nid in range(mut_lo - R, mut_lo):              # frozen left-context (<= queued_through)
            v = self._buffer.get(nid) if nid <= queued_through else None
            if v is not None:
                ctxv[nid - (mut_lo - R)] = np.asarray(v)[smooth_cols]
                pres[nid - (mut_lo - R)] = 1.0
        num = np.zeros((n_ids, ns), dtype=np.float64)
        den = np.zeros(n_ids, dtype=np.float64)
        for d in range(-R, R + 1):                         # neighbor offset; ext idx of id i is R+i+d
            k = gauss[d + R]
            sl = slice(R + d, R + d + n_ids)
            num += k * ctxv[sl] * pres[sl][:, None]
            den += k * pres[sl]
        out = tent.copy()                                  # carries grippers + the (a) fallback
        good = den > 0.0
        smoothed = out[:, smooth_cols]                     # fancy-index copy; write back after filling
        smoothed[good] = num[good] / den[good][:, None]    # pos + rot6d (linear) smoothed along id
        out[:, smooth_cols] = smoothed
        for gc in grip_cols:
            out[:, gc] = (out[:, gc] >= 0.5)               # each gripper re-binarised (not low-passed)
        for i in np.nonzero(present)[0]:                   # write MUTABLE ids only; frozen untouched
            self._buffer[mut_lo + int(i)] = out[i]

        self._prune_frozen(queued_through, R)

    def _prune_frozen(self, queued_through, R):
        """Drop frozen rows older than the retention window (keep the last TE_RADIUS for context)."""
        cutoff = queued_through - R + 1
        for k in [k for k in self._buffer if k < cutoff]:
            del self._buffer[k]

    def _materialize(self):
        """The contiguous run of self._buffer starting at its lowest id, stopping at the first gap.
        Returns (base_id, ndarray(M,W)); (-1, empty) when the buffer is empty. append maps master id
        -> row by index = id - base_id, so the run handed on MUST be contiguous."""
        if not self._buffer:
            return -1, np.empty((0, 10), dtype=np.float32)
        lo = min(self._buffer)
        rows = []
        i = lo
        while i in self._buffer:                           # walk forward, stop at first missing id
            rows.append(self._buffer[i])
            i += 1
        return lo, np.asarray(rows, dtype=np.float32)

    # ------------------------------------------------------------------ #
    #  Stage 3: dual-arm IK   +   Stage 4: sim validation                  #
    # ------------------------------------------------------------------ #
    def _ik_robust(self, solver, pos, quat, seed, nominal):
        """Solve IK from the live/chained `seed`; if that lands UNREACHABLE or LIMIT-PINNED
        (contorted), retry from `nominal` (the training posture) and prefer a clean solve. Returns
        (q7, reachable). Same rule for both arms: a warm seed keeps its natural branch (no retry, no
        regression); a parked / out-of-distribution seed escapes contortion. This is what makes the
        robot's PARKED right arm resolve to a sane pose instead of 'going crazy' — the sim eval never
        saw it because it seeds from the recorded (warm) joints."""
        q1 = solver.solve(IKQuery(target_pos=pos, target_quat=quat, current_joints=seed))
        if solver.last_reachable and not _limit_pinned(solver, q1):
            return np.asarray(q1, dtype=np.float64), True          # live seed is clean -> keep it
        r1 = solver.last_reachable
        q2 = solver.solve(IKQuery(target_pos=pos, target_quat=quat, current_joints=nominal))
        if solver.last_reachable and not _limit_pinned(solver, q2):
            return np.asarray(q2, dtype=np.float64), True          # nominal fallback is clean
        if solver.last_reachable:                                  # neither clean; prefer reachable
            return np.asarray(q2, dtype=np.float64), True
        if r1:
            return np.asarray(q1, dtype=np.float64), True
        return np.asarray(q2, dtype=np.float64), False             # genuinely unreachable

    def solve_chunk_ik(self, action_chunk, seed_q, skip_unreachable=False):
        """Solve IK for an ENTIRE chunk up front, OUTSIDE the sim validation job: per-row workspace
        check (H4) + orientation smoothing (H3) + dual-arm IK, chaining each row's warm-start from the
        previous raw IK solution starting at seed_q (fresh quat-smoothing state per call). Returns
        (configs, kept_ids, ok, reason) where configs = [(q14, [gl,gr]), ...] are the raw IK joints the
        sim then validates, and kept_ids[j] = the ORIGINAL chunk-row index of configs[j].

        When a row is UNREACHABLE (either arm's IK unsolvable, or a target outside the workspace box):
          * skip_unreachable=True (auto/streaming + calibrate): DROP that row and keep going. kept_ids
            records the survivors' original indices, so the auto path still tags each kept row with its
            true master id (start_id + kept_ids[...]) — the skipped id becomes a genuine gap in the
            queue, which pop_next_substep tolerates (it advances the clock to the popped tag). This is
            the streaming behaviour: an unreachable waypoint is simply left uncommanded, and the arm
            bridges to the next reachable one.
          * skip_unreachable=False (manual validate): abort the whole chunk with a reason (nothing
            partial staged), so a manual step-through never silently omits a row."""
        configs, kept_ids = [], []
        reason = None
        seed = np.asarray(seed_q, dtype=np.float64).copy()   # 14-vec [left7, right7]
        seedL, seedR = seed[:7].copy(), seed[7:14].copy()
        qprevL = qprevR = None    # per-arm quat-smoothing state (don't disturb the preview's)
        for k, action in enumerate(action_chunk):
            Lpos, Lquat, Lgrip, Rpos, Rquat, Rgrip = decode_action_row_dual(action)
            Lpos = np.asarray(Lpos, dtype=np.float64); Rpos = np.asarray(Rpos, dtype=np.float64)
            if not pos_in_workspace(Lpos, "left") or not pos_in_workspace(Rpos, "right"):
                reason = f"action {k}: target EE pos outside workspace envelope"
                if skip_unreachable:
                    continue
                return [], [], False, reason
            qprevL = smooth_quat_step(qprevL, Lquat, QUAT_ALPHA)                  # H3 (per arm)
            qprevR = smooth_quat_step(qprevR, Rquat, QUAT_ALPHA)
            # live/chained seed first, nominal-training-posture fallback on a contorted/unreachable
            # solve (per arm) — keeps a warm arm's natural branch, un-contorts a parked one.
            qL, okL = self._ik_robust(self.solver, Lpos, qprevL, seedL, self.nominal_q14[:7])
            qR, okR = self._ik_robust(self.solver_r, Rpos, qprevR, seedR, self.nominal_q14[7:])
            if not (okL and okR):
                bad = "LEFT" if not okL else "RIGHT"
                err = (self.solver if not okL else self.solver_r).last_pos_err * 1000
                reason = f"action {k}: {bad} IK unreachable (pos err {err:.0f} mm)"
                if skip_unreachable:
                    continue                               # drop this row; the arm bridges past it
                return [], [], False, reason               # manual: abort the whole chunk
            configs.append((np.concatenate([qL, qR]), [Lgrip, Rgrip]))
            kept_ids.append(k)
            seedL, seedR = qL, qR
        if configs:
            return configs, kept_ids, True, None
        return [], [], False, reason or "empty action chunk"

    def validate_chunk(self, sim, configs, seed_q, substeps_per_row, fast=False, seed_gap=False):
        """Run PRE-SOLVED joint configs through `sim` from seed_q (substep + self-collision + joint
        readback) and return (traj, ok, reason, rows): traj is the sim-ACHIEVED [(q14, grip), ...] and
        rows[i] is the config-row index traj[i] realizes (for exact master-id tagging). The single
        validation primitive shared by the manual validate_and_stage and auto-inference paths.
        Assumes `sim` is not None and `configs` non-empty (the env wrapper checks both).
        fast=True (auto path): skip the per-substep real-time sleep (operator-preview only).
        substeps_per_row: the SAME K the caller uses for master-id tagging, so a live speed_scale change
        between expansion and tagging can't desync them.
        seed_gap=True (streaming append): expand the seed->row0 gap too, so traj is continuous with the
        queue tail and every row (row 0 included) contributes exactly K substeps."""
        return sim.validate(configs, seed_q, MAX_JOINT_STEP, fast=fast,
                            substeps_per_row=substeps_per_row, seed_gap=seed_gap)

    # ------------------------------------------------------------------ #
    #  Stage 5: master-id tagging + seam ramp + subdivide + queue splice    #
    #  (the robot queue is the pipeline OUTPUT; the env release loop is the #
    #   consumer that drains it one substep per tick).                      #
    # ------------------------------------------------------------------ #
    def _throttled(self, key, interval=1.0):
        """True at most once per `interval` seconds for `key` (rate-limits hot-loop prints)."""
        now = time.monotonic()
        if now - self._log_ts.get(key, 0.0) >= interval:
            self._log_ts[key] = now
            return True
        return False

    @staticmethod
    def _ramp(q_from, q_to, cap, grip):
        """Joint-space ramp from q_from to q_to with every step <= cap (rad). Includes q_to, excludes
        q_from. Used so the first released command starts at the current pose (C5)."""
        q_from = np.asarray(q_from, dtype=np.float64)
        q_to = np.asarray(q_to, dtype=np.float64)
        n = int(np.ceil(np.max(np.abs(q_to - q_from)) / cap))
        n = max(n, 1)
        # grip is carried opaquely (a [gl, gr] pair for dual-arm release, or None); do NOT coerce.
        return [(q_from + (q_to - q_from) * (i / n), grip) for i in range(1, n + 1)]

    @staticmethod
    def _bridge(q_from, q_to, v_from, v_to, fallback_step, cap):
        """Velocity-merged seam bridge: a LINEAR joint-space ramp q_from -> q_to (excl. q_from, incl.
        q_to) whose per-substep speed is the MEAN of the two rows it links — v_from (how the previous
        row is moving as it ends) and v_to (how the next row starts). The step count is
        span / merged-speed, so the bridge cruises at the SAME speed as the motion on either side
        instead of darting at a fixed cap: the seam gains no velocity spike.

        v_from / v_to are per-substep joint deltas (= consecutive waypoints q[i]-q[i-1]); the release
        loop drains one point per STEP_TIME, so a per-substep delta IS the joint velocity in these
        units. Pass None for a side with no motion context (queue < 2 pts, or a single-row slice); the
        mean is taken over whatever is moving, and `fallback_step` is used only when NEITHER side moves
        (a hold-to-hold jump). `cap` (MAX_JOINT_STEP) is a SAFETY ceiling — it can only ADD substeps,
        never speed the bridge up. Returns [q_to] when already there (n collapses to 1)."""
        q_from = np.asarray(q_from, dtype=np.float64)
        q_to = np.asarray(q_to, dtype=np.float64)
        span = float(np.max(np.abs(q_to - q_from)))
        speeds = []
        for v in (v_from, v_to):
            if v is not None:
                s = float(np.max(np.abs(v)))
                if s > 0.0:
                    speeds.append(s)
        cruise = min((sum(speeds) / len(speeds)) if speeds else fallback_step, cap)
        n = max(1, int(np.ceil(span / cruise)))
        return [q_from + (q_to - q_from) * (i / n) for i in range(1, n + 1)]

    @staticmethod
    def _subdivide_points(points):
        """Insert linearly-interpolated waypoints so every consecutive gap <= MAX_JOINT_STEP (C5
        velocity bound). grip carries from each segment's target point. A single point (or already-fine
        spacing) passes through unchanged."""
        pts = list(points)
        if len(pts) < 2:
            return pts
        out = [pts[0]]
        prev = np.asarray(pts[0][0], dtype=np.float64)
        for q, grip in pts[1:]:
            q = np.asarray(q, dtype=np.float64)
            d = q - prev
            n = max(int(np.ceil(np.max(np.abs(d)) / MAX_JOINT_STEP)), 1) if MAX_JOINT_STEP > 0 else 1
            for i in range(1, n + 1):
                out.append((prev + d * (i / n), grip))
            prev = q
        return out

    def queue_status(self):
        """(current_row_id, queued_through) read atomically under the lock. The inference controller
        uses this to split its smoothed buffer into the FROZEN region (id <= queued_through) and the
        MUTABLE region (id > queued_through), and to know the live clock for pruning."""
        with self._lock:
            return self._current_row_id, self._queued_through

    def robot_q_preview(self, n=10):
        """The next up-to-n commands queued for the robot (_robot_q) as (14-joint array, grip) pairs
        (copies), in execution order. grip may be None on a ramp/hold substep. Cheap snapshot."""
        with self._lock:
            items = list(self._robot_q)[:n]
        return [(np.asarray(sub[0], dtype=np.float64).copy(),
                 (None if sub[1] is None else [float(sub[1][0]), float(sub[1][1])])) for sub in items]

    def pop_next_substep(self):
        """Consumer API for the env's release loop: pop the head substep under the lock and advance the
        master row clock to the row it realizes (auto path tags a row id as the 3rd element; manual
        2-tuples leave the clock alone). Returns the substep (q14, grip[, row_id]) or None if empty."""
        with self._lock:
            sub = self._robot_q.popleft() if self._robot_q else None
            if sub is not None and len(sub) > 2 and sub[2] is not None:
                self._current_row_id = int(sub[2])
        return sub

    def clear_queue(self):
        """Drop all pending robot commands and re-anchor the streaming cursor (queue emptied -> next
        append re-anchors to the clock). Returns the count dropped. E-stop path; the env clears its own
        manual staging buffer separately."""
        with self._lock:
            n = len(self._robot_q)
            self._robot_q.clear()
            self._queued_through = -1
        return n

    def append_actions(self, action_chunk, obs_step_id, n_rows=None):
        """Streaming auto-inference entry point: keep n_rows policy rows queued ahead of the master
        clock, appending only the ids not yet queued. _queued_through tracks the highest master id in
        the queue, so the append window is just [_queued_through+1 .. clock+n_rows]; in the steady state
        the clock has advanced by one row since the last call and exactly ONE new row is added. Rows are
        only ever appended (never cleared) and the release loop only ever pops the head, so the arm runs
        one master id at a time, in order. Substeps are tagged with their absolute master id. Refused
        while E-stopped / without a sim. ok=True with a no-op reason when the buffer is already full or
        the chunk doesn't cover the next needed id."""
        if not self.real:
            return False, "env built with real=False"
        sim = self.get_sim()
        if sim is None:
            return False, "no sim running — launch the preview first"
        if self.estopped():
            return False, "E-stop latched"
        if n_rows is None:
            n_rows = self.append_ahead_rows                 # live-tunable commit-ahead distance
        with self._lock:
            cur = self._current_row_id
            start_id = max(self._queued_through + 1, cur)   # first master id not yet queued
            # Continuity seed: continue from the queue TAIL (where these rows attach), not the arm's
            # transient live pose. None -> queue empty, read the live pose below.
            seed = (np.asarray(self._robot_q[-1][0], dtype=np.float64).copy()
                    if self._robot_q else None)
        last_id = cur + n_rows                              # keep n_rows rows queued ahead of clock
        if start_id > last_id:
            return True, f"buffer full (queued through {start_id - 1}, clock {cur})"
        # Slice THIS chunk to the rows covering [start_id, last_id]. A chunk that starts AFTER start_id
        # can't fill the next needed id without leaving a hole in the id stream -> skip it.
        lo = start_id - int(obs_step_id)
        if lo < 0:
            return True, f"chunk starts at {obs_step_id}, past next needed id {start_id} — skipped"
        hi = min(last_id - int(obs_step_id), len(action_chunk) - 1)
        if hi < lo:
            return True, "chunk does not reach the append window"

        if seed is None:                                    # queue empty -> continue from live pose
            arm14 = self.read_arm14()
            seed = arm14 if arm14 is not None else self._last_q14   # 14-vec dual seed
        # STRICT (skip_unreachable=False): an unreachable row aborts the whole chunk (arm HOLDS)
        # rather than being dropped and bridged over -- dropping rows mid-stream makes the arm skip
        # waypoints and jump across the gap, which reads as jerky/fast motion. (Bisect candidate.)
        configs, kept_ids, ok, reason = self.solve_chunk_ik(action_chunk[lo:hi + 1], seed)
        if not ok:
            return False, reason
        K = self.substeps_per_row                       # capture ONCE: sim expansion + tagging share it
        # seed_gap=True: the sim also expands the seed(=queue tail)->row0 gap, so traj is CONTINUOUS
        # with the tail and every row — row 0 included — contributes its substeps.
        traj, ok, reason, rows = self.validate_chunk(sim, configs, seed, K, fast=True, seed_gap=True)
        if not ok or not traj:
            return False, reason or "empty validated trajectory"
        # Tag each substep with its absolute master id from the sim's EXACT per-substep row index
        # (rows[s] = the CONFIG index it realizes) mapped through kept_ids to the ORIGINAL chunk row:
        # config j -> master id start_id + kept_ids[j]. This keeps ids correct when solve_chunk_ik
        # skipped an unreachable row (that id is simply absent — a gap the release loop tolerates), and
        # stays exact even if a velocity-capped row emits more than K substeps.
        tagged = [(np.asarray(q14, dtype=np.float64), grip,     # q14 = both arms; grip = [gl, gr]
                   start_id + int(kept_ids[rows[s]]))
                  for s, (q14, grip) in enumerate(traj)]
        with self._lock:
            cur = self._current_row_id
            # Validation took time; drop any row the clock has since passed or that's already queued.
            new = [t for t in tagged if t[2] >= cur and t[2] > self._queued_through]
            if not new:
                return True, f"fell behind during validation (clock now {cur})"
            if new[0][2] == start_id:
                # CONTIGUOUS (steady state): the clock stayed at/behind our start_id, so no leading
                # substeps were dropped and — because seed_gap expanded seed(=queue tail)->row0 — the
                # validated traj already flows continuously from the tail. Append it as-is; no bridge.
                self._robot_q.extend(new)
                n_bridge = 0
            else:
                # CATCH-UP: the clock advanced PAST start_id during validation, so new[0] jumped several
                # rows ahead of the tail. Bridge that gap velocity-merged (mean of the tail's exit
                # velocity v_in and the new rows' entry velocity v_out) so the catch-up isn't a jump.
                # v_in/v_out are None when there's no context (queue < 2 pts, or a single-row slice).
                seam_from = (np.asarray(self._robot_q[-1][0], dtype=np.float64)
                             if self._robot_q else seed)
                fq, fg, fid = new[0]
                v_in = (np.asarray(self._robot_q[-1][0], dtype=np.float64) -
                        np.asarray(self._robot_q[-2][0], dtype=np.float64)) \
                    if len(self._robot_q) >= 2 else None
                v_out = (np.asarray(new[1][0], dtype=np.float64) - np.asarray(fq, dtype=np.float64)) \
                    if len(new) >= 2 else None
                bridge = self._bridge(seam_from, fq, v_in, v_out, self.ramp_joint_step, MAX_JOINT_STEP)
                self._robot_q.extend([(q, fg, fid) for q in bridge] + new[1:])  # bridge ends AT new[0]
                n_bridge = len(bridge)
            self._queued_through = new[-1][2]
            qlen = len(self._robot_q)
            self._last_q14 = np.asarray(traj[-1][0], dtype=np.float64).copy()   # dual IK warm-start
        if n_bridge > 0 or self._throttled("append", 1.0):      # always surface a real catch-up bridge
            print(f"[pipeline] append: obs_row={obs_step_id}, clock={cur}, "
                  f"appended ids {new[0][2]}..{new[-1][2]} (+{n_bridge} catch-up) -> queue {qlen}")
        return True, ""

    def auto_ingest_chunk(self, action_chunk, obs_step_id):
        """Auto-inference entry point (queue-REPLACE variant): validate a predicted chunk on the
        preview sim and, on success, REPLACE the live robot queue with a fresh ramp-in from the arm's
        current pose to the chunk's still-future rows, ALIGNED by the master row id `obs_step_id`. Row j
        targets master id obs_step_id + j; we execute only the rows the arm has NOT yet passed
        (id >= current row id). Refused while E-stopped / without a sim. Returns (ok, reason)."""
        if not self.real:
            return False, "env built with real=False"
        sim = self.get_sim()
        if sim is None:
            return False, "no sim running — launch the preview first"
        if self.estopped():
            return False, "E-stop latched"
        # Seed IK + validation from a FRESH real left-arm read so the self-collision check and IK plan
        # from the arm's ACTUAL pose (what the policy observed), not a stale planned config.
        arm14 = self.read_arm14()                         # read outside the lock (DDS I/O)
        seed = arm14 if arm14 is not None else self._last_q14      # 14-vec dual seed
        # STRICT: abort-and-hold on an unreachable row (see the append-ahead path above); no mid-
        # stream row-dropping that would make the arm bridge across a gap. (Bisect candidate.)
        configs, kept_ids, ok, reason = self.solve_chunk_ik(action_chunk, seed)
        if not ok:
            return False, reason
        K = self.substeps_per_row                       # capture ONCE: sim expansion + tagging share it
        traj, ok, reason, rows = self.validate_chunk(sim, configs, seed, K, fast=True)
        if not ok or not traj:
            return False, reason or "empty validated trajectory"
        # Tag each validated substep with its ABSOLUTE master row id from the sim's EXACT per-substep
        # row index (rows[s] = the CONFIG index it realizes) mapped through kept_ids to the original
        # chunk row: config j -> master id obs_step_id + kept_ids[j] (a skipped-unreachable row is an
        # absent id). No seed_gap here (queue-REPLACE ramps from the live pose), so the release ramp
        # bridges live->first row below.
        tagged = [(np.asarray(q14, dtype=np.float64), grip,
                   int(obs_step_id) + int(kept_ids[rows[s]]))
                  for s, (q14, grip) in enumerate(traj)]
        with self._lock:
            cur = self._current_row_id
            kept = [t for t in tagged if t[2] >= cur]      # drop rows the arm already passed
            if not kept:
                return False, (f"chunk fully elapsed (obs_row {obs_step_id}, now {cur})")
            # Ramp from the live pose to the first future row, then follow the rest. Read the start pose
            # FRESH inside the lock: validation took time during which the release loop drained substeps,
            # so the pre-validation arm14 is stale; holding the lock keeps the release loop from
            # advancing between this read and the queue swap.
            live = self.read_arm14()
            start = live if live is not None else kept[0][0]   # 14-vec both-arm start pose
            fq, fg, fid = kept[0]
            # Ramp in from the live pose at the speed the chunk STARTS at (v_out); the queue is being
            # replaced, so there's no previous-row velocity -> v_in is None and cruise = the entry speed.
            v_out = (np.asarray(kept[1][0], dtype=np.float64) - np.asarray(fq, dtype=np.float64)) \
                if len(kept) >= 2 else None
            bridge = self._bridge(start, fq, None, v_out, self.ramp_joint_step, MAX_JOINT_STEP)
            ramp = [(q, fg, fid) for q in bridge]
            self._robot_q.clear()
            self._robot_q.extend(ramp + kept[1:])          # bridge ends AT kept[0]
            qlen = len(self._robot_q)
            self._last_q14 = np.asarray(traj[-1][0], dtype=np.float64).copy()   # dual IK warm-start
        if self._throttled("auto_ramp", 1.0):
            print(f"[pipeline] auto-ramp: obs_row={obs_step_id}, now={cur}, "
                  f"kept={len(kept)}/{len(tagged)} (+{len(ramp)} ramp) -> queue {qlen}")
        return True, ""
