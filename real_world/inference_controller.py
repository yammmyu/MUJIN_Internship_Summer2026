"""Standalone inference controller for the left_arm_ee_image policy.

Was gui/inference.py's InferenceMixin; converted to a plain class with explicit
dependencies so it has no GUI coupling and can be driven by either the GUI or a
headless runner:

    ctl = InferenceController(env, robot_info)
    ctl.start()                 # env lifecycle is owned by the caller (GUI)
    ctl.auto_inference()        # predict + submit in a loop
    ...
    ctl.auto_inference(stop=True)

The HumanoidEnv (injected) owns the collection + execution threads and the SDK
resources; this controller only calls the policy server, publishes predictions
to robot_info (for robot_info_server / visualisation), and hands action chunks
to the env.


"""

import copy
import http.client
import json
import logging
import os
import threading
import time

import numpy as np

from collections import deque

from real_world.humanoid_env import GRIPPER_CLOSE_THRESH
from real_world.timing import RECORD_HZ   # single source of timing truth (see timing.py)
# Client-side inference preprocessing: image crop/resize/encode (MUST stay identical to training
# build_dataset.py and the server decode) plus the obs -> /predict request assembly. encode_image
# is re-exported here so callers doing `from real_world.inference_controller import encode_image`
# keep working.
from real_world.build_data import encode_image, build_predict_request

log = logging.getLogger(__name__)

# Per-inference JSONL traces (requests/chunks/buffer) are debug aids only and sit on the
# real-time inference thread, so they are OFF by default and gated behind this flag. The
# buffer dump in particular re-serialises the whole smoothed run every cycle; enable only
# when diagnosing offline. Set HUMANOID_INFER_TRACE=1 to turn on.
TRACE_JSONL = os.environ.get("HUMANOID_INFER_TRACE", "") not in ("", "0", "false", "False")

INFERENCE_HZ = 0  # TUNE (Hz): auto-inference cadence cap. <=0 -> run back-to-back = MAX chunk
                  # overlap, so TE has the most chunks to average (smoother). A positive cap slows
                  # the loop (less overlap, less smoothing) but cuts policy-server load.

# --- Temporal ensemble (ACT-style) -------------------------------------------------
# Chunk-level EE-space smoothing: each new chunk is averaged against recent overlapping chunks
# before validation. Every action row carries an absolute MASTER ROW ID (the robot's own execution
# clock — see real_world/timing.py / HumanoidEnv): row k of a chunk observed at master id S is at
# id S + k. For each row of the newest chunk we gather every buffered chunk's row with the SAME
# absolute id (exact integer match) and take a recency-weighted mean (weight exp(-TE_M * age), age
# = inferences old, newest = 0). This only smooths once chunks actually OVERLAP in id space (when
# horizon > inferences-worth-of-rows); below that it passes the newest chunk through (identity).
# Aligning by master id (not wall-clock) tracks the arm's real progress, so the merge can't run
# ahead of the arm when execution lags real-time (latency / slow-down / control-loop jitter).
# ┌──────────────────────────────────────────────────────────────────────────────────────────┐
# │ PERSISTED / LIVE-TUNABLE (tuning_config.json): the four constants below — TE_M, TE_BUFFER_LEN,│
# │ TE_RADIUS, TE_SIGMA — are only DEFAULT SEEDS. The GUI restores the operator's saved values    │
# │ from tuning_config.json at startup and tunes them live (InferenceController.set_smoothing /    │
# │ set_buffer_len). Editing a literal here only changes the fallback used when there is no saved  │
# │ JSON entry (e.g. a headless run that never loads the file). It is NOT the live runtime value:  │
# │ that is inference.te_m / .te_buffer_len / .te_radius / .te_sigma.                              │
# └──────────────────────────────────────────────────────────────────────────────────────────┘
USE_TEMPORAL_ENSEMBLE = True   # TUNE: master on/off for ALL temporal-ensemble smoothing (bool).
TE_M = 0.02            # [persisted: tuning_config.json] TUNE [~0.005..0.5]: recency decay of the
                       # per-id cross-chunk mean. LARGER -> trust the newest chunk more (less
                       # averaging, more reactive, rougher); SMALLER -> heavier averaging (smoother).
TE_BUFFER_LEN = 8      # [persisted: tuning_config.json] TUNE [~2..16]: how many recent raw chunks are
                       # kept = max overlap depth to average over. More -> deeper averaging (smoother).
# Neighbor smoothing along the master-id axis (the second smoothing dimension). The smoothed buffer
# is low-passed by a symmetric Gaussian of half-width TE_RADIUS so the long sequence stays smooth
# id-to-id, not just averaged per id. TE_RADIUS doubles as how many already-committed ("frozen")
# rows we RETAIN past the clock as fixed left-context, so the filter window is full right at the
# seam to the rows already on the robot.
TE_RADIUS = 6          # [persisted: tuning_config.json] TUNE [~1..8]: Gaussian half-width (ids) == #
                       # of frozen rows retained for context. LARGER -> smoother id-to-id, but more
                       # lag and blurs sharp intended motion (and widens the frozen-context window).
TE_SIGMA = 1.5         # [persisted: tuning_config.json] TUNE [~0.5..TE_RADIUS]: Gaussian sigma (id
                       # units) = in-window shape. LARGER -> stronger smoothing. Keep <= TE_RADIUS or
                       # the kernel is clipped at the edges. (TE_GAUSS below is derived from this seed.)
_te_ks = np.arange(-TE_RADIUS, TE_RADIUS + 1)
TE_GAUSS = np.exp(-(_te_ks ** 2) / (2.0 * TE_SIGMA ** 2))
TE_GAUSS = TE_GAUSS / TE_GAUSS.sum()   # normalized symmetric kernel, length 2*TE_RADIUS+1
# Smoothness guard: warn when a per-row position step in the buffer exceeds this (m). The seam
# between chunks is where roughness shows up, so this catches a bad merge before it reaches the arm.
SMOOTHNESS_WARN_DPOS = 0.03    # TUNE (m): DIAGNOSTIC threshold only — logs a warning, never clamps.
                               # Lower to surface smaller seams; not a smoothing knob.
PC4080_HOST = "10.12.11.144"
PC4080_PORT = 9001


def post_predict(host: str, port: int, req: dict, timeout: float = 60.0) -> dict:
    """POST an obs dict to the policy server's /predict and return the JSON reply."""
    body = json.dumps(req).encode('utf-8')
    log.debug("POST http://%s:%s/predict (%d bytes, timeout=%ss)", host, port, len(body), timeout)
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    start = time.monotonic()
    try:
        conn.request(
            'POST', '/predict', body=body,
            headers={
                'Content-Type': 'application/json; charset=utf-8',
                'Content-Length': str(len(body)),
            })
        resp = conn.getresponse()
        resp_body = resp.read()
        log.debug("response: HTTP %d (%d bytes) in %.3fs",
                  resp.status, len(resp_body), time.monotonic() - start)
        try:
            resp_obj = json.loads(resp_body.decode('utf-8')) if resp_body else {}
        except Exception as e:
            log.warning("failed to parse JSON response: %r (status=%d)", e, resp.status)
            return {'error': f'failed to parse JSON response: {e!r} '
                             f'(status={resp.status})'}
        if resp.status != 200:
            log.warning("non-200 response: HTTP %d: %r", resp.status, resp_obj)
            if isinstance(resp_obj, dict) and 'error' in resp_obj:
                return resp_obj
            return {'error': f'HTTP {resp.status}: {resp_obj!r}'}
        return resp_obj
    except Exception as e:
        log.warning("request failed after %.3fs: %r", time.monotonic() - start, e)
        raise
    finally:
        conn.close()


class InferenceController:
    """Drives the left_arm_ee_image policy: server round-trip + env hand-off."""

    def __init__(self, env, robot_info=None,
                 host=PC4080_HOST, port=PC4080_PORT,
                 record_hz=RECORD_HZ, inference_hz=INFERENCE_HZ,
                 obs_source=None):
        self.humanoid_env = env                 # owned/started/stopped by the caller (GUI)
        self.robot_info = robot_info            # shared with robot_info_server (None = headless)
        # Where observations come from: the live env by default, or an injected source (e.g.
        # a RecordedObsSource) for the offline sim runner. Actions are always submitted to env.
        self.obs_source = obs_source if obs_source is not None else env
        self.host = host
        self.port = port
        self.record_hz = record_hz
        self.inference_hz = inference_hz

        self.inference_thread = None
        self.is_auto_inference = False
        self._last_inference_obs_ts = 0.0
        # Temporal-ensemble state: rolling buffer of recent RAW chunks (ts_obs, action[N,10]).
        # Always averaged from raw — never re-buffer ensembled output, or smoothing compounds.
        self.use_temporal_ensemble = USE_TEMPORAL_ENSEMBLE
        self._te_buffer = deque(maxlen=TE_BUFFER_LEN)
        # LIVE-TUNABLE smoothing config, held as ONE immutable snapshot (radius, sigma, m, kernel) so
        # the inference thread reads a consistent set even while the GUI changes it (assignment of the
        # dict reference is atomic under the GIL). Scalar mirrors (te_radius/te_sigma/te_m) are for
        # display/readback only. Use set_smoothing() to change at runtime.
        self.te_radius = TE_RADIUS
        self.te_sigma = TE_SIGMA
        self.te_m = TE_M
        self._smooth = self._make_smoothing(TE_RADIUS, TE_SIGMA, TE_M)
        # The long smoothed buffer fed to the robot: master_id -> EE row [pos3, rot6d6, grip1].
        # Ids <= the env's queued_through are FROZEN (committed, read-only context); ids above it are
        # MUTABLE and rebuilt every inference. Only the inference thread touches this -> no lock.
        self._buffer = {}
        # Last queued_through seen from the env. queued_through only climbs in normal streaming; a
        # DROP (env re-anchored: E-stop lock_robot resets it to -1) means the live queue was cleared,
        # so our buffer is stale and must be cleared too before rebuilding.
        self._last_queued_through = -1
        # Manual step-through cursor: index of the NEXT unexecuted action row in the
        # current chunk. Reset to 0 on every fresh inference; advanced by
        # execute_inference_result. cursor >= chunk length => the chunk is fully consumed
        # and the manual validate buttons become no-ops (see steps_remaining).
        self._exec_cursor = 0
        # Open trace-file handles, kept open for the controller's life so the hot loop never
        # re-opens per inference. Empty (and every _trace call a no-op) unless TRACE_JSONL is set.
        self._trace_files = {}
        if TRACE_JSONL:
            self._trace_files = {name: open(f"{name}.jsonl", "a")
                                 for name in ("requests", "chunks", "buffer")}

    def _trace(self, name, obj):
        """Append one JSON line to the named trace file (no-op unless TRACE_JSONL is set)."""
        f = self._trace_files.get(name)
        if f is not None:
            f.write(json.dumps(obj) + "\n")

    # ------------------------------------------------------------------ #
    #  Lifecycle: the env is owned by the caller; we only reset our state #
    # ------------------------------------------------------------------ #
    def start(self):
        """Prepare for inference. The injected env's threads are started by the
        caller (the GUI starts env before this), so we only reset our cursor."""
        self._last_inference_obs_ts = 0.0
        self._te_buffer.clear()
        self._buffer.clear()
        self._last_queued_through = -1
        log.info("InferenceController ready (env owned by caller).")

    def stop(self):
        """Stop auto inference. Does NOT tear down the env (caller owns it)."""
        self.auto_inference(stop=True)
        for f in self._trace_files.values():               # persist any buffered trace lines
            f.flush()

    # ------------------------------------------------------------------ #
    #  One inference: predict -> publish to robot_info (-> optional submit) #
    # ------------------------------------------------------------------ #
    def _run_inference(self, submit: bool) -> bool:
        """Pull one obs from the env, run the policy server, publish the result.

        submit=True also hands the predicted chunk to the env's execution queue.
        Returns True if a fresh inference was produced.
        """
        env = self.humanoid_env
        if env is None:
            log.warning("humanoid_env not accessible")
            return False

        obs = self.obs_source.get_obs()
        if obs is None:
            return False
        # Refuse to predict from stale/frozen sensors or under a firmware error (H2). A frozen
        # camera/EE feed still has an advancing wall-clock timestamp, so the timestamp check
        # below is NOT sufficient — `stale` reflects whether the data actually changed.
        if obs.get('stale'):
            log.warning("ABORT: observations are stale (age=%.2fs, firmware_error=%s). "
                        "Not predicting from frozen sensor data.",
                        obs.get('age'), obs.get('firmware_error'))
            return False
        # Skip if we've already inferred on this observation (env publishes a
        # wall-clock timestamp with every snapshot). ts is used ONLY for dedup + logs/staleness;
        # alignment uses sid (the master row id the obs was anchored to).
        ts = obs['timestamp']
        sid = obs['step_id']
        if ts - self._last_inference_obs_ts < 1e-4:
            log.debug("timestamp has not advanced")
            return False

        req = build_predict_request(obs)

        # Log the proprioception sent to the policy server, keyed by the same obs_ts that
        # chunks.jsonl/traj.jsonl use, so a request can be joined to its resulting action.
        # Images (agentview/eye_in_hand) are omitted — large base64 JPEGs, not needed here.
        # GUARDED: build the dict only when tracing is on — it reads request fields, and building
        # it unconditionally (with stale keys) would raise on every inference. Logs BOTH arms' EE.
        if self._trace_files:
            self._trace("requests", {"obs_ts": sid,
                                     "robotl_eef_pos": req.get("robotl_eef_pos"),
                                     "robotr_eef_pos": req.get("robotr_eef_pos"),
                                     "robot0_grip": req.get("robot0_grip")})

        resp = post_predict(self.host, self.port, req, timeout=10)
        if 'error' in resp:
            log.warning("server error: %s", resp["error"])
            return False

        # The dual_arm policy returns 20-col rows: L[pos(3)+rot6d(6)+grip(1)] ++ R[pos(3)+rot6d(6)+
        # grip(1)]. We now drive BOTH arms, so the full 20-col row is carried through the temporal
        # ensemble, IK (env solves left+right), sim validation, and release. (10-col left-only rows
        # still pass through unchanged for back-compat with a single-arm policy.)
        action = np.asarray(resp['action'], dtype=np.float32)
        log.debug("response received in %.3fs", time.time() - ts)
        # Binarize the gripper column(s) HERE, as soon as the chunk arrives: the raw [0,~85] gripper
        # signal is noisy (transient spikes), so only a (near-)fully-closed reading (>=
        # GRIPPER_CLOSE_THRESH) becomes closed=1, else open=0. col 9 = left grip; col 19 = right grip
        # (dual). Everything downstream — sim preview, validation, staging, release — then carries a
        # clean {0,1} per arm, and the spikes no longer pollute the next inference's state context.
        if action.ndim == 2:
            for gc in (9, 19):
                if action.shape[1] > gc:
                    action[:, gc] = (action[:, gc] >= GRIPPER_CLOSE_THRESH).astype(action.dtype)
        self._last_inference_obs_ts = ts
        # A fresh chunk restarts the manual step-through from the first row.
        self._exec_cursor = 0

        # Log the RAW chunk (pre-smoothing), keyed by the master id it's anchored to.
        if self._trace_files:
            self._trace("chunks", {"obs_ts": sid, "action": action.tolist()})

        # Buffer + smoothing (AUTO/streaming only): append this raw chunk, then rebuild the long
        # smoothed master buffer over its MUTABLE tail (id > queued_through). The BUFFER — not this
        # chunk — is what feeds the robot, so we maintain one continuous, id-to-id smooth sequence
        # instead of re-emitting per-inference chunks. queue_status gives the live clock + commit
        # frontier for the split. Manual one-shots (submit=False) bypass this: they feed the raw
        # chunk directly and never touch the streaming buffer (which would otherwise grow unbounded
        # while queued_through sits at -1, and pollute the chunk that manual step-through reads).
        cur, queued_through = env.queue_status()
        use_buffer = submit and self.use_temporal_ensemble and action.ndim == 2 and action.shape[1] >= 10
        if use_buffer:
            if queued_through < self._last_queued_through:     # env re-anchored (E-stop) -> stale
                self._buffer.clear()
                self._te_buffer.clear()
            self._last_queued_through = queued_through
            self._te_buffer.append((sid, action.copy()))      # raw chunk, keyed on master row id
            self._rebuild_buffer(queued_through)
            base_id, buf = self._materialize()
        else:                                                 # manual / TE off -> raw chunk direct
            base_id, buf = sid, action

        # Smoothness guard: the buffer feeds the robot, so any id-to-id jump becomes a fast seam.
        # Measure the executed sequence's roughness in EE space over the STILL-MUTABLE rows (the part
        # we can still influence; frozen rows are committed). |Δpos| is the per-row position step and
        # |Δ²pos| the bend (jerk proxy); warn when a step is unusually large so a bad merge surfaces
        # in the logs. Logged, not clamped — clamping committed EE would fight the env seam ramp.
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

        # The robot-facing buffer as a plain list, computed ONCE and reused by the trace, the
        # robot_info publish, and append_actions (buf.tolist() is the single biggest per-cycle
        # allocation, so it must not be repeated).
        buf_list = buf.tolist()

        # Log the smoothed buffer (contiguous run actually fed to the robot) for offline inspection.
        self._trace("buffer", {"obs_ts": sid, "base_id": int(base_id),
                               "clock": int(cur), "queued_through": int(queued_through),
                               "jerk": jerk, "buffer": buf_list})
        # Publish for robot_info_server / visualisation. left_*_predict_* carry EE-pose data here:
        #   *_start_values  = last two left EE obs rows ([pos(3), rot6d(6)], dual_arm_ee_image layout)
        #   *_action_values = the smoothed buffer rows
        # Skipped when robot_info is None (headless sim runner has no GUI to publish to).
        if self.robot_info is not None:
            with self.robot_info.lock:
                self.robot_info.left_joint_predict_start_values = copy.deepcopy(obs['robotl_eef_pos'])
                self.robot_info.left_joint_predict_action_values = buf_list
                self.robot_info.inference_timestamp = ts

        if submit and buf.size:
            # AUTO-to-robot (streaming): append_actions reads the live clock + queued_through itself
            # and pulls the window [queued_through+1 .. clock+n] from this contiguous buffer (base_id
            # = its first master id), validating + appending only the not-yet-queued ids. Rows are
            # only ever appended (never cleared); the release loop drains one master id at a time.
            ok, reason = env.append_actions(buf_list, int(base_id))
            if not ok:
                log.info("auto: append skipped — %s", reason)
        return True

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
        is at id S + j — the robot's own execution clock; see real_world/timing.py / HumanoidEnv):

          (a) ACROSS CHUNKS (per id): for each mutable id, gather every buffered chunk's row at the
              SAME absolute id and take a recency-weighted mean (w = exp(-TE_M * age), age = how many
              inferences old, newest = 0). Exact integer alignment, no rounding — tracks the arm's
              real progress regardless of inference latency. This is the old ACT temporal ensemble.
          (b) ALONG THE ID AXIS (neighbors): low-pass the per-id estimate with a symmetric Gaussian
              of half-width TE_RADIUS so the sequence is smooth id-to-id, not just averaged per id.

        The buffer is split by the env's commit frontier:
          * FROZEN  (id <= queued_through): already committed to the robot. Kept (the last TE_RADIUS
            of them) as READ-ONLY left-context for (b) so the seam to the rows already on the arm
            stays continuous; never recomputed or rewritten.
          * MUTABLE (id >  queued_through): rebuilt here every inference from passes (a) then (b).

        pos (0:3) and rot6d (3:9) smooth linearly (rot6d is re-orthonormalised downstream by
        rot6d_to_quat); gripper (9), already {0,1}, is NOT low-passed along id (that would blur
        open/close timing) — it carries the cross-chunk value, re-thresholded at 0.5. Assumes the new
        raw chunk is already in self._te_buffer (deque(maxlen=TE_BUFFER_LEN), OLDEST -> NEWEST), each
        element a (master_id, raw chunk (N,10)) tuple; a chunk leaves only by eviction / clear()."""
        raw = list(self._te_buffer)                       # oldest -> newest
        if not raw:
            return
        s = self._smooth                                  # atomic snapshot (live-tunable)
        R, gauss, te_m = s["radius"], s["gauss"], s["m"]
        newest_idx = len(raw) - 1
        max_id = max(sid + a.shape[0] - 1 for sid, a in raw)
        min_sid = min(sid for sid, _ in raw)
        # First MUTABLE id (everything <= queued_through is frozen). Bounded below by the oldest
        # buffered chunk: no id below any chunk can have a cross-chunk estimate, so don't iterate
        # there (after an E-stop re-anchor queued_through+1 would otherwise be ~0 while max_id is the
        # live clock, spinning over thousands of empty ids).
        mut_lo = max(queued_through + 1, min_sid)
        n_ids = max_id - mut_lo + 1                        # mutable id i <-> master id (mut_lo + i)
        if n_ids <= 0:                                     # whole buffer already committed/frozen
            self._prune_frozen(queued_through, R)         # nothing mutable to rebuild; just prune
            return

        # Column layout: W = 10 (left-only) or 20 (dual L++R). Gripper cols {9, 19} are binary and
        # NOT low-passed along id (that would blur open/close timing); every other col (pos + rot6d,
        # per arm) is smoothed linearly. Derived from the row width so the same code handles both.
        W = raw[-1][1].shape[1]
        grip_cols = [c for c in (9, 19) if c < W]
        smooth_cols = [c for c in range(W) if c not in grip_cols]
        ns = len(smooth_cols)

        # (a) cross-chunk recency-weighted mean per mutable id (the ACT temporal ensemble).
        # Each chunk overlaps a CONTIGUOUS span of mutable ids, so it scatter-adds in one vectorised
        # slice (w*rows) instead of an id-by-id Python loop; per-id weight = sum of contributors.
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
        tent = acc / np.where(present, wsum, 1.0)[:, None]  # normalized weighted mean (== old w/w.sum)

        # (b) symmetric Gaussian along id over the SMOOTH (pos+rot6d, both arms) columns; frozen rows
        # (<= queued_through) are fixed left-context. Read from a dense padded context array (2R wider
        # than the mutable span) so the convolution is 2R+1 vectorised shifted adds.
        ext = n_ids + 2 * R
        ctxv = np.zeros((ext, ns), dtype=np.float64)       # neighbor smooth-col values, (mut_lo-R)..(max_id+R)
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
        Returns (base_id, ndarray(M,10)); (-1, empty) when the buffer is empty. append_actions maps
        master id -> row by index = id - base_id, so the run handed to it MUST be contiguous."""
        if not self._buffer:
            return -1, np.empty((0, 10), dtype=np.float32)
        lo = min(self._buffer)
        rows = []
        i = lo
        while i in self._buffer:                           # walk forward, stop at first missing id
            rows.append(self._buffer[i])
            i += 1
        return lo, np.asarray(rows, dtype=np.float32)

    def inference_once(self) -> bool:
        """Predict + publish only (no execution).

        Manual mode: call this to predict, then execute_inference_result to run.
        Warms the policy cameras first (they may be idle-evicted), since a single
        shot can't rely on the retry loop that auto_inference has.
        """
        env = self.humanoid_env
        if env is None:
            log.warning("humanoid_env not accessible")
            return False
        deadline = time.monotonic() + 0.3
        while not self.obs_source.inf_ready and time.monotonic() < deadline:
            self.obs_source.get_obs()   # requests head + hand_left, warming them (no-op for recorded)
            time.sleep(0.02)
        return self._run_inference(submit=False)

    # ------------------------------------------------------------------ #
    #  Manual: validate the last prediction IN THE SIM, then stage it      #
    # ------------------------------------------------------------------ #
    def _current_actions(self):
        """The last published action chunk (rows), or [] if none / no robot_info."""
        if self.robot_info is None:
            return []
        with self.robot_info.lock:
            return copy.deepcopy(self.robot_info.left_joint_predict_action_values) or []

    def steps_remaining(self):
        """How many action rows of the current chunk are still unexecuted (manual mode).

        0 means the chunk is exhausted (or there's none): the manual validate buttons
        should be no-ops until the next 推理一次.
        """
        return max(0, len(self._current_actions()) - self._exec_cursor)

    def execute_inference_result(self, once: bool = False):
        """Step through the last published chunk in the SIM (step + self-collision + readback)
        and stage the sim-validated trajectory for release. Manual "执行" path; never touches
        the robot — that needs a subsequent release_to_robot().

        A cursor tracks the next unexecuted row of the current chunk:
          * once=True  -> validate+stage just the NEXT row, advance the cursor by one.
          * once=False -> validate+stage all REMAINING rows, advance the cursor to the end.
        Once the cursor reaches the end the chunk is consumed and both modes are no-ops
        until a fresh 推理一次. Returns (ok, reason). The cursor only advances on success.
        """
        env = self.humanoid_env
        if env is None or self.robot_info is None:
            return False, "no env / robot_info"
        actions = self._current_actions()
        if not actions:
            return False, "no prediction yet (run 推理一次 first)"
        cursor = self._exec_cursor
        n = len(actions)
        if cursor >= n:                     # chunk already fully executed
            return False, "本次推理已全部执行，请先 推理一次"
        end = cursor + 1 if once else n
        ok, reason = env.validate_and_stage(actions[cursor:end])
        if ok:
            self._exec_cursor = end
        return ok, reason

    # ------------------------------------------------------------------ #
    #  Auto loop: predict + submit continuously                            #
    # ------------------------------------------------------------------ #
    def auto_inference(self, stop: bool = False):
        """Continuous auto-inference that DRIVES THE REAL ROBOT. Each inference is validated on the
        preview sim and its still-unqueued rows are appended to the live robot queue
        (env.append_actions), which streams them one master id at a time — no manual release. The
        inference cadence is set by the server+validation latency itself (the substep queue absorbs
        the gap), so there is no fixed inference_hz sleep. Requires the preview sim to be running
        (validation) and real=True; the GUI launches the sim before starting."""
        if stop:
            self.is_auto_inference = False
            if self.inference_thread is not None:
                log.info("Stopping auto_inference thread...")
                self.inference_thread.join()
                self.inference_thread = None
            # Stop feeding NEW chunks; let the release loop drain whatever is already queued. (Use
            # the env E-stop to halt immediately instead.)
            return

        env = self.humanoid_env
        if env is None or getattr(env, "sim", None) is None:
            log.warning("auto refused: launch the simulation preview first "
                        "(validation runs through it).")
            return

        self.is_auto_inference = True

        def _run_auto_inference():
            # Target a fixed inference cadence so the loop is controllable via inference_hz (a
            # value <= 0 means "as fast as latency allows"). We measure each cycle and sleep only
            # the REMAINDER of the period; if an inference already overran the period we go again
            # immediately (no sleep), so this caps the rate without ever slowing a slow cycle.
            period = (1.0 / self.inference_hz) if self.inference_hz and self.inference_hz > 0 else 0.0
            while self.is_auto_inference:
                cycle_start = time.monotonic()
                # Any E-stop source (operator 急停 OR a firmware-triggered lock_robot from
                # _release_loop) disarms auto: we exit the loop so motion does NOT silently
                # resume when the E-stop is later reset — the operator must press 启动 again.
                # (append_actions also refuses while latched, so nothing is queued even in
                # the brief window before we notice here.)
                if self.humanoid_env.estopped:
                    log.warning("E-stop latched — disarming auto-inference.")
                    self.is_auto_inference = False
                    break
                # submit=True -> validate + splice onto the robot. On a skipped inference (stale
                # obs, server error, validation failure) back off briefly so we don't busy-spin.
                if not self._run_inference(submit=True):
                    time.sleep(0.01)
                    continue
                # Pace to inference_hz: sleep whatever is left of the period after this cycle's
                # work. remaining <= 0 (cycle overran the period) -> no sleep, loop immediately.
                if period > 0.0:
                    remaining = period - (time.monotonic() - cycle_start)
                    if remaining > 0.0:
                        time.sleep(remaining)

        # Reap a thread that self-exited on E-stop (handle left non-None) so a restart is treated
        # as a fresh start instead of being blocked by the dead handle.
        if self.inference_thread is not None and not self.inference_thread.is_alive():
            self.inference_thread.join()
            self.inference_thread = None

        if self.inference_thread is None:
            self.inference_thread = threading.Thread(target=_run_auto_inference, daemon=True)
            log.info("Starting auto_inference (-> robot) thread...")
            self.inference_thread.start()
