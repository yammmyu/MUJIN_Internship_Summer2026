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

import base64
import copy
import http.client
import json
import threading
import time

import cv2
import numpy as np

from collections import deque

from real_world.humanoid_env import GRIPPER_CLOSE_THRESH
from real_world.timing import RECORD_HZ   # single source of timing truth (see timing.py)

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


# Client-side image preprocessing — MUST stay identical to MDM_data_collection/build_dataset.py
# (training) and the policy server's decode. Doing the crop+resize HERE, BEFORE the JPEG encode,
# shrinks the head frame from native (e.g. 1280x800) to IMG_W x IMG_H first, so imencode runs on
# ~25x fewer pixels (the old ~0.2s encode was dominated by JPEG-ing two full-res head frames).
# NOTE: with AGENT_CROP_ZOOM == 1.0 the server's own crop+resize become no-ops on an already-
# target-size frame (aspect matches, zoom 1, shape == target -> resize skipped), so NO server
# change is needed. If AGENT_CROP_ZOOM is ever raised > 1, the server MUST disable its center-crop
# or the head frame gets zoomed twice.
IMG_H = 180             # target rows  (build_dataset.IMG_H)
IMG_W = 240             # target cols  (build_dataset.IMG_W)
AGENT_CROP_ZOOM = 1.0   # head center-crop zoom (build_dataset.AGENT_CROP_ZOOM); keep in sync


def center_crop_to_aspect(frame, target_w, target_h, zoom=1.0):
    """Center-crop `frame` to the target aspect ratio (target_w/target_h), optionally zooming in.
    Identical to build_dataset.center_crop_to_aspect / the server's crop. Returns a cropped view."""
    h, w = frame.shape[:2]
    target_ar = target_w / target_h
    if w / h > target_ar:        # source too wide -> limited by height
        crop_h, crop_w = h, int(round(h * target_ar))
    else:                        # source too tall -> limited by width
        crop_w, crop_h = w, int(round(w / target_ar))
    crop_w = min(w, int(round(crop_w / zoom)))
    crop_h = min(h, int(round(crop_h / zoom)))
    x0, y0 = (w - crop_w) // 2, (h - crop_h) // 2
    return frame[y0:y0 + crop_h, x0:x0 + crop_w]


def encode_image(rgb_image, center_crop=False):
    """RGB frame -> base64 JPEG string (matches the training/recording encode).

    Applies the SAME preprocessing as training/server BEFORE encoding: head (center_crop=True) is
    center-cropped to the target aspect (+AGENT_CROP_ZOOM); both cameras are then INTER_AREA-resized
    to (IMG_W, IMG_H). Encoding the small frame is what makes this fast."""
    if center_crop:
        rgb_image = center_crop_to_aspect(rgb_image, IMG_W, IMG_H, AGENT_CROP_ZOOM)
    if rgb_image.shape[:2] != (IMG_H, IMG_W):
        rgb_image = cv2.resize(rgb_image, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
    rgb_image_cv2 = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.jpg', rgb_image_cv2, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buffer).decode('utf-8')


def post_predict(host: str, port: int, req: dict, timeout: float = 60.0) -> dict:
    """POST an obs dict to the policy server's /predict and return the JSON reply."""
    print(f"inference request sent")
    body = json.dumps(req).encode('utf-8')
    print(f"POST http://{host}:{port}/predict ({len(body)} bytes, timeout={timeout}s)")
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
        elapsed = time.monotonic() - start
        print(f"response: HTTP {resp.status} ({len(resp_body)} bytes) in {elapsed:.3f}s")
        try:
            resp_obj = json.loads(resp_body.decode('utf-8')) if resp_body else {}
        except Exception as e:
            print(f"failed to parse JSON response: {e!r} (status={resp.status})")
            return {'error': f'failed to parse JSON response: {e!r} '
                             f'(status={resp.status})'}
        if resp.status != 200:
            print(f"non-200 response: HTTP {resp.status}: {resp_obj!r}")
            if isinstance(resp_obj, dict) and 'error' in resp_obj:
                return resp_obj
            return {'error': f'HTTP {resp.status}: {resp_obj!r}'}
        return resp_obj
    except Exception as e:
        print(f"request failed after {time.monotonic() - start:.3f}s: {e!r}")
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
        print("[InferenceController] InferenceController ready (env owned by caller).")

    def stop(self):
        """Stop auto inference. Does NOT tear down the env (caller owns it)."""
        self.auto_inference(stop=True)

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
            print("[InferenceController] humanoid_env not accessible")
            return False

        obs = self.obs_source.get_obs()
        print(f"[InferenceController] inference pipeline starting")
        if obs is None:
            return False
        # Refuse to predict from stale/frozen sensors or under a firmware error (H2). A frozen
        # camera/EE feed still has an advancing wall-clock timestamp, so the timestamp check
        # below is NOT sufficient — `stale` reflects whether the data actually changed.
        if obs.get('stale'):
            print(f"[InferenceController] ABORT: observations are stale "
                  f"(age={obs.get('age'):.2f}s, firmware_error={obs.get('firmware_error')}). "
                  f"Not predicting from frozen sensor data.")
            return False
        # Skip if we've already inferred on this observation (env publishes a
        # wall-clock timestamp with every snapshot). ts is used ONLY for dedup + logs/staleness;
        # alignment uses sid (the master row id the obs was anchored to).
        ts = obs['timestamp']
        sid = obs['step_id']
        if ts - self._last_inference_obs_ts < 1e-4:
            print("[InferenceController] timestamp has not advanced")
            return False

        print(f"[InferenceController] preping request | Time elapsed; {time.time()- ts}")
        req = {
            # head (agent) is center-cropped to the target aspect like training; hand is resize-only.
            'agent_imgs': [encode_image(img, center_crop=True) for img in obs['agent_imgs']],
            'hand_imgs': [encode_image(img, center_crop=False) for img in obs['hand_imgs']],
            'state': obs['state'],
            # robot0_left_joint policy input: [j_{t-1}, j_t], each = 7 left-arm joints (rad) + raw
            # gripper. Server must map this to robot0_left_joint (see train config shape_meta).
            'left_joint': obs['joint_state'],
        }

        # Log the proprioception sent to the policy server, keyed by the same obs_ts that
        # chunks.jsonl/traj.jsonl use, so a request can be joined to its resulting action.
        # Images (agent_imgs/hand_imgs) are omitted — large base64 JPEGs, not needed here.
        with open("requests.jsonl", "a") as f:
            f.write(json.dumps({"obs_ts": sid, 
                                "state": req['state'],
                                "left_joint": req['left_joint']}) + "\n")
        
        print(f"input ee_state{req['state']}")

        print(f"[InferenceController] sending request | Time elapsed; {time.time()- ts}")
        resp = post_predict(self.host, self.port, req, timeout=10)

        if 'error' in resp:
            print(f'[InferenceController] \nServer error: {resp["error"]}')
            return False

        # action rows: [eef_pos(3), 6D_rot(6), gripper(1)]
        action = np.asarray(resp['action'], dtype=np.float32)
        #print(f"[InferenceController] response recieved! | Time elapsed; {time.time()- ts} | Details:{action}")   # raw (incl. raw gripper)
        print(f"[InferenceController] response recieved! | Time elapsed; {time.time()- ts}")   # raw (incl. raw gripper)
        # Binarize the gripper column (idx 9) HERE, as soon as the chunk arrives: the raw [0,~85]
        # gripper signal is noisy (transient spikes), so only a (near-)fully-closed reading
        # (>= GRIPPER_CLOSE_THRESH) becomes closed=1, else open=0. Everything downstream — sim
        # preview, validation, staging, release — then carries a clean {0,1}, and the spikes no
        # longer pollute the next inference's state context.
        if action.ndim == 2 and action.shape[1] >= 10:
            action[:, 9] = (action[:, 9] >= GRIPPER_CLOSE_THRESH).astype(action.dtype)
        self._last_inference_obs_ts = ts
        # A fresh chunk restarts the manual step-through from the first row.
        self._exec_cursor = 0

        # Log the RAW chunk (pre-smoothing), keyed by the master id it's anchored to.
        with open("chunks.jsonl", "a") as f:
            f.write(json.dumps({"obs_ts": sid, "action": action.tolist()}) + "\n")

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
        print(f"[InferenceController] buffer rebuilt ({buf.shape[0]} rows) | Time elapsed; {time.time()- ts}")

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
                d2pos = np.linalg.norm(np.diff(seg[:, :3], n=2, axis=0), axis=1)
                jerk = float(d2pos.max())
                if dpos.max() > SMOOTHNESS_WARN_DPOS:
                    print(f"[InferenceController] SMOOTHNESS WARN: buffer |Δpos|max={dpos.max():.4f} "
                          f"|Δ²pos|max={jerk:.4f} m near seam (clock={cur}, qt={queued_through})")

        # Log the smoothed buffer (contiguous run actually fed to the robot) for offline inspection.
        with open("buffer.jsonl", "a") as f:
            f.write(json.dumps({"obs_ts": sid, "base_id": int(base_id),
                                "clock": int(cur), "queued_through": int(queued_through),
                                "jerk": jerk, "buffer": buf.tolist()}) + "\n")
        # Publish for robot_info_server / visualisation. left_*_predict_* carry EE-pose data here:
        #   *_start_values  = last two left EE states ([pos(3), quat(4), grip(1)])
        #   *_action_values = the smoothed buffer rows
        # Skipped when robot_info is None (headless sim runner has no GUI to publish to).
        if self.robot_info is not None:
            with self.robot_info.lock:
                self.robot_info.left_joint_predict_start_values = copy.deepcopy(obs['state'])
                self.robot_info.left_joint_predict_action_values = buf.tolist()
                self.robot_info.inference_timestamp = ts

        print(f"[InferenceController] feeding buffer to robot | Time elapsed; {time.time()- ts}")
        if submit and buf.size:
            # AUTO-to-robot (streaming): append_actions reads the live clock + queued_through itself
            # and pulls the window [queued_through+1 .. clock+n] from this contiguous buffer (base_id
            # = its first master id), validating + appending only the not-yet-queued ids. Rows are
            # only ever appended (never cleared); the release loop drains one master id at a time.
            ok, reason = env.append_actions(buf.tolist(), int(base_id))
            if not ok:
                print(f"[InferenceController] auto: append skipped — {reason}")
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
        print(f"[InferenceController] te_buffer_len set: {n}")
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
        print(f"[InferenceController] smoothing set: radius={self.te_radius} "
              f"sigma={self.te_sigma:.2f} m={self.te_m:.3f}")
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

        # (a) cross-chunk recency-weighted mean per mutable id (the ACT temporal ensemble).
        tentative = {}
        for tid in range(mut_lo, max_id + 1):
            rows, weights = [], []
            for idx, (sid, a) in enumerate(raw):
                j = tid - sid                             # exact integer alignment (no rounding)
                if 0 <= j < a.shape[0]:
                    rows.append(a[j])
                    weights.append(np.exp(-te_m * (newest_idx - idx)))   # newest chunk -> age 0
            if rows:
                w = np.asarray(weights, dtype=np.float64)
                w /= w.sum()
                tentative[tid] = (w[:, None] * np.asarray(rows, dtype=np.float64)).sum(axis=0)

        # (b) symmetric Gaussian along id over the mutable region; frozen rows (<= queued_through)
        # are fixed left-context. Mutable neighbors read the PRE-smoothing 'tentative' estimate so
        # this stays a plain symmetric filter (not a recursive/IIR one).
        for tid in range(mut_lo, max_id + 1):
            if tid not in tentative:
                continue
            acc = np.zeros(9, dtype=np.float64)
            wsum = 0.0
            for d in range(-R, R + 1):
                nid = tid + d
                ctx = self._buffer.get(nid) if nid <= queued_through else tentative.get(nid)
                if ctx is None:                           # past the buffer edges -> just skip
                    continue
                k = gauss[d + R]
                acc += k * np.asarray(ctx[:9], dtype=np.float64)
                wsum += k
            out = tentative[tid].copy()
            if wsum > 0.0:
                out[:9] = acc / wsum                      # pos + rot6d (linear) smoothed along id
            out[9] = 1.0 if out[9] >= 0.5 else 0.0        # gripper re-binarised (not low-passed)
            self._buffer[tid] = out                       # write MUTABLE ids only; frozen untouched

        # Prune frozen rows older than the retention window (keep the last TE_RADIUS for context).
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
            print("[InferenceController] humanoid_env not accessible")
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
                print("[InferenceController] Stopping auto_inference thread...")
                self.inference_thread.join()
                self.inference_thread = None
            # Stop feeding NEW chunks; let the release loop drain whatever is already queued. (Use
            # the env E-stop to halt immediately instead.)
            return

        env = self.humanoid_env
        if env is None or getattr(env, "sim", None) is None:
            print("[InferenceController] auto refused: launch the simulation preview first "
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
                    print("[InferenceController] E-stop latched — disarming auto-inference.")
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
            print("[InferenceController] Starting auto_inference (-> robot) thread...")
            self.inference_thread.start()
