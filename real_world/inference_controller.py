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

from real_world.timing import RECORD_HZ, TRACE_DIR   # single source of timing truth (see timing.py)
# Client-side inference preprocessing: image crop/resize/encode (MUST stay identical to training
# build_dataset.py and the server decode) plus the obs -> /predict request assembly. encode_image
# is re-exported here so callers doing `from real_world.inference_controller import encode_image`
# keep working.
from real_world.build_data import encode_image, build_predict_request

log = logging.getLogger(__name__)


def _hms(t):
    """Wall-clock `t` (time.time()) as HH:MM:SS.mmm for compact per-inference timeline logs."""
    return time.strftime("%H:%M:%S", time.localtime(t)) + f".{int((t % 1) * 1000):03d}"

# Per-inference JSONL traces (requests/chunks/buffer) are debug aids only and sit on the
# real-time inference thread, so they are OFF by default and gated behind this flag. The
# buffer dump in particular re-serialises the whole smoothed run every cycle; enable only
# when diagnosing offline. Set HUMANOID_INFER_TRACE=1 to turn on.
TRACE_JSONL = os.environ.get("HUMANOID_INFER_TRACE", "") not in ("", "0", "false", "False")

INFERENCE_HZ = 0  # TUNE (Hz): auto-inference cadence cap. <=0 -> run back-to-back = MAX chunk
                  # overlap, so TE has the most chunks to average (smoother). A positive cap slows
                  # the loop (less overlap, less smoothing) but cuts policy-server load.

# Temporal-ensemble chunk merging (binarize + cross-chunk recency mean + along-id Gaussian) now lives
# in real_world/postprocess.py (PostProcessor). The env owns one PostProcessor (env.pipeline); this
# controller hands each server chunk to pipeline.merge and exposes the live-tunable knobs (set_smoothing
# / set_buffer_len / te_radius / te_sigma / te_m / te_buffer_len) as delegators for the GUI. The four
# TE_* seeds and USE_TEMPORAL_ENSEMBLE / SMOOTHNESS_WARN_DPOS defaults live in postprocess.py.
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
                 obs_source=None, grasp_detector_path=None):
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
        self._infer_count = 0            # total inferences this process (per-inference trace id)
        # The temporal-ensemble merge state (rolling raw-chunk buffer + smoothed master buffer +
        # live-tunable smoothing config) lives in env.pipeline (a PostProcessor). This controller only
        # delegates the GUI-facing knobs to it (see set_smoothing / te_radius etc. below).
        # Manual step-through cursor: index of the NEXT unexecuted action row in the
        # current chunk. Reset to 0 on every fresh inference; advanced by
        # execute_inference_result. cursor >= chunk length => the chunk is fully consumed
        # and the manual validate buttons become no-ops (see steps_remaining).
        self._exec_cursor = 0
        # Open trace-file handles, kept open for the controller's life so the hot loop never
        # re-opens per inference. The emitted-buffer trace is ALWAYS on: the smoothed run fed to the
        # robot is logged to buffer.jsonl every inference so it can be joined (by master row id) to
        # the always-on released_substeps.jsonl / live_joints.jsonl recorders in the env release loop.
        # The verbose extras (raw request + raw chunk) stay gated behind TRACE_JSONL.
        TRACE_DIR.mkdir(parents=True, exist_ok=True)   # all traces land in one folder (see timing.py)
        self._trace_files = {"buffer": open(TRACE_DIR / "buffer.jsonl", "a")}
        if TRACE_JSONL:
            self._trace_files.update({name: open(TRACE_DIR / f"{name}.jsonl", "a")
                                      for name in ("requests", "chunks")})

        # CCDP grasp-failure recovery (opt-in). Built ONLY when a detector checkpoint is given (arg
        # or HUMANOID_GRASP_DETECTOR) and exists on disk; otherwise self.recovery stays None and the
        # three loop call-ins are no-ops. torch/torchvision are imported lazily inside the module so
        # the controller keeps no hard dependency on them when recovery is off.
        self.recovery = None
        det_path = grasp_detector_path or os.environ.get("HUMANOID_GRASP_DETECTOR", "")
        if det_path and os.path.exists(det_path):
            try:
                from real_world.grasp_recovery import GraspRecoveryMonitor
                from real_world.postprocess import GRIPPER_CLOSE_THRESH
                self.recovery = GraspRecoveryMonitor(det_path, closed_grip_min=GRIPPER_CLOSE_THRESH)
            except Exception as e:
                log.warning("grasp recovery disabled (failed to load %s): %r", det_path, e)
        elif det_path:
            log.warning("grasp recovery disabled: detector checkpoint not found at %s", det_path)

    def _trace(self, name, obj):
        """Append one JSON line to the named trace file (no-op if that file's handle isn't open:
        buffer is always open; requests/chunks only under TRACE_JSONL)."""
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
        if self.humanoid_env is not None and getattr(self.humanoid_env, "pipeline", None) is not None:
            self.humanoid_env.pipeline.reset_merge()
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
        t_start_wall = time.time()             # inference START (wall clock, for the per-cycle trace)
        t_start_mono = time.monotonic()
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

        # CCDP grasp-failure recovery (opt-in: inert unless self.recovery is set). If a grasp
        # check is due and the right hand closed on nothing, this clears the queue and enters
        # the streaming-retreat mode (pumped from the auto loop) — skip predicting this cycle.
        rec = getattr(self, "recovery", None)
        if rec is not None and rec.maybe_start(env, obs):
            return True

        req = build_predict_request(obs)

        # Log the proprioception sent to the policy server, keyed by the same obs_ts that
        # chunks.jsonl/traj.jsonl use, so a request can be joined to its resulting action.
        # Images (agentview/eye_in_hand) are omitted — large base64 JPEGs, not needed here.
        # GUARDED: build the dict only when tracing is on — it reads request fields, and building
        # it unconditionally (with stale keys) would raise on every inference. Logs BOTH arms' EE.
        if TRACE_JSONL:
            self._trace("requests", {"obs_ts": sid,
                                     "robotl_eef_pos": req.get("robotl_eef_pos"),
                                     "robotr_eef_pos": req.get("robotr_eef_pos"),
                                     "robot0_grip": req.get("robot0_grip")})

        t_req = time.monotonic()
        resp = post_predict(self.host, self.port, req, timeout=10)
        srv_ms = (time.monotonic() - t_req) * 1e3          # policy-server round-trip latency
        if 'error' in resp:
            log.warning("server error: %s", resp["error"])
            return False

        # The dual_arm policy returns 20-col rows: L[pos(3)+rot6d(6)+grip(1)] ++ R[pos(3)+rot6d(6)+
        # grip(1)]. We now drive BOTH arms, so the full 20-col row is carried through the temporal
        # ensemble, IK (env solves left+right), sim validation, and release. (10-col left-only rows
        # still pass through unchanged for back-compat with a single-arm policy.)
        action = np.asarray(resp['action'], dtype=np.float32)
        log.debug("response received in %.3fs", time.time() - ts)
        self._last_inference_obs_ts = ts
        # A fresh chunk restarts the manual step-through from the first row.
        self._exec_cursor = 0

        # Track the right-gripper command so recovery can spot an open->close grasp attempt.
        if rec is not None:
            rec.note_action(action)

        # POST-PROCESSING (stage 1: gripper binarize + temporal-ensemble merge). The pipeline binarizes
        # the grippers in place and, for AUTO/streaming (submit=True), splices this chunk into its
        # smoothed master buffer and returns the contiguous run to feed the robot; manual one-shots
        # (submit=False) pass the raw chunk through. queue_status gives the live clock + commit frontier
        # the merge aligns on. `base_id` is the run's first master id; `jerk` is the smoothness-guard
        # diagnostic. See real_world/postprocess.py.
        cur, queued_through = env.queue_status()
        base_id, buf, jerk = env.pipeline.merge(action, sid, cur, queued_through, submit)

        # Log the RAW chunk (post-binarize, pre-buffer would differ only by smoothing), keyed by the
        # master id it's anchored to.
        if TRACE_JSONL:
            self._trace("chunks", {"obs_ts": sid, "action": action.tolist()})

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

        append_ok = None
        if submit and buf.size:
            # AUTO-to-robot (streaming): append_actions reads the live clock + queued_through itself
            # and pulls the window [queued_through+1 .. clock+n] from this contiguous buffer (base_id
            # = its first master id), validating + appending only the not-yet-queued ids. Rows are
            # only ever appended (never cleared); the release loop drains one master id at a time.
            append_ok, reason = env.append_actions(buf_list, int(base_id))
            if not append_ok:
                log.info("auto: append skipped — %s", reason)

        # ---- per-inference lifecycle trace ---------------------------------------------------
        # One line per inference so the timeline is legible: START/END wall-clock, how long the
        # whole cycle took (obs -> server -> ensemble -> validate+append), the master-id RANGE this
        # inference carried onto the robot, and where the robot's release clock is right now (so the
        # "lead" = how many master ids are queued ahead of the arm). Only in the AUTO->robot path;
        # manual one-shots (submit=False) don't stream and would just add noise.
        t_end_wall = time.time()
        dur_ms = (time.monotonic() - t_start_mono) * 1e3
        clk, qthru = env.queue_status()                    # re-read: queue advanced during append
        rows = action.shape[0] if action.ndim == 2 else len(action)
        self._infer_count += 1
        if submit:
            span = f"{int(base_id)}..{int(base_id) + rows - 1}" if rows else str(int(base_id))
            log.info("[infer] #%d | start %s end %s | took %.1f ms | carried ids %s | "
                     "robot@id %d (queued->%d, lead %d)%s%s",
                     self._infer_count, _hms(t_start_wall), _hms(t_end_wall), dur_ms, span,
                     clk, qthru, qthru - clk,
                     f" | srv {srv_ms:.0f} ms" if srv_ms >= 50 else "",
                     "" if append_ok is None or append_ok else " | append SKIPPED")
        return True

    # ------------------------------------------------------------------ #
    #  Live-tunable smoothing knobs — delegated to env.pipeline (the       #
    #  PostProcessor owns the merge state). Kept here so the GUI's         #
    #  existing self.inference.set_smoothing / .te_radius etc. still work. #
    # ------------------------------------------------------------------ #
    @property
    def _pipeline(self):
        return self.humanoid_env.pipeline

    def set_smoothing(self, radius=None, sigma=None, m=None):
        """Live-tunable temporal-ensemble smoothing (delegates to env.pipeline). Returns
        (radius, sigma, m)."""
        return self._pipeline.set_smoothing(radius=radius, sigma=sigma, m=m)

    def set_buffer_len(self, n):
        """Live-tunable overlap depth = # recent raw chunks averaged (delegates to env.pipeline)."""
        return self._pipeline.set_buffer_len(n)

    @property
    def te_buffer_len(self):
        return self._pipeline.te_buffer_len

    @property
    def te_radius(self):
        return self._pipeline.te_radius

    @property
    def te_sigma(self):
        return self._pipeline.te_sigma

    @property
    def te_m(self):
        return self._pipeline.te_m

    @property
    def use_temporal_ensemble(self):
        return self._pipeline.use_temporal_ensemble

    @use_temporal_ensemble.setter
    def use_temporal_ensemble(self, v):
        self._pipeline.use_temporal_ensemble = v

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
                log.info("[auto] STOP requested — draining queue (%d inferences this run).",
                         self._infer_count)
                self.inference_thread.join()
                self.inference_thread = None
            # Clear any in-flight retreat so a later restart begins fresh instead of resuming a stale
            # one anchored at an old clock value.
            rec = getattr(self, "recovery", None)
            if rec is not None:
                rec.reset()
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
                    rec = getattr(self, "recovery", None)
                    if rec is not None:
                        rec.reset()                        # drop any in-flight retreat (queue was cleared)
                    self.is_auto_inference = False
                    break
                # CCDP recovery: while retreating from a missed grasp, stream the scripted
                # retreat (~APPEND_AHEAD_ROWS/cycle) instead of the policy and skip the server.
                # Exits back to normal inference once the retreat has drained (see grasp_recovery).
                rec = getattr(self, "recovery", None)
                if rec is not None and rec.is_retreating:
                    rec.pump(self.humanoid_env)
                    time.sleep(0.02)
                    continue
                # submit=True -> validate + splice onto the robot (and emit the per-inference
                # lifecycle line, see _run_inference). On a skipped inference (stale obs, server
                # error, validation failure) back off briefly so we don't busy-spin.
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
            cadence = f"{self.inference_hz:.1f} Hz cap" if self.inference_hz and self.inference_hz > 0 \
                else "uncapped (latency-bound)"
            log.info("[auto] START -> robot | server %s:%s | %s | temporal-ensemble %s",
                     self.host, self.port, cadence, "ON" if self.use_temporal_ensemble else "OFF")
            self.inference_thread.start()
