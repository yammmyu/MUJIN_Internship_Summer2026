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

RECORD_HZ = 30
INFERENCE_HZ = 0  # auto-inference cadence cap (Hz); <=0 -> run back-to-back (max overlap for TE)

# --- Temporal ensemble (ACT-style) -------------------------------------------------
# Chunk-level EE-space smoothing: each new chunk is averaged against recent overlapping
# chunks before validation. The policy emits action rows at its record/training rate, so row k of
# a chunk observed at ts_obs targets wall-clock ts_obs + k*ROW_DT. For each row of the newest chunk
# we find every buffered chunk's row at the SAME wall-clock (j = round((t_k - ts_i)/ROW_DT)) and
# take a recency-weighted mean (weight exp(-TE_M * age), age = inferences old, newest = 0). This
# only smooths once chunks actually OVERLAP in time, i.e. when (horizon * ROW_DT) > inference_period;
# below that it safely passes the newest chunk through (identity).
USE_TEMPORAL_ENSEMBLE = True
TE_M = 0.02            # decay; larger -> trust the newest chunk more (less averaging)
TE_BUFFER_LEN = 8      # how many recent raw chunks to keep for averaging
# Wall-clock spacing between consecutive policy action rows = the policy's action timestep (the
# rate it was trained/recorded at). The env subdivides each row into STEP_TIME substeps for
# execution, but that doesn't change this nominal per-row spacing — the ensemble aligns rows by it.
# (If the policy's action rate ever differs from RECORD_HZ, change this one constant.)
ROW_DT = 1.0 / RECORD_HZ
PC4080_HOST = "10.12.11.144"
PC4080_PORT = 9001


def encode_image(rgb_image):
    """RGB frame -> base64 JPEG string (matches the training/recording encode)."""
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
        # wall-clock timestamp with every snapshot).
        ts = obs['timestamp']
        if ts - self._last_inference_obs_ts < 1e-4:
            print("[InferenceController] timestamp has not advanced")
            return False

        print(f"[InferenceController] preping request | Time elapsed; {time.time()- ts}")
        req = {
            'agent_imgs': [encode_image(img) for img in obs['agent_imgs']],
            'hand_imgs': [encode_image(img) for img in obs['hand_imgs']],
            'state': obs['state'],
            # robot0_left_joint policy input: [j_{t-1}, j_t], each = 7 left-arm joints (rad) + raw
            # gripper. Server must map this to robot0_left_joint (see train config shape_meta).
            'left_joint': obs['joint_state'],
        }

        # Log the proprioception sent to the policy server, keyed by the same obs_ts that
        # chunks.jsonl/traj.jsonl use, so a request can be joined to its resulting action.
        # Images (agent_imgs/hand_imgs) are omitted — large base64 JPEGs, not needed here.
        with open("requests.jsonl", "a") as f:
            f.write(json.dumps({"obs_ts": float(ts),
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

        # Temporal ensemble: buffer this RAW chunk, then average it with recent overlapping chunks
        # in EE space. The ensembled chunk is what we publish + validate; the raw one stays only in
        # the buffer for future averaging. No-op (returns the newest chunk) until chunks overlap.
        with open("chunks.jsonl", "a") as f:
            f.write(json.dumps({"obs_ts": float(ts), "action": action.tolist()}) + "\n")

        print(f"[InferenceController] sending for temporal ensemble! | Time elapsed; {time.time()- ts}")
        if self.use_temporal_ensemble and action.ndim == 2 and action.shape[1] >= 10:
            self._te_buffer.append((ts, action.copy()))
            action = self._temporal_ensemble(ts, action)
        print(f"[InferenceController] ensemble finished | Time elapsed; {time.time()- ts}")

        with open("TEchunks.jsonl", "a") as f:
            f.write(json.dumps({"obs_ts": float(ts), "action": action.tolist()}) + "\n")
        # Publish for robot_info_server / visualisation. left_*_predict_* carry
        # EE-pose data here (see robot_info_server.RobotInfo notes):
        #   *_start_values  = last two left EE states ([pos(3), quat(4), grip(1)])
        #   *_action_values = predicted action rows
        # Skipped when robot_info is None (headless sim runner has no GUI to publish to).
        if self.robot_info is not None:
            with self.robot_info.lock:
                self.robot_info.left_joint_predict_start_values = copy.deepcopy(obs['state'])
                self.robot_info.left_joint_predict_action_values = action.tolist()
                self.robot_info.inference_timestamp = ts

        print(f"[InferenceController] sending chunk for validation | Time elapsed; {time.time()- ts}")
        if submit:
            # AUTO-to-robot: validate this chunk on the preview sim and time-aligned-splice it into
            # the live robot queue (no manual release). ts is the obs wall-clock used to align the
            # splice; validation failure just skips this inference (the previous trajectory drains on).
            ok, reason = env.auto_ingest_chunk(action.tolist(), ts)
            if not ok:
                print(f"[InferenceController] auto: chunk skipped — {reason}")
        return True

    def _temporal_ensemble(self, ts_new, new_action):
        """ACT-style chunk-level temporal ensemble, in EE-pose space.

        Returns an (N,10) chunk: for each row k of the newest chunk (target wall-clock
        t_k = ts_new + k*ROW_DT) we gather, from every buffered RAW chunk, its row nearest to t_k
        (j = round((t_k - ts_i)/ROW_DT)), weight it w = exp(-TE_M * age) (age = how many inferences
        old, newest = 0), and take the weighted mean. pos (0:3) and rot6d (3:9) average linearly
        (rot6d is re-orthonormalised downstream by rot6d_to_quat); gripper (9), already {0,1},
        averages then re-thresholds at 0.5. With no temporal overlap only the newest chunk
        contributes -> identity (safe no-op). Assumes the new raw chunk is already in self._te_buffer.

        ROW_DT (not STEP_TIME) is the unit: the buffer holds RAW chunks whose rows are one policy
        action timestep apart, BEFORE the env subdivides them into STEP_TIME substeps."""
        N = new_action.shape[0]
        out = new_action.copy()
        buf = list(self._te_buffer)                 # oldest -> newest
        newest_idx = len(buf) - 1
        for k in range(N):
            t_k = ts_new + k * ROW_DT
            rows, weights = [], []
            for idx, (ts_i, act_i) in enumerate(buf):
                j = int(round((t_k - ts_i) / ROW_DT))
                if 0 <= j < act_i.shape[0]:
                    age = newest_idx - idx              # newest chunk -> 0
                    rows.append(act_i[j])
                    weights.append(np.exp(-TE_M * age))
            if len(rows) <= 1:                          # only the newest covers t_k -> identity
                continue
            w = np.asarray(weights, dtype=np.float64)
            w /= w.sum()
            avg = (w[:, None] * np.asarray(rows, dtype=np.float64)).sum(axis=0)
            out[k, :9] = avg[:9]                         # pos + rot6d (linear)
            out[k, 9] = 1.0 if avg[9] >= 0.5 else 0.0    # gripper re-binarised
        return out

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
        preview sim and time-aligned-spliced into the live robot queue (env.auto_ingest_chunk) — no
        manual release. The inference cadence is set by the server+validation latency itself (the
        substep queue absorbs the gap), so there is no fixed inference_hz sleep. Requires the preview
        sim to be running (validation) and real=True; the GUI launches the sim before starting."""
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
                # (auto_ingest_chunk also refuses while latched, so nothing is spliced even in
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
