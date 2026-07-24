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
PC4080_PORT = 9000


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


# Default grasp-detector checkpoint: the 3-class YOLO gripper-state model shipped next to this
# file in real_world/assets/. Resolved from __file__ so it works regardless of the CWD.
DEFAULT_GRASP_DETECTOR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "assets", "right_yolo.pt")


class InferenceController:
    """Drives the left_arm_ee_image policy: server round-trip + env hand-off."""

    def __init__(self, env, robot_info=None,
                 host=PC4080_HOST, port=PC4080_PORT,
                 record_hz=RECORD_HZ, inference_hz=INFERENCE_HZ,
                 obs_source=None, grasp_detector_path=DEFAULT_GRASP_DETECTOR):
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
        # ABSOLUTE 14-joint START pose the arm slowly homes to at the top of each auto run, BEFORE any
        # server call, so every run begins from the same configuration. = arm_joints[0] of
        # MDM_data_collection/recordings/recording001. Set to None to skip the homing step.
        self.auto_start_pose = np.asarray(
            [2.041772, -0.287803, -2.07299, -1.220383, 0.279462, 1.207418, -1.292626,
             -1.525758, 0.706167, 1.504347, 0.82008, 0.019143, -1.606412, -1.901265],
            dtype=float)
        # Pre-grasp APPROACH waypoints (n, 14) averaged from the recordings
        # (scripts/estimate_retreat_waypoints.py -> config/retreat_waypoints.json). The missed-grasp
        # recovery retreats to the nearest of these that isn't ahead of the arm's current approach
        # phase, instead of always homing to the start. Row 0 is the average start pose; when the file
        # is present it also supersedes auto_start_pose as the top-of-run homing target. Falls back to
        # a single-waypoint [auto_start_pose] (= today's fixed-home behaviour) when the file is absent.
        from real_world.postprocess import load_retreat_waypoints
        _wps = load_retreat_waypoints()
        self.retreat_waypoints = _wps if _wps is not None else self.auto_start_pose[None, :]
        if _wps is not None:
            self.auto_start_pose = np.asarray(_wps[0], dtype=float)   # home to waypoint 1
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
                self.recovery = GraspRecoveryMonitor(det_path, closed_grip_min=GRIPPER_CLOSE_THRESH,
                                                     retreat_waypoints=self.retreat_waypoints)
            except Exception as e:
                log.warning("grasp recovery disabled (failed to load %s): %r", det_path, e)
        elif det_path:
            log.warning("grasp recovery disabled: detector checkpoint not found at %s", det_path)

        # Speed of the scripted place macros (flip + no-flip), as a fraction of MAX joint velocity.
        # Lower = slower/gentler release motion. (The policy's own auto motion is env.speed_scale.)
        PLACE_MACRO_VEL_FRAC = 0.25

        # Flip-place release macro (opt-in): after the policy's grab+lift+flip, move out to a fixed
        # release pose, open the gripper, and come back — see real_world/flip_place.py. Inert unless the
        # baked recording path exists.
        self.flip_place = None
        try:
            from real_world.flip_place import FlipPlaceMacro, DEFAULT_PATH
            if os.path.exists(DEFAULT_PATH):
                self.flip_place = FlipPlaceMacro(vel_frac=PLACE_MACRO_VEL_FRAC)
            else:
                log.warning("flip-place disabled: release path not found at %s", DEFAULT_PATH)
        except Exception as e:
            log.warning("flip-place disabled (failed to init): %r", e)

        # No-flip-place release macro (opt-in): when a camera sees a BARCODE on the grasped package it
        # is oriented right-side-up and must NOT be flipped, so place it as-is — see
        # real_world/no_flip_place.py. Barcode-armed, so it fires (before the flip macro) only on
        # packages that show a code; inert otherwise.
        self.no_flip_place = None
        barcode_gate = None
        try:
            from real_world.no_flip_place import NoFlipPlaceMacro, YoloGate, DEFAULT_PATH as NFP_PATH
            if os.path.exists(NFP_PATH):
                # The model now emits both "barcode" and "package", so the no-flip cue must be filtered
                # to the barcode class ONLY — otherwise a package with no barcode would false-trigger the
                # no-flip release. (Case-insensitive: matches a model class named "Barcode" too.)
                barcode_gate = YoloGate(classes={"barcode"})
                self.no_flip_place = NoFlipPlaceMacro(detector=barcode_gate, vel_frac=PLACE_MACRO_VEL_FRAC)
            else:
                log.warning("no-flip-place disabled: release path not found at %s", NFP_PATH)
        except Exception as e:
            log.warning("no-flip-place disabled (failed to init): %r", e)

        # Package-presence gate: pause auto inference + park at home when no package is present, resume
        # when one appears (see real_world/package_gate.py). Reuses the barcode gate's already-loaded
        # YOLO model. Fail-open: inert until a package-capable model is trained, so it is safe to ship
        # alongside today's barcode-only model.
        self.package_gate = None
        try:
            from real_world.package_gate import PackagePresenceGate
            shared_model = barcode_gate.model if barcode_gate is not None else None
            self.package_gate = PackagePresenceGate(model=shared_model)
        except Exception as e:
            log.warning("package-gate disabled (failed to init): %r", e)

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
                # Unreachable rows are dropped INSIDE solve_chunk_ik (the arm just bridges past them),
                # so an append skip here is a benign no-op (buffer full / chunk past the window / no
                # reachable rows this cycle) — log and move on.
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
            # DIAGNOSTIC: how far is the row auto is about to run from where the arm ACTUALLY is?
            # The row for the live clock sits at buf index (clk - base_id). If |current EE - that row|
            # is large, auto is commanding a jump away from the current pose (the "ignores current
            # state / moves to another location" symptom) rather than continuing from it.
            gap = self._anchor_gap(obs, buf, base_id, clk)
            log.info("[infer] #%d | start %s end %s | took %.1f ms | carried ids %s | "
                     "robot@id %d (queued->%d, lead %d)%s%s%s",
                     self._infer_count, _hms(t_start_wall), _hms(t_end_wall), dur_ms, span,
                     clk, qthru, qthru - clk,
                     f" | srv {srv_ms:.0f} ms" if srv_ms >= 50 else "",
                     "" if append_ok is None or append_ok else " | append SKIPPED",
                     gap)
        return True

    @staticmethod
    def _anchor_gap(obs, buf, base_id, clk):
        """Diagnostic string: EE-position gap (m) between the arm's CURRENT pose and the buffer row
        that maps to the live master clock (index clk-base_id). A big gap = auto is anchoring away
        from the current state. Returns '' if it can't be computed (shape/index out of range)."""
        try:
            b = np.asarray(buf, dtype=float)
            i0 = int(clk) - int(base_id)
            if b.ndim != 2 or not (0 <= i0 < b.shape[0]):
                return f" | anchor idx {i0} OOR (buf {b.shape})"
            row = b[i0]
            lcur = np.asarray(obs["robotl_eef_pos"][-1], dtype=float)[0:3]
            dl = float(np.linalg.norm(row[0:3] - lcur))
            msg = f" | L-gap {dl*100:.1f}cm"
            if b.shape[1] > 19:                            # dual-arm row: right EE too
                rcur = np.asarray(obs["robotr_eef_pos"][-1], dtype=float)[0:3]
                msg += f" R-gap {float(np.linalg.norm(row[10:13] - rcur))*100:.1f}cm"
            return msg
        except Exception:
            return ""

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

    @property
    def flip_place_enabled(self):
        """Whether the post-flip release macro is armed. False (or no macro loaded, e.g. the baked
        release path is missing) -> the macro never fires and auto inference just keeps running."""
        fp = getattr(self, "flip_place", None)
        return fp is not None and fp.enabled

    @flip_place_enabled.setter
    def flip_place_enabled(self, v):
        fp = getattr(self, "flip_place", None)
        if fp is not None:
            fp.enabled = bool(v)
        elif v:
            log.warning("flip-place cannot be enabled: no macro loaded (baked release path missing).")

    @property
    def no_flip_place_enabled(self):
        """Whether the barcode-triggered no-flip release macro is armed. False (or no macro loaded)
        -> it never fires and auto inference just keeps running (the flip path still applies)."""
        nfp = getattr(self, "no_flip_place", None)
        return nfp is not None and nfp.enabled

    @no_flip_place_enabled.setter
    def no_flip_place_enabled(self, v):
        nfp = getattr(self, "no_flip_place", None)
        if nfp is not None:
            nfp.enabled = bool(v)
        elif v:
            log.warning("no-flip-place cannot be enabled: no macro loaded (baked release path missing).")

    def no_flip_place_status(self):
        """Live snapshot for the GUI barcode indicator, or None if no macro is loaded."""
        nfp = getattr(self, "no_flip_place", None)
        return nfp.status() if nfp is not None else None

    # YoloGate tunables the GUI exposes for live calibration (order = display order).
    _LABEL_PARAM_KEYS = ("conf", "iou", "imgsz")

    def no_flip_detector(self):
        """The live no-flip detector object (YoloGate) for the tuning page to run analyze() on, or
        None if there's no macro. Same instance the tuning widgets mutate, so edits show immediately."""
        nfp = getattr(self, "no_flip_place", None)
        return getattr(nfp, "detector", None) if nfp is not None else None

    def no_flip_detector_params(self):
        """Current live no-flip detection tunables (YoloGate params + commit_count), or None if there
        is no macro / the detector isn't a YoloGate. Used to seed the GUI tuning widgets."""
        nfp = getattr(self, "no_flip_place", None)
        det = getattr(nfp, "detector", None) if nfp is not None else None
        if det is None or not all(hasattr(det, k) for k in self._LABEL_PARAM_KEYS):
            return None
        out = {k: getattr(det, k) for k in self._LABEL_PARAM_KEYS}
        out["commit_count"] = nfp.commit_count
        return out

    def set_no_flip_param(self, name, value):
        """Live-set one no-flip detection tunable (a YoloGate param, or commit_count on the macro).
        Takes effect on the very next scan. No-op if there's no macro / the param isn't present."""
        nfp = getattr(self, "no_flip_place", None)
        if nfp is None:
            return
        if name == "commit_count":
            nfp.commit_count = max(1, int(value))
            return
        det = getattr(nfp, "detector", None)
        if det is not None and hasattr(det, name):
            cur = getattr(det, name)
            setattr(det, name, type(cur)(value))     # keep the param's original type (int vs float)

    def no_flip_place_scan(self):
        """Advance barcode detection one tick WITHOUT firing. Lets the GUI keep the live indicator
        responsive while auto-run is OFF (the auto loop already scans while running). No-op if there's
        no macro / env; failures are swallowed so a camera hiccup can never disturb the UI thread."""
        nfp = getattr(self, "no_flip_place", None)
        env = self.humanoid_env
        if nfp is not None and env is not None:
            try:
                nfp.scan(env)
            except Exception as e:
                log.debug("no-flip-place idle scan skipped: %r", e)

    # ---- Gripper-state detector (grasp-recovery YOLO): live indicator + tuning-view surface -------- #
    def gripper_detector(self):
        """The live gripper-state detector (grasp_recovery._Detector) for the tuning page to run
        analyze() on, or None if grasp recovery isn't loaded. Same instance the recovery uses, so a
        confidence edit in the tuning tab changes the live recovery check too."""
        rec = getattr(self, "recovery", None)
        return getattr(rec, "det", None) if rec is not None else None

    def gripper_detector_params(self):
        """Current live gripper-detector tunable(s) to seed the tuning widgets, or None."""
        det = self.gripper_detector()
        return {"conf": det.conf} if det is not None else None

    def set_gripper_param(self, name, value):
        """Live-set one gripper-detector tunable (currently just 'conf'); effective on the next scan."""
        det = self.gripper_detector()
        if det is not None and hasattr(det, name):
            cur = getattr(det, name)
            setattr(det, name, type(cur)(value))     # keep the param's original type

    def gripper_state_status(self):
        """CHEAP snapshot for the GUI gripper pill — reads the recovery monitor's cached last read, NO
        forward pass. During auto the loop's maybe_start keeps that cache fresh; while idle the GUI
        calls gripper_scan() (below) to refresh it. None if no detector is loaded."""
        rec = getattr(self, "recovery", None)
        return rec.live_status() if rec is not None else None

    def gripper_scan(self, imgsz=320):
        """Run ONE right-wrist forward pass and cache it for the pill. Called by the GUI ONLY while auto
        is OFF and the indicator is visible, so YOLO never runs on the UI thread during auto (where it
        would double-infer and race the loop on the same model). No-op if recovery isn't loaded."""
        rec = getattr(self, "recovery", None)
        if rec is not None:
            rec.scan_live(self.humanoid_env, imgsz=imgsz)

    def scan_head_gates(self):
        """Drive the barcode (no-flip) and package gates from ONE shared head-camera YOLO pass rather
        than two. Both gates reuse the same model, so this halves the idle head-cam inference the GUI
        pills would otherwise do (one predict feeds both latch/debounce updates). No-op if env or both
        gates are missing; degrades to per-gate scans inside scan_gates_once if the model isn't shared."""
        from real_world.no_flip_place import scan_gates_once
        env = self.humanoid_env
        nfp = getattr(self, "no_flip_place", None)
        pkg = getattr(self, "package_gate", None)
        bg = getattr(nfp, "detector", None) if nfp is not None else None
        pg = getattr(pkg, "detector", None) if pkg is not None else None
        gates = [g for g in (bg, pg) if g is not None]
        if env is None or not gates:
            return
        seen = scan_gates_once(env, gates)
        if nfp is not None and bg is not None:
            nfp.scan(env, seen=seen.get(bg, False))
        if pkg is not None and pg is not None:
            pkg.scan(env, seen=seen.get(pg, False))

    # ---- Package-presence gate: pause + home when there's no package to work on ------------------- #
    @property
    def package_gate_enabled(self):
        """Whether the package-presence gate is armed. When off (or fail-open, i.e. the model can't
        detect a 'package' class) auto inference never pauses on package absence."""
        pkg = getattr(self, "package_gate", None)
        return pkg is not None and pkg.enabled

    @package_gate_enabled.setter
    def package_gate_enabled(self, v):
        pkg = getattr(self, "package_gate", None)
        if pkg is not None:
            pkg.enabled = bool(v)
            if not v:
                pkg.reset()                            # disarming -> drop any active pause immediately
        elif v:
            log.warning("package-gate cannot be enabled: no gate loaded.")

    def package_gate_status(self):
        """Live snapshot for the GUI package indicator, or None if no gate is loaded."""
        pkg = getattr(self, "package_gate", None)
        return pkg.status() if pkg is not None else None

    def _home_arm(self, env):
        """Move the arm to the fixed auto start pose (absolute joints), clearing any queued/streamed
        motion first so homing has sole control. No-op if disabled / unsupported / E-stopped. Shared by
        the initial auto homing intent and the package-gate pause."""
        if self.auto_start_pose is None or not hasattr(env, "move_to_joints") or env.estopped:
            return
        if getattr(env, "pipeline", None) is not None:
            env.pipeline.clear_queue()
        try:                                           # drop robot queue + staging so nothing snaps
            with env._lock:
                env._robot_q.clear()
                env._staged_release.clear()
                env._queued_through = -1
        except AttributeError:
            pass
        env.move_to_joints(self.auto_start_pose)

    def _maybe_pause_for_package(self, env) -> bool:
        """Top-of-auto-loop step. Returns True if inference should be SKIPPED this cycle because there is
        no package to work on (the arm is homed and paused until one reappears). Pauses ONLY when idle
        (gripper empty); while holding an item it never pauses (the arm occludes the package). Fail-open:
        never pauses unless the gate is enabled AND capable of detecting the package class."""
        pkg = getattr(self, "package_gate", None)
        if pkg is None or not pkg.enabled or not pkg.capable:
            return False
        from real_world.no_flip_place import R_GRIP_IDX
        grip = getattr(env, "_last_grip_cmd", None)
        holding = grip is not None and grip[R_GRIP_IDX] >= 1
        if holding:                                    # busy -> treat as present, never pause mid-task
            pkg.on_busy()
            return False
        pkg.scan(env)                                  # idle -> debounced presence belief
        if pkg.present:
            if pkg.paused:
                log.warning("[package-gate] package detected -> resuming auto inference")
            pkg.paused = False
            return False
        # No package (debounced) while idle -> park at home ONCE, then hold (skip inference) until it
        # returns. Re-home only on the transition into pause so we don't fight the arm every tick.
        if not pkg.paused:
            log.warning("[package-gate] no package while idle -> homing + pausing auto inference")
            self._home_arm(env)
            nfp = getattr(self, "no_flip_place", None)
            if nfp is not None:
                nfp.reset()                            # drop any stale barcode commit latch
            pkg.paused = True
        return True

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
            fp = getattr(self, "flip_place", None)
            if fp is not None:
                fp.reset()
            nfp = getattr(self, "no_flip_place", None)
            if nfp is not None:
                nfp.reset()
            pkg = getattr(self, "package_gate", None)
            if pkg is not None:
                pkg.reset()
            # Stop feeding NEW chunks; let the release loop drain whatever is already queued. (Use
            # the env E-stop to halt immediately instead.)
            return

        env = self.humanoid_env
        if env is None or getattr(env, "sim", None) is None:
            log.warning("auto refused: launch the simulation preview first "
                        "(validation runs through it).")
            return

        self.is_auto_inference = True
        if hasattr(env, "reset_grip_latch"):
            env.reset_grip_latch()                         # fresh run -> gripper can re-latch from scratch
        nfp = getattr(self, "no_flip_place", None)
        if nfp is not None:
            nfp.reset()                                    # drop any idle pre-arm; decide live this run
        pkg = getattr(self, "package_gate", None)
        if pkg is not None:
            pkg.reset()                                    # fresh run -> assume present, decide live

        def _run_auto_inference():
            # Home to a fixed START pose (absolute joints, slow) BEFORE any server call, so every run
            # begins from the same configuration. Blocks until the arm arrives; skipped if disabled,
            # unsupported, or E-stopped.
            if (self.auto_start_pose is not None and hasattr(env, "move_to_joints")
                    and not env.estopped):
                # Drop any residual queued substeps so the release loop is idle and homing has sole
                # control of the arm (both would otherwise call the SDK concurrently).
                if getattr(env, "pipeline", None) is not None:
                    env.pipeline.clear_queue()
                log.info("[auto] homing to start pose (absolute joints) before inference…")
                env.move_to_joints(self.auto_start_pose)
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
                    fp = getattr(self, "flip_place", None)
                    if fp is not None:
                        fp.reset()                         # disarm the flip macro too
                    nfp = getattr(self, "no_flip_place", None)
                    if nfp is not None:
                        nfp.reset()                        # ...and the no-flip macro
                    pkg = getattr(self, "package_gate", None)
                    if pkg is not None:
                        pkg.reset()                        # ...and drop any package-gate pause
                    self.is_auto_inference = False
                    break
                # Package-presence gate: while idle (empty gripper), if no package is in view the arm is
                # homed and inference is PAUSED until one reappears — so the robot doesn't run on an empty
                # workspace. Checked first; inert (never pauses) unless a package-capable model is loaded.
                if self._maybe_pause_for_package(self.humanoid_env):
                    continue
                # No-flip-place: if a camera sees a barcode on the grasped package it is right-side-up
                # and must NOT be flipped, so place it as-is. Checked BEFORE the flip macro so a barcode
                # (seen right after the grasp) takes over before the wrist ever swings; on packages
                # without a barcode this is inert and the flip path below runs instead.
                nfp = getattr(self, "no_flip_place", None)
                if nfp is not None and nfp.maybe_trigger(self.humanoid_env):
                    continue
                # Flip-place release macro: once the policy's grab+lift+flip completes, this stops
                # predicting, moves out to the fixed release pose, opens the gripper, and comes back —
                # all inline (clears every queue before/after so auto resumes without snapping).
                fp = getattr(self, "flip_place", None)
                if fp is not None and fp.maybe_trigger(self.humanoid_env):
                    continue
                # CCDP recovery is now SYNCHRONOUS: a detected miss opens the gripper + moves the arm
                # home inside maybe_start() (called from _run_inference below), then normal inference
                # resumes from home — no streamed-retreat pump step here anymore.
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
            rec_state = "ON" if getattr(self, "recovery", None) is not None \
                else "OFF (no detector checkpoint — set grasp_detector_path or HUMANOID_GRASP_DETECTOR)"
            log.info("[auto] START -> robot | server %s:%s | %s | temporal-ensemble %s | grasp-recovery %s",
                     self.host, self.port, cadence, "ON" if self.use_temporal_ensemble else "OFF", rec_state)
            self.inference_thread.start()
