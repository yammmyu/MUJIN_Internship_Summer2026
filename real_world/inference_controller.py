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

RECORD_HZ = 30
INFERENCE_HZ = 10
PC4080_HOST = "10.12.11.144"
PC4080_PORT = 9001


def encode_image(rgb_image):
    """RGB frame -> base64 JPEG string (matches the training/recording encode)."""
    rgb_image_cv2 = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.jpg', rgb_image_cv2, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buffer).decode('utf-8')


def post_predict(host: str, port: int, req: dict, timeout: float = 60.0) -> dict:
    """POST an obs dict to the policy server's /predict and return the JSON reply."""
    print(f"inference request sent. Details:{req}")
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

    def __init__(self, env, robot_info,
                 host=PC4080_HOST, port=PC4080_PORT,
                 record_hz=RECORD_HZ, inference_hz=INFERENCE_HZ):
        self.humanoid_env = env                 # owned/started/stopped by the caller (GUI)
        self.robot_info = robot_info            # shared with robot_info_server
        self.host = host
        self.port = port
        self.record_hz = record_hz
        self.inference_hz = inference_hz

        self.inference_thread = None
        self.is_auto_inference = False
        self._last_inference_obs_ts = 0.0

    # ------------------------------------------------------------------ #
    #  Lifecycle: the env is owned by the caller; we only reset our state #
    # ------------------------------------------------------------------ #
    def start(self):
        """Prepare for inference. The injected env's threads are started by the
        caller (the GUI starts env before this), so we only reset our cursor."""
        self._last_inference_obs_ts = 0.0
        print("InferenceController ready (env owned by caller).")

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

        print(f"running inference sumbit={submit}")
        env = self.humanoid_env
        if env is None:
            print("humanoid_env not accessible")
            return False

        obs = env.get_obs()
        if obs is None:
            return False
        # Skip if we've already inferred on this observation (env publishes a
        # wall-clock timestamp with every snapshot).
        ts = obs['timestamp']
        if ts - self._last_inference_obs_ts < 1e-4:
            print("timestamp has not advanced")
            return False

        req = {
            'agent_imgs': [encode_image(img) for img in obs['agent_imgs']],
            'hand_imgs': [encode_image(img) for img in obs['hand_imgs']],
            'state': obs['state'],
        }

        resp = post_predict(self.host, self.port, req, timeout=10)

        if 'error' in resp:
            print(f'\nServer error: {resp["error"]}')
            return False

        # action rows: [eef_pos(3), 6D_rot(6), gripper(1)]
        action = np.asarray(resp['action'], dtype=np.float32)
        print(f"response recieved! Details:{action}")
        self._last_inference_obs_ts = ts

        # Publish for robot_info_server / visualisation. left_*_predict_* carry
        # EE-pose data here (see robot_info_server.RobotInfo notes):
        #   *_start_values  = last two left EE states ([pos(3), quat(4), grip(1)])
        #   *_action_values = predicted action rows
        with self.robot_info.lock:
            self.robot_info.left_joint_predict_start_values = copy.deepcopy(obs['state'])
            self.robot_info.left_joint_predict_action_values = action.tolist()
            self.robot_info.inference_timestamp = ts

        if submit:
            env.submit_actions(action.tolist())
        return True

    def inference_once(self) -> bool:
        """Predict + publish only (no execution).

        Manual mode: call this to predict, then execute_inference_result to run.
        Warms the policy cameras first (they may be idle-evicted), since a single
        shot can't rely on the retry loop that auto_inference has.
        """
        env = self.humanoid_env
        if env is None:
            print("humanoid_env not accessible")
            return False
        deadline = time.monotonic() + 0.3
        while not env.inf_ready and time.monotonic() < deadline:
            env.get_obs()           # requests head + hand_left, warming them
            time.sleep(0.02)
        return self._run_inference(submit=False)

    # ------------------------------------------------------------------ #
    #  Hand the most recent stored prediction to the env's exec queue      #
    # ------------------------------------------------------------------ #
    def execute_inference_result(self, once: bool = False):
        """Submit the last published action chunk to the env for execution.

        once=True submits only the first action row.
        """
        env = self.humanoid_env
        if env is None:
            return
        with self.robot_info.lock:
            actions = copy.deepcopy(self.robot_info.left_joint_predict_action_values) or []
        if not actions:
            return
        env.submit_actions(actions[:1] if once else actions)

    # ------------------------------------------------------------------ #
    #  Auto loop: predict + submit continuously                            #
    # ------------------------------------------------------------------ #
    def auto_inference(self, stop: bool = False):
        if stop:
            self.is_auto_inference = False
            if self.inference_thread is not None:
                print("Stopping auto_inference thread...")
                self.inference_thread.join()
                self.inference_thread = None
            # Stop feeding the robot: drop any queued actions in the env.
            if self.humanoid_env is not None:
                self.humanoid_env.submit_actions([])
            return

        self.is_auto_inference = True

        def _run_auto_inference():
            while self.is_auto_inference:
                if self._run_inference(submit=True):
                    time.sleep(1 / self.inference_hz)
                else:
                    time.sleep(0.01)

        if self.inference_thread is None:
            self.inference_thread = threading.Thread(target=_run_auto_inference, daemon=True)
            print("Starting auto_inference thread...")
            self.inference_thread.start()
