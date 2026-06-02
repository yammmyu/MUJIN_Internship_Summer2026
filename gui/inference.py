import base64
import copy
import http.client
import json
import threading
import time

import cv2
import numpy as np

from constants import *


CAMERA_NAMES = ["hand_left", "hand_right", "head"]
RECORD_HZ = 30
PC4080_HOST = "10.12.11.144"
PC4080_PORT = 9001

# Camera roles for the left_arm_ee_image policy obs.
AGENT_CAMERA = "head"        # -> agentview_image
HAND_CAMERA = "hand_left"    # -> robot0_eye_in_hand_image
INFERENCE_CAMERAS = [AGENT_CAMERA, HAND_CAMERA]


def rot6d_to_quat(rot6d):
    """6D rotation (first two rotation-matrix columns) -> quaternion [x, y, z, w].

    Mirrors diffusion_policy/scripts/build_left_arm_ee_replay_buffer.py:_6d_rot_to_quat
    for a single sample, so the live decode matches how the training data was built.
    """
    rot6d = np.asarray(rot6d, dtype=np.float64)
    c1 = rot6d[:3]
    c2 = rot6d[3:6]
    # Gram-Schmidt orthonormalisation
    c1 = c1 / (np.linalg.norm(c1) + 1e-8)
    c2 = c2 - np.dot(c2, c1) * c1
    c2 = c2 / (np.linalg.norm(c2) + 1e-8)
    c3 = np.cross(c1, c2)
    m = np.stack([c1, c2, c3], axis=-1)  # rotation matrix (columns = c1,c2,c3)
    t = m[0, 0] + m[1, 1] + m[2, 2]
    if t > 0:
        s = 0.5 / np.sqrt(t + 1.0)
        q = [(m[2, 1] - m[1, 2]) * s, (m[0, 2] - m[2, 0]) * s,
             (m[1, 0] - m[0, 1]) * s, 0.25 / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = [0.25 * s, (m[0, 1] + m[1, 0]) / s,
             (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s]
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = [(m[0, 1] + m[1, 0]) / s, 0.25 * s,
             (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s]
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = [(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s,
             0.25 * s, (m[0, 1] - m[1, 0]) / s]
    return [float(v) for v in q]


class InferenceMixin:
    """推理：数据采集线程、单步/自动推理与推理轨迹执行。"""

    def auto_inference(self, stop: bool = False):
        if stop:
            self.is_auto_inference = False
            if self.execution_thread is not None:
                print("Stopping execution_thread thread...")
                self.execution_thread.join()
                self.execution_thread = None
            if self.inference_thread is not None:
                print("Stopping auto_inference thread...")
                self.inference_thread.join()
                self.inference_thread = None
            return
        self.is_auto_inference = True

        def _run_auto_inference():
            while self.is_auto_inference:
                # TODO: test
                if self.inference_once(use_deep_copy=True):
                    time.sleep(0.03)
                else:
                    time.sleep(0.01)

        def _run_auto_execution():
            # Direct EE-pose execution: consume the latest predicted action chunk,
            # command each absolute pose + gripper, paced at the action rate.
            pos_alpha = 0.5            # position low-pass (1.0 = no smoothing)
            dt = 1.0 / RECORD_HZ
            last_inference_timestamp: float = 0.0
            smoothed_pos = None
            self.actions = []
            while self.is_auto_inference:
                # Refresh the action queue whenever a newer inference arrives.
                if self.robot_info.inference_timestamp > last_inference_timestamp + 1e-3:
                    last_inference_timestamp = self.robot_info.inference_timestamp
                    with self.robot_info.lock:
                        preds = self.robot_info.left_joint_predict_action_values
                        self.actions = copy.deepcopy(preds) if preds else []

                if not self.actions:
                    time.sleep(dt)
                    continue

                action = self.actions.pop(0)
                pos, quat, grip = self._decode_ee_action(action)

                pos = np.asarray(pos, dtype=np.float64)
                if smoothed_pos is None:
                    smoothed_pos = pos
                else:
                    smoothed_pos = pos_alpha * pos + (1.0 - pos_alpha) * smoothed_pos

                self._command_left_ee(smoothed_pos, quat, grip)
                time.sleep(dt)

        if self.inference_thread is None:
            self.inference_thread = threading.Thread(target=_run_auto_inference, daemon=True)
            print("Starting auto_inference thread...")
            self.inference_thread.start()
        if self.execution_thread is None:
            self.execution_thread = threading.Thread(target=_run_auto_execution, daemon=True)
            print("Starting execution_thread thread...")
            self.execution_thread.start()


    def inference_once(self, use_deep_copy: bool = False):
        # left_arm_ee_image policy: obs = head + hand_left images + left EE state.
        if self.robot_info.timestamp - self.robot_info.inference_timestamp < 0.0001:
            return False
        with self.robot_info.lock:
            last_two_left_ee_states = copy.deepcopy(self.last_two_left_ee_states)
            assert len(last_two_left_ee_states) == 2
            inference_timestamp = self.robot_info.timestamp

        agent_frames = self.camera_images[AGENT_CAMERA]
        hand_frames = self.camera_images[HAND_CAMERA]
        if use_deep_copy:
            agent_frames = copy.deepcopy(agent_frames)
            hand_frames = copy.deepcopy(hand_frames)
        assert agent_frames is not None and len(agent_frames) == 2
        assert hand_frames is not None and len(hand_frames) == 2

        def _encode_image(rgb_image):
            rgb_image_cv2 = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            _, buffer = cv2.imencode('.jpg', rgb_image_cv2, [cv2.IMWRITE_JPEG_QUALITY, 90])
            base64_string = base64.b64encode(buffer).decode('utf-8')
            return base64_string

        req = {
            'agent_imgs': [_encode_image(image) for image in agent_frames],
            'hand_imgs': [_encode_image(image) for image in hand_frames],
            'state': last_two_left_ee_states,
        }

        def post_predict(host: str, port: int, req: dict, timeout: float = 60.0) -> dict:
            body = json.dumps(req).encode('utf-8')
            conn = http.client.HTTPConnection(host, port, timeout=timeout)
            try:
                conn.request(
                    'POST', '/predict', body=body,
                    headers={
                        'Content-Type': 'application/json; charset=utf-8',
                        'Content-Length': str(len(body)),
                    })
                resp = conn.getresponse()
                resp_body = resp.read()
                try:
                    resp_obj = json.loads(resp_body.decode('utf-8')) if resp_body else {}
                except Exception as e:
                    return {'error': f'failed to parse JSON response: {e!r} '
                                     f'(status={resp.status})'}
                if resp.status != 200:
                    if isinstance(resp_obj, dict) and 'error' in resp_obj:
                        return resp_obj
                    return {'error': f'HTTP {resp.status}: {resp_obj!r}'}
                return resp_obj
            finally:
                conn.close()

        resp = post_predict(PC4080_HOST, PC4080_PORT, req, timeout=10)

        if 'error' in resp:
            print(f'\nServer error: {resp["error"]}')
            return

        # NOTE: left_joint_predict_* now carry EE-pose data, not joints:
        #   *_start_values   = last_two_left_ee_states ([pos(3), quat(4), grip(1)])
        #   *_action_values  = predicted action rows [eef_pos(3), 6D_rot(6), grip(1)]
        action = np.asarray(resp['action'], dtype=np.float32)
        with self.robot_info.lock:
            self.robot_info.left_joint_predict_start_values = copy.deepcopy(last_two_left_ee_states)
            self.robot_info.left_joint_predict_action_values = action.tolist()
            self.robot_info.inference_timestamp = inference_timestamp
        return True


    def _command_left_ee(self, pos, quat, gripper, lifetime: float = 2.0):
        """Command the left arm to an absolute EE pose + set the gripper.

        pos:  (x, y, z) metres; quat: (qx, qy, qz, qw); gripper: scalar (>0.5 -> close).
        Uses the same SDK path as gui/manual_control.py:move_arm_to_position.
        """
        pose_dict = {
            'x': float(pos[0]), 'y': float(pos[1]), 'z': float(pos[2]),
            'qx': float(quat[0]), 'qy': float(quat[1]),
            'qz': float(quat[2]), 'qw': float(quat[3]),
        }
        self.robot_controller.set_end_effector_pose_control(
            lifetime=lifetime,
            control_group=["left_arm"],
            left_pose=pose_dict,
            right_pose=None,
        )
        self.move_gripper("left", 1.0 if gripper > 0.5 else 0.0)

    def _decode_ee_action(self, action):
        """action row [eef_pos(3), 6D_rot(6), gripper(1)] -> (pos(3), quat(4), grip)."""
        pos = action[:3]
        quat = rot6d_to_quat(action[3:9])
        grip = action[9]
        return pos, quat, grip

    def execute_inference_result(self, once: bool = False):
        def safety_check():
            return True
        with self.robot_info.lock:
            while self.robot_info.left_joint_predict_action_values:
                action = self.robot_info.left_joint_predict_action_values.pop(0)
                if not safety_check():
                    self.robot_info.left_joint_predict_start_values = None
                    self.robot_info.left_joint_predict_action_values = None
                    break
                # DANGEROUS!!!!  absolute EE-pose command
                pos, quat, grip = self._decode_ee_action(action)
                self._command_left_ee(pos, quat, grip)
                time.sleep(1.0 / RECORD_HZ)
                if once:
                    break

    def start_inference_data_collection_thread(self):
        """采集线程：以固定频率 (RECORD_HZ) 同步采集 head/hand_left 相机帧与左臂末端
        位姿，写入 self.camera_images / self.robot_info / self.last_two_left_ee_states
        供 left_arm_ee_image 策略推理使用。

        由本线程独占采集/缓存 INFERENCE_CAMERAS 中的相机，start_camera_thread 不再
        重复 cache，以免覆盖供推理使用的帧。
        """
        if getattr(self, "_inference_collection_thread", None) is not None:
            print("Inference data collection thread already running.")
            return

        self.inference_managed_cameras.update(INFERENCE_CAMERAS)
        for name in INFERENCE_CAMERAS:
            self.camera_images.setdefault(name, None)
        if getattr(self, "last_two_left_ee_states", None) is None:
            self.last_two_left_ee_states = []

        self._inference_stop_event = threading.Event()
        interval = 1.0 / RECORD_HZ

        def update_inference_data():
            next_tick = time.monotonic()
            while not self._inference_stop_event.is_set():
                now = time.time()

                # --- 左臂末端位姿 + 夹爪 -> state = [pos(3), quat xyzw(4), grip(1)] ---
                frame = self.robot_controller.get_motion_status()['frames']['arm_left_link7']
                pos = frame['position']
                quat = frame['orientation']['quaternion']
                gripper_states, _ = self.robot.gripper_states()
                left_ee_state = [
                    pos['x'], pos['y'], pos['z'],
                    quat['x'], quat['y'], quat['z'], quat['w'],
                    1.0 if gripper_states[0] > 0.5 else 0.0,
                ]

                # --- 相机帧：每个相机滚动保留最近两帧 ---
                for name in INFERENCE_CAMERAS:
                    image, _ = self.camera.get_latest_image(name)
                    if image is None:
                        continue
                    prev = self.camera_images.get(name)
                    self.camera_images[name] = [prev[-1], image] if prev else [image, image]

                # --- 发布给推理：末端位姿与相机在同一 tick 共采样 ---
                with self.robot_info.lock:
                    self.robot_info.timestamp = now
                    if self.last_two_left_ee_states:
                        self.last_two_left_ee_states = [
                            self.last_two_left_ee_states[-1], left_ee_state,
                        ]
                    else:
                        self.last_two_left_ee_states = [left_ee_state, left_ee_state]

                # --- 限速到 RECORD_HZ（stop_event 可立即唤醒）---
                next_tick += interval
                sleep_for = next_tick - time.monotonic()
                if sleep_for > 0:
                    self._inference_stop_event.wait(sleep_for)
                else:
                    next_tick = time.monotonic()

        self._inference_collection_thread = threading.Thread(
            target=update_inference_data, daemon=True,
        )
        print("Starting inference data collection thread...")
        self._inference_collection_thread.start()
