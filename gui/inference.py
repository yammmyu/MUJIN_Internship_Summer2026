import base64
import copy
import http.client
import json
import threading
import time

import cv2
import numpy as np

from constants import *
from smoothing import ExponentialSmoother


PC4080_HOST = "10.12.11.144"
PC4080_PORT = 9000


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
            exponential_smoother = ExponentialSmoother(alpha=0.2)
            last_inference_timestamp: float = 0.0
            smooth_step = 0.002
            self.actions = []
            while self.is_auto_inference:
                # ignore traj execution time
                now = time.time()
                if self.actions:
                    actions = self.actions[:2]
                    gripper_action = actions[-1][-1]
                    current_left_arm_joint_values = self.left_arm_joint_values
                    exponential_smoother.prev_smoothed = np.array(current_left_arm_joint_values)
                    actions = [current_left_arm_joint_values] + [exponential_smoother.smooth(a[:7]) for a in actions]
                    # actions = [exponential_smoother.smooth(a[:7]) for a in actions]
                    print(f"Executing actions: {actions}, all actions: {len(self.actions)}")
                    # self.run_trajectory(self.get_smooth_paths([self.left_arm_joint_values] + actions, smooth_step=0.0002, validate=True, validate_step=0.3)[:7], "left", 0.0001, validate=True, validate_step=0.2)
                    # moving_average or savgol
                    filter_mode = "moving_average"
                    # use_uniform_filter1d = not self.is_grabbing_target
                    # if use_uniform_filter1d:
                    #     smooth_step = 0.005 if self.is_grabbing_target else 0.005
                    # else:
                    # self.run_trajectory(self.get_smooth_paths([self.left_arm_joint_values] + actions, smooth_step=smooth_step, validate=True, validate_step=0.3)[:7], "left", 0.0001, validate=True, validate_step=0.2)
                    self.run_trajectory(self.get_smooth_paths(actions, smooth_step=smooth_step, validate=True, validate_step=0.3, filter_mode=filter_mode)[:7], "left", 0.0001, validate=True, validate_step=0.2)
                    # self.run_trajectory(actions, "left", 0.02, validate=True, validate_step=0.3)
                    if gripper_action > 0.5:
                        if not self.is_grabbing_target:
                            self.robot.move_gripper([1, 0])
                            self.is_grabbing_target = True
                            self.actions = []
                            time.sleep(1)
                            for _ in range(20):
                                self.move_arm_relative('left', [0, 0, 0.002], time_step=0.02)
                            last_inference_timestamp = time.time() + 1
                            continue
                    self.actions = self.actions[2:]
                if not self.actions:
                    x, y, z = self.left_hand_pos
                    # stop and move to home
                    if self.is_grabbing_target and x < 0.565 and z > 0.9:
                        print("stop auto inference and move back to home position")
                        self.is_auto_inference = False
                        self.run_trajectory(self.get_smooth_paths([self.left_arm_joint_values, LEFT_HAND_HOME_JOINT_VALUES]), "left", 0.01)
                        break
                    # check robot postion when there's no action
                    # TODO: test: robot is next to container
                    if self.is_grabbing_target:
                        if z < 0.9:
                            alpha = 0.3
                            smooth_step = 0.002
                        # elif x > 0.6:
                        #     alpha = 0.6
                        #     smooth_step = 0.01
                        else:
                            alpha = 0.6
                            smooth_step = 0.01
                    else:
                        if x > 0.64:
                            alpha = 0.1
                            smooth_step = 0.002
                        elif x > 0.6:
                            alpha = 0.2
                            smooth_step = 0.0025
                        elif x > 0.55:
                            alpha = 0.4
                            smooth_step = 0.005
                        else:
                            alpha = 0.3
                            smooth_step = 0.01
                    exponential_smoother.set_alpha(alpha)


                # TODO: no need to lock
                if self.robot_info.inference_timestamp <= last_inference_timestamp + 0.001:
                    if not self.actions:
                        time.sleep(0.01)
                    continue
                last_inference_timestamp = self.robot_info.inference_timestamp
                if self.is_grabbing_target and any(v[-1] < 0.5 for v in self.robot_info.left_joint_predict_action_values):
                    print(f"robot is grabbing target, skip the old inference which would release target")
                    continue
                diff = now - last_inference_timestamp
                if diff > 0.7:
                    print(f"inference diff={diff} > 0.7, not use at all")
                    continue
                if diff > 0.5:
                    print(f"inference diff={diff} > 0.5, use the last 1 action")
                    self.actions.append(self.robot_info.left_joint_predict_action_values[-1])
                elif diff > 0.3:
                    print(f"inference diff={diff} > 0.3, use last 2 actions")
                    self.actions.extend(self.robot_info.left_joint_predict_action_values[-2:])
                elif self.actions:
                    if len(self.actions) <= 4:
                        self.actions.extend(self.robot_info.left_joint_predict_action_values[-4:])
                    else:
                        self.actions = self.actions[-2:] + self.robot_info.left_joint_predict_action_values[-4:]
                else:
                    self.actions = self.robot_info.left_joint_predict_action_values

        if self.inference_thread is None:
            self.inference_thread = threading.Thread(target=_run_auto_inference, daemon=True)
            print("Starting auto_inference thread...")
            self.inference_thread.start()
        if self.execution_thread is None:
            gripper_states, _ = self.robot.gripper_states()
            self.is_grabbing_target = gripper_states[0] > 0.5
            self.execution_thread = threading.Thread(target=_run_auto_execution, daemon=True)
            print("Starting execution_thread thread...")
            self.execution_thread.start()


    def inference_once(self, use_deep_copy: bool = False):
        # TODO: left_hand only
        if self.robot_info.timestamp - self.robot_info.inference_timestamp < 0.0001:
            return False
        with self.robot_info.lock:
            last_two_left_arm_joint_values = copy.deepcopy(self.last_two_left_arm_joint_values)
            assert len(last_two_left_arm_joint_values) == 2
            inference_timestamp = self.robot_info.timestamp

        camera_images = copy.deepcopy(self.camera_images["hand_left"]) if use_deep_copy else self.camera_images["hand_left"]
        assert camera_images is not None and len(camera_images) == 2

        def _encode_image(rgb_image):
            rgb_image_cv2 = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            _, buffer = cv2.imencode('.jpg', rgb_image_cv2, [cv2.IMWRITE_JPEG_QUALITY, 90])
            base64_string = base64.b64encode(buffer).decode('utf-8')
            return base64_string

        req = {
            'left_imgs': [_encode_image(image) for image in camera_images],
            'state': last_two_left_arm_joint_values,
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

        action = np.asarray(resp['action'], dtype=np.float32)
        with self.robot_info.lock:
            self.robot_info.left_joint_predict_start_values = copy.deepcopy(last_two_left_arm_joint_values)
            self.robot_info.left_joint_predict_action_values = action.tolist()
            self.robot_info.inference_timestamp = inference_timestamp
        # print("left_joint_predict_start_values=", self.robot_info.left_joint_predict_start_values)
        # print("left_joint_predict_action_values=", self.robot_info.left_joint_predict_action_values)
        return True


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
                # DANGEROUS!!!!
                self.run_trajectory([action[:7]], "left", 0.2, validate=True, validate_step=0.4)
                self.robot.move_gripper([1 if action[-1] > 0.5 else 0, 0])
                if once:
                    break

    def start_inference_data_collection_thread(self):
        # 由本线程独占采集/缓存的相机，start_camera_thread 不再重复 cache，
        # 以免覆盖与关节时间戳精确配对的帧
        self.inference_managed_cameras.add("hand_left")

        def update_inference_data():
            camera_name = "hand_left"
            # fps = 30  # hz
            fps = 31  # hz
            joint_values_by_timestamp = []
            next_update_camera_image_timestamp = 0.0
            while True:
                now = time.time()
                # update joint values
                arm_states, _ = self.robot.arm_joint_states()
                gripper_states, _ = self.robot.gripper_states()
                joint_values_by_timestamp.append((now, arm_states[:7] + [1 if gripper_states[0] > 0.5 else 0], arm_states[7:14] + [1 if gripper_states[1] > 0.5 else 0]))
                if now < next_update_camera_image_timestamp:
                    time.sleep(0.01)
                    continue
                # update camera images
                image, timestamp = self.camera.get_latest_image(camera_name)
                if image is not None:
                    timestamp_s = timestamp * 1e-9
                    wait_time_s = 0.001
                    if len(joint_values_by_timestamp) >= 2 and timestamp_s > self.robot_info.timestamp + wait_time_s:
                        next_update_camera_image_timestamp = timestamp_s + 1.0 / fps
                        if self.camera_images[camera_name] is None:
                            self.camera_images[camera_name] = [image, image]
                        else:
                            self.camera_images[camera_name] = [self.camera_images[camera_name][-1], image]

                        for i in range(len(joint_values_by_timestamp) - 1):
                            if joint_values_by_timestamp[i][0] <= timestamp_s and joint_values_by_timestamp[i + 1][0] > timestamp_s:
                                break
                        with self.robot_info.lock:
                            self.robot_info.timestamp = timestamp_s
                            self.robot_info.left_joint_values = joint_values_by_timestamp[i][1]
                            self.robot_info.right_joint_values = joint_values_by_timestamp[i][2]

                            left_arm_state = copy.deepcopy(joint_values_by_timestamp[i][1])
                            if not self.last_two_left_arm_joint_values:
                                self.last_two_left_arm_joint_values = [left_arm_state, left_arm_state]
                            else:
                                self.last_two_left_arm_joint_values = [self.last_two_left_arm_joint_values[-1], left_arm_state]
                        joint_values_by_timestamp = joint_values_by_timestamp[-1000:]
                time.sleep(0.01)  # 10ms

        inference_data_collection_thread = threading.Thread(target=update_inference_data, daemon=True)
        inference_data_collection_thread.start()
