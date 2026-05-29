"""Motion Planning TCP 客户端 Mixin。

与外部 motion planning 服务器通过 *一个* TCP 长连接双向通信。
帧格式：4 字节大端长度 + UTF-8 编码的 JSON 负载。

GUI → Server
  - type=state            周期上报机械臂状态（关节、末端、运动标志）
  - type=detection_image  响应 detection_trigger，回传最新头部 RGB / Depth
  - type=trajectory_ack   accepted / rejected
  - type=trajectory_result completed / error

Server → GUI
  - type=detection_trigger {request_id}
  - type=trajectory       {trajectory_id, side, path[ [7], ... ], delta_time?}
"""

import base64
import json
import os
import socket
import struct
import threading
import time

import cv2
import numpy as np


# ---- 连接配置（可通过环境变量覆盖） ----
MOTION_PLANNING_SERVER_HOST = os.environ.get("MP_HOST", "127.0.0.1")
MOTION_PLANNING_SERVER_PORT = int(os.environ.get("MP_PORT", "9100"))

# 状态上报频率
STATE_PUBLISH_HZ = 10.0

# ---- 轨迹安全参数 ----
JOINT_LIMIT = 3.14               # 关节角度极限 (rad)
MAX_JOINT_STEP = 0.3             # 相邻路径点单关节最大变化 (rad)
MAX_FIRST_POINT_DEVIATION = 0.2  # 首点与当前关节最大允许偏差 (rad)
TRAJECTORY_STEP_TIME = 0.02      # 每个路径点默认执行时间 (s)

# 接收单帧的安全上限，避免恶意/错乱的长度字段把内存吃爆
_MAX_FRAME_BYTES = 50 * 1024 * 1024  # 50MB


class MotionPlanningMixin:
    """与 motion planning 服务器交互：状态上报 / 视觉触发 / 轨迹执行。"""

    # ---------- 启动 / 关闭 ----------

    def start_motion_planning_thread(self):
        """启动 TCP 客户端 + 状态上报线程。所有相关状态在此初始化。"""
        # 运动执行标志（被 _mp_handle_trajectory 维护，也供状态上报使用）
        self.is_left_arm_moving = False
        self.is_right_arm_moving = False
        self._mp_arm_locks = {
            "left": threading.Lock(),
            "right": threading.Lock(),
        }

        # 网络资源
        self._mp_sock = None
        self._mp_send_lock = threading.Lock()
        self._mp_stop = threading.Event()

        threading.Thread(target=self._mp_client_loop, daemon=True,
                         name="motion-planning-client").start()
        threading.Thread(target=self._mp_state_publish_loop, daemon=True,
                         name="motion-planning-state").start()

    def stop_motion_planning(self):
        """on_closing 兜底调用：通知线程退出并关闭 socket。"""
        self._mp_stop.set()
        self._mp_close_socket()

    # ---------- 主循环 ----------

    def _mp_client_loop(self):
        """连接 → 接收循环 → 断线重连，指数退避。"""
        host = MOTION_PLANNING_SERVER_HOST
        port = MOTION_PLANNING_SERVER_PORT
        backoff = 1.0
        while not self._mp_stop.is_set():
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5.0)
                sock.connect((host, port))
                sock.settimeout(None)
                self._mp_sock = sock
                print(f"[MP] connected to {host}:{port}")
                backoff = 1.0
                self._mp_receive_loop(sock)
            except OSError as e:
                print(f"[MP] connection error: {e}; retry in {backoff:.1f}s")
                if self._mp_stop.wait(backoff):
                    break
                backoff = min(backoff * 2, 10.0)
            finally:
                self._mp_close_socket()
        print("[MP] client loop stopped")

    def _mp_receive_loop(self, sock):
        while not self._mp_stop.is_set():
            payload = self._mp_recv_frame(sock)
            if payload is None:
                print("[MP] connection closed by peer")
                return
            try:
                obj = json.loads(payload.decode("utf-8"))
            except Exception as e:
                print(f"[MP] invalid JSON frame: {e}")
                continue
            mtype = obj.get("type")
            if mtype == "detection_trigger":
                threading.Thread(target=self._mp_handle_detection_trigger,
                                 args=(obj,), daemon=True,
                                 name="mp-detection").start()
            elif mtype == "trajectory":
                threading.Thread(target=self._mp_handle_trajectory,
                                 args=(obj,), daemon=True,
                                 name=f"mp-traj-{obj.get('side')}").start()
            else:
                print(f"[MP] unknown message type: {mtype!r}")

    def _mp_state_publish_loop(self):
        period = 1.0 / STATE_PUBLISH_HZ
        next_tick = time.monotonic()
        while not self._mp_stop.is_set():
            try:
                self._mp_publish_state()
            except Exception as e:
                print(f"[MP] state publish error: {e}")
            next_tick += period
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                if self._mp_stop.wait(sleep_for):
                    break
            else:
                next_tick = time.monotonic()

    # ---------- 帧编解码 ----------

    def _mp_send_json(self, obj):
        return self._mp_send_frame(json.dumps(obj).encode("utf-8"))

    def _mp_send_frame(self, payload):
        sock = self._mp_sock
        if sock is None:
            return False
        try:
            with self._mp_send_lock:
                sock.sendall(struct.pack(">I", len(payload)) + payload)
            return True
        except OSError as e:
            print(f"[MP] send error: {e}")
            self._mp_close_socket()
            return False

    def _mp_recv_frame(self, sock):
        head = self._mp_recv_exact(sock, 4)
        if head is None:
            return None
        (length,) = struct.unpack(">I", head)
        if length == 0 or length > _MAX_FRAME_BYTES:
            print(f"[MP] bogus frame length {length}, dropping connection")
            return None
        return self._mp_recv_exact(sock, length)

    def _mp_recv_exact(self, sock, n):
        buf = bytearray()
        while len(buf) < n:
            try:
                chunk = sock.recv(n - len(buf))
            except OSError:
                return None
            if not chunk:
                return None
            buf.extend(chunk)
        return bytes(buf)

    def _mp_close_socket(self):
        sock = self._mp_sock
        self._mp_sock = None
        if sock is not None:
            try:
                sock.close()
            except Exception:
                pass

    # ---------- 1) 状态上报 ----------

    def _mp_publish_state(self):
        if self._mp_sock is None:
            return
        try:
            arm_states, _ = self.robot.arm_joint_states()
        except Exception:
            arm_states = None
        if not arm_states or len(arm_states) < 14:
            return
        head_states = self._mp_safe_call(self.robot.head_joint_states)
        waist_states = self._mp_safe_call(self.robot.waist_joint_states)
        msg = {
            "type": "state",
            "timestamp": time.time(),
            "arm_joint_values": list(arm_states[:14]),
            "head_joint_values": list(head_states[:2]) if head_states else None,
            "waist_joint_values": list(waist_states[:2]) if waist_states else None,
            "left_ee_position": self._mp_get_ee_pos("arm_left_link7"),
            "right_ee_position": self._mp_get_ee_pos("arm_right_link7"),
            "left_arm_moving": bool(self.is_left_arm_moving),
            "right_arm_moving": bool(self.is_right_arm_moving),
        }
        self._mp_send_json(msg)

    @staticmethod
    def _mp_safe_call(fn):
        try:
            states, _ = fn()
            return states
        except Exception:
            return None

    def _mp_get_ee_pos(self, frame_name):
        try:
            pos = self.robot_controller.get_motion_status()['frames'][frame_name]['position']
            return [pos['x'], pos['y'], pos['z']]
        except Exception:
            return None

    # ---------- 2) Detection 触发 → 回传 RGB + Depth ----------

    def _mp_handle_detection_trigger(self, obj):
        req_id = obj.get("request_id")
        rgb = self._latest_camera_frame("head")
        depth = self._latest_camera_frame("head_depth")
        intr = self.camera_intrinsics.get("head") or {}
        rgb_intrinsics = None
        if intr:
            rgb_intrinsics = {
                "fx": intr.get("fx"), "fy": intr.get("fy"),
                "cx": intr.get("cx"), "cy": intr.get("cy"),
                "width": intr.get("width"), "height": intr.get("height"),
            }
        msg = {
            "type": "detection_image",
            "request_id": req_id,
            "timestamp": time.time(),
            "rgb_jpeg_base64": self._mp_encode_rgb_jpeg(rgb),
            "depth_png_base64": self._mp_encode_depth_png(depth),
            "rgb_intrinsics": rgb_intrinsics,
            "depth_scale": 0.001,  # uint16 mm -> m
        }
        self._mp_send_json(msg)

    def _mp_encode_rgb_jpeg(self, image):
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            return None
        try:
            if image.ndim == 3 and image.shape[2] == 3:
                # 与 inference_once 一致：先 RGB→BGR 再 JPEG 编码
                bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
                if ok:
                    return base64.b64encode(buf).decode("ascii")
        except Exception as e:
            print(f"[MP] rgb encode error: {e}")
        return None

    def _mp_encode_depth_png(self, image):
        """深度用 PNG 无损保存，保留 uint16 精度。"""
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            return None
        try:
            arr = image
            if arr.ndim == 3 and arr.shape[2] == 1:
                arr = arr[:, :, 0]
            ok, buf = cv2.imencode(".png", arr)
            if ok:
                return base64.b64encode(buf).decode("ascii")
        except Exception as e:
            print(f"[MP] depth encode error: {e}")
        return None

    # ---------- 3) 轨迹执行 ----------

    def _mp_handle_trajectory(self, obj):
        side = obj.get("side")
        path = obj.get("path")
        delta_time = float(obj.get("delta_time", TRAJECTORY_STEP_TIME))
        traj_id = obj.get("trajectory_id")

        ok, reason = self._mp_validate_trajectory(side, path)
        if not ok:
            print(f"[MP] reject trajectory {traj_id}: {reason}")
            self._mp_send_json({
                "type": "trajectory_ack",
                "trajectory_id": traj_id,
                "status": "rejected",
                "reason": reason,
            })
            return

        # 同一只手不允许并发；左右手互不阻塞
        lock = self._mp_arm_locks[side]
        if not lock.acquire(blocking=False):
            print(f"[MP] reject trajectory {traj_id}: {side} arm busy")
            self._mp_send_json({
                "type": "trajectory_ack",
                "trajectory_id": traj_id,
                "status": "rejected",
                "reason": f"{side}_arm_busy",
            })
            return

        self._mp_send_json({
            "type": "trajectory_ack",
            "trajectory_id": traj_id,
            "status": "accepted",
            "num_points": len(path),
        })

        self._mp_set_moving(side, True)
        status, err = "completed", None
        try:
            # run_trajectory 已在 PickPlaceMixin 中实现：发送 ABS_JOINT 序列
            self.run_trajectory(
                path, side, delta_time,
                validate=True, validate_step=MAX_FIRST_POINT_DEVIATION)
        except AssertionError as e:
            status, err = "rejected", f"validate failed: {e}"
            print(f"[MP] trajectory {traj_id} validate failed: {e}")
        except Exception as e:
            status, err = "error", str(e)
            print(f"[MP] trajectory {traj_id} exec error: {e}")
        finally:
            self._mp_set_moving(side, False)
            lock.release()

        self._mp_send_json({
            "type": "trajectory_result",
            "trajectory_id": traj_id,
            "status": status,
            "reason": err,
        })

    def _mp_set_moving(self, side, value):
        if side == "left":
            self.is_left_arm_moving = value
        elif side == "right":
            self.is_right_arm_moving = value

    def _mp_validate_trajectory(self, side, path):
        if side not in ("left", "right"):
            return False, f"invalid side: {side!r}"
        if not isinstance(path, list) or not path:
            return False, "empty path"
        for i, node in enumerate(path):
            if not isinstance(node, list) or len(node) != 7:
                return False, f"node[{i}] must be list of 7 joints"
            if not all(isinstance(v, (int, float)) for v in node):
                return False, f"node[{i}] contains non-numeric"
            if any(abs(v) > JOINT_LIMIT for v in node):
                return False, f"node[{i}] exceeds joint limit ±{JOINT_LIMIT}"
            if i > 0:
                prev = path[i - 1]
                if any(abs(node[j] - prev[j]) > MAX_JOINT_STEP for j in range(7)):
                    return False, (
                        f"step {i} exceeds max joint step {MAX_JOINT_STEP}")
        # 首点与当前关节偏差
        try:
            arm_states, _ = self.robot.arm_joint_states()
        except Exception as e:
            return False, f"state read failed: {e}"
        if not arm_states or len(arm_states) < 14:
            return False, "no arm state available"
        current = arm_states[:7] if side == "left" else arm_states[7:14]
        if any(abs(path[0][j] - current[j]) > MAX_FIRST_POINT_DEVIATION
               for j in range(7)):
            return False, "first point too far from current pose"
        return True, None
