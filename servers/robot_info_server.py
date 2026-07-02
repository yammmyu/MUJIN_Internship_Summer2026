import argparse
import dataclasses
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

@dataclasses.dataclass
class RobotInfo:
    lock = threading.Lock()
    timestamp: float = 0.0
    left_joint_values: list[float] = None  # 7 joint values + 1 gripper joint value
    right_joint_values: list[float] = None  # 7 joint values + 1 gripper joint value
    # NOTE: for the left_arm_ee_image policy these left_*_predict_* fields now hold
    # EE-pose data, not joints: *_start_values = last_two left EE states
    # ([pos(3), quat(4), grip(1)]); *_action_values = action rows
    # ([eef_pos(3), 6D_rot(6), grip(1)]).
    left_joint_predict_start_values: list[list[float]] = None
    right_joint_predict_start_values: list[list[float]] = None
    left_joint_predict_action_values: list[list[float]] = None
    right_joint_predict_action_values: list[list[float]] = None
    inference_timestamp: float = 0.0


# ---------------------------- request handling -------------------------- #


def handle_robot_info_request(req: dict, robot_info: RobotInfo) -> dict:
    # numpy -> nested lists for JSON
    return {
        'left_joint_values': robot_info.left_joint_values,
        'right_joint_values': robot_info.right_joint_values,
        'left_joint_predict_start_values': robot_info.left_joint_predict_start_values,
        'right_joint_predict_start_values': robot_info.right_joint_predict_start_values,
        'left_joint_predict_action_values': robot_info.left_joint_predict_action_values,
        'right_joint_predict_action_values': robot_info.right_joint_predict_action_values,
    }


# ---------------------------- HTTP plumbing ----------------------------- #

def create_robot_handler(robot_info: RobotInfo):

    class RobotInfoHandler(BaseHTTPRequestHandler):
        """Exposes GET /robot_info."""

        def __init__(self, *args, **kwargs):
            self.server_version = 'RobotInfoHTTP/1.1'
            self.robot_info = robot_info
            super().__init__(*args, **kwargs)

        def _send_json(self, status: int, obj) -> None:
            body = json.dumps(obj).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path != '/robot_info':
                self._send_json(404, {'error': f'unknown path {self.path}'})
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                req = {}
                if length > 0:
                    raw = self.rfile.read(length)
                    req = json.loads(raw.decode('utf-8'))
                    if not isinstance(req, dict):
                        raise ValueError('request body must be a JSON object')
            except Exception as e:
                print(f'  ERROR (parse): {e!r}')
                self._send_json(400, {'error': f'bad request: {e!r}'})
                return

            try:
                resp = handle_robot_info_request(
                    req,
                    self.robot_info
                )
            except Exception as e:
                print(f'  ERROR (inference): {e!r}')
                self._send_json(500, {'error': repr(e)})
                return

            self._send_json(200, resp)

        def log_message(self, fmt, *args):
            # print(f'[{self.address_string()}] {fmt % args}')
            pass

    return RobotInfoHandler


# ---------------------------- entrypoint -------------------------------- #

def create_robot_info_http_server(robot_info: RobotInfo, host: str = "0.0.0.0", port: int = 9000):
    def _run():
        httpd = ThreadingHTTPServer((host, port), create_robot_handler(robot_info))
        print(f'HTTP/JSON listening on http://{host}:{port}/robot_info  '
              f'(Ctrl-C to stop)')
        try:
            httpd.serve_forever()
        finally:
            httpd.server_close()
    t = threading.Thread(target=_run, daemon=True)
    t.start()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='0.0.0.0')
    ap.add_argument('--port', type=int, default=9000)
    args = ap.parse_args()

    robot_info = RobotInfo()
    create_robot_info_http_server(robot_info, args.host, args.port)
    from IPython import embed
    embed()


if __name__ == '__main__':
    main()
