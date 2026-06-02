#!/usr/bin/env python3
"""
Send one real recording (from MDM_data_collection/recordings) to the
left_arm_ee_image inference server and print what comes back.

Builds the obs the same way the live GUI does:
  agent_imgs = head camera, n_obs_steps frames     (-> agentview_image)
  hand_imgs  = hand_left camera, n_obs_steps frames (-> robot0_eye_in_hand_image)
  state      = [left_pos(3), left_quat xyzw(4), gripper(1)]  per frame, shape (To, 8)

Needs numpy + cv2 (i.e. run it in the `robodiff` conda env, e.g. on the server box).

Usage:
    python3 send_recording_to_server.py \
        --recording MDM_data_collection/recordings/recording001 \
        --host 127.0.0.1 --port 9001 --index 0 --obs_steps 2
"""

import argparse
import base64
import http.client
import json
import os
import time

import cv2
import numpy as np


def read_frames(mp4_path: str, indices):
    """Return {idx: BGR frame} for the requested frame indices."""
    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        raise FileNotFoundError(f'cannot open {mp4_path}')
    want = set(indices)
    out, i = {}, 0
    while want:
        ok, frame = cap.read()
        if not ok:
            break
        if i in want:
            out[i] = frame
            want.discard(i)
        i += 1
    cap.release()
    missing = set(indices) - set(out)
    if missing:
        raise IndexError(f'{mp4_path}: frames {sorted(missing)} not found '
                         f'(video has {i} frames)')
    return out


def encode_jpg(bgr_frame) -> str:
    # VideoCapture already yields BGR; the server does imdecode->BGR2RGB, so
    # encoding the BGR frame directly reproduces the original RGB on the server.
    ok, buf = cv2.imencode('.jpg', bgr_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError('cv2.imencode failed')
    return base64.b64encode(buf).decode('utf-8')


def post_predict(host, port, req, timeout):
    body = json.dumps(req).encode('utf-8')
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        conn.request('POST', '/predict', body=body, headers={
            'Content-Type': 'application/json; charset=utf-8',
            'Content-Length': str(len(body)),
        })
        resp = conn.getresponse()
        raw = resp.read()
        try:
            obj = json.loads(raw.decode('utf-8')) if raw else {}
        except Exception as e:
            return {'error': f'bad JSON (status={resp.status}): {e!r}'}
        if resp.status != 200:
            return obj if isinstance(obj, dict) and 'error' in obj \
                else {'error': f'HTTP {resp.status}: {obj!r}'}
        return obj
    finally:
        conn.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--recording', required=True, help='path to a recordingNNN/ dir')
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=9001)
    ap.add_argument('--index', type=int, default=0,
                    help='first of the obs_steps consecutive frames to send')
    ap.add_argument('--obs_steps', type=int, default=2,
                    help='n_obs_steps (server reported 2)')
    ap.add_argument('--timeout', type=float, default=30.0)
    args = ap.parse_args()

    rec = args.recording
    npz = np.load(os.path.join(rec, 'robot_states.npz'))
    left_pos, left_quat, gripper = npz['left_pos'], npz['left_quat'], npz['gripper']
    n = len(left_pos)
    idxs = list(range(args.index, args.index + args.obs_steps))
    if idxs[-1] >= n:
        raise IndexError(f'index range {idxs} out of bounds (n_frames={n})')

    # state rows: [pos(3), quat xyzw(4), gripper(1)]
    state = [
        [*map(float, left_pos[i]), *map(float, left_quat[i]), float(gripper[i][0])]
        for i in idxs
    ]

    head_frames = read_frames(os.path.join(rec, 'cameras', 'head.mp4'), idxs)
    hand_frames = read_frames(os.path.join(rec, 'cameras', 'hand_left.mp4'), idxs)

    req = {
        'agent_imgs': [encode_jpg(head_frames[i]) for i in idxs],
        'hand_imgs':  [encode_jpg(hand_frames[i]) for i in idxs],
        'state':      state,
    }

    print(f'recording = {rec}  (n_frames={n})')
    print(f'sending frames {idxs}')
    print('state rows [pos(3), quat(4), grip(1)]:')
    for i, row in zip(idxs, state):
        print(f'  f{i}: pos={[round(v,4) for v in row[0:3]]} '
              f'quat={[round(v,4) for v in row[3:7]]} grip={round(row[7],3)}')
    print(f'POST http://{args.host}:{args.port}/predict ...')

    t0 = time.time()
    resp = post_predict(args.host, args.port, req, args.timeout)
    rtt = time.time() - t0

    if 'error' in resp:
        print(f'\n*** Server error: {resp["error"]}')
        return

    action = np.asarray(resp['action'], dtype=np.float32)
    action_pred = np.asarray(resp.get('action_pred', []), dtype=np.float32)
    np.set_printoptions(precision=4, suppress=True, linewidth=140)
    print(f'\nOK  round-trip={rtt:.3f}s  server inference_time_s={resp.get("inference_time_s")}')
    print(f'action shape      = {action.shape}   (expect (n_action_steps, 10))')
    print(f'action_pred shape = {action_pred.shape} (expect (horizon, 10))')
    print('\naction chunk [eef_pos(3), 6D_rot(6), gripper(1)]:')
    print(action)
    row = action[0]
    print('\nfirst step:')
    print(f'  eef_pos = {row[0:3]}')
    print(f'  6D_rot  = {row[3:9]}')
    print(f'  gripper = {row[9]:.3f} -> {"CLOSE" if row[9] > 0.5 else "OPEN"}')


if __name__ == '__main__':
    main()
