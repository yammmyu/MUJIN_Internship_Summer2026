#!/usr/bin/env python3
"""
Round-trip smoke test for the left_arm_ee_image inference server.

Fabricates a valid /predict request (2 synthetic PNGs per camera + a (2,8) EE
state) and prints what comes back. Pure standard library -- no numpy / cv2 / PIL --
so it runs anywhere, including the bare server machine.

Usage:
    # on the SAME machine as the server (simplest):
    python3 ping_inference_server.py --host 127.0.0.1 --port 9001

    # from another machine:
    python3 ping_inference_server.py --host 10.12.11.144 --port 9001
"""

import argparse
import base64
import http.client
import json
import struct
import time
import zlib


# ---- minimal PNG encoder (stdlib only) -------------------------------------

def _png_chunk(tag: bytes, data: bytes) -> bytes:
    return (struct.pack('>I', len(data)) + tag + data
            + struct.pack('>I', zlib.crc32(tag + data) & 0xffffffff))


def make_png(w: int, h: int, rgb=(127, 127, 127)) -> bytes:
    """Solid-colour RGB PNG, w x h."""
    sig = b'\x89PNG\r\n\x1a\n'
    ihdr = struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0)  # 8-bit, colour type 2 (RGB)
    row = b'\x00' + bytes(rgb) * w                        # filter byte 0 + pixels
    idat = zlib.compress(row * h)
    return sig + _png_chunk(b'IHDR', ihdr) + _png_chunk(b'IDAT', idat) + _png_chunk(b'IEND', b'')


def b64png(color=(120, 120, 120), w=84, h=84) -> str:
    return base64.b64encode(make_png(w, h, color)).decode('ascii')


# ---- request / transport ---------------------------------------------------

def build_request() -> dict:
    # state row = [eef_pos(3), eef_quat xyzw(4), gripper(1)]; identity orientation.
    state_row = [0.5, 0.0, 0.9, 0.0, 0.0, 0.0, 1.0, 0.0]
    return {
        'agent_imgs': [b64png((40, 80, 160)), b64png((40, 80, 160))],   # head
        'hand_imgs':  [b64png((160, 80, 40)), b64png((160, 80, 40))],   # hand_left
        'state':      [state_row, state_row],                            # (2, 8)
    }


def post_predict(host: str, port: int, req: dict, timeout: float) -> dict:
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


def shape(x):
    if isinstance(x, list) and x and isinstance(x[0], list):
        return (len(x), len(x[0]))
    return (len(x),) if isinstance(x, list) else ()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=9001)
    ap.add_argument('--timeout', type=float, default=30.0)
    args = ap.parse_args()

    req = build_request()
    print(f'POST http://{args.host}:{args.port}/predict  '
          f'(agent_imgs={shape(req["agent_imgs"])}, hand_imgs={shape(req["hand_imgs"])}, '
          f'state={shape(req["state"])})')

    t0 = time.time()
    resp = post_predict(args.host, args.port, req, args.timeout)
    rtt = time.time() - t0

    if 'error' in resp:
        print(f'\n*** Server error: {resp["error"]}')
        return

    action = resp['action']
    action_pred = resp.get('action_pred', [])
    print(f'\nOK  round-trip={rtt:.3f}s  server inference_time_s={resp.get("inference_time_s")}')
    print(f'action shape      = {shape(action)}   (expect (n_action_steps, 10))')
    print(f'action_pred shape = {shape(action_pred)} (expect (horizon, 10))')

    row = action[0]
    print('\nfirst action row [eef_pos(3), 6D_rot(6), gripper(1)]:')
    print(f'  eef_pos = {[round(v, 4) for v in row[0:3]]}')
    print(f'  6D_rot  = {[round(v, 4) for v in row[3:9]]}')
    print(f'  gripper = {round(row[9], 4)}  -> {"CLOSE" if row[9] > 0.5 else "OPEN"}')


if __name__ == '__main__':
    main()
