#!/usr/bin/env python3
"""Standalone webcam tester for the printed-LABEL detector (real_world/no_flip_place.LabelGate).

Checks whether the no-flip label detection fires on a LIVE camera on this machine, and lets you tune
the thresholds on real packages before touching the robot. It uses the SAME LabelGate the robot uses
(loaded from real_world/no_flip_place.py), so what works here works there.

Usage:
    python scripts/label_webcam_test.py                       # camera 0, 30 s
    python scripts/label_webcam_test.py --show                # draw detected label blocks live
    python scripts/label_webcam_test.py --min-fill 0.45 --min-char-comps 8   # tune thresholds

Hold a labelled package (and a plain/wrinkled bag) in front of the camera and watch which one trips it.
On the first detection it saves ./label_hit.png (with the block drawn); if nothing ever detects it
saves ./label_nodetect.png so you can eyeball what the camera saw.
"""
import argparse
import importlib.util
import os
import time

import cv2

# Load LabelGate directly from the module file (no package import -> no SDK/ROS deps pulled in).
_NFP = os.path.join(os.path.dirname(__file__), "..", "real_world", "no_flip_place.py")
_spec = importlib.util.spec_from_file_location("no_flip_place", _NFP)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
LabelGate = _mod.LabelGate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--secs", type=float, default=30.0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--show", action="store_true", help="draw detected label blocks (needs a display)")
    # LabelGate thresholds (calibrate on real packages)
    ap.add_argument("--min-area-frac", type=float, default=0.003)
    ap.add_argument("--min-fill", type=float, default=0.50)
    ap.add_argument("--min-edge-density", type=float, default=0.10)
    ap.add_argument("--min-char-comps", type=int, default=6)
    ap.add_argument("--max-aspect", type=float, default=6.0)
    args = ap.parse_args()

    gate = LabelGate(min_area_frac=args.min_area_frac, min_fill=args.min_fill,
                     min_edge_density=args.min_edge_density, min_char_comps=args.min_char_comps,
                     max_aspect=args.max_aspect)

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise SystemExit(f"cannot open camera {args.cam} (try --cam 1)")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    aw, ah = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"camera {args.cam} open at {aw}x{ah}. Hold a LABELLED package in front of it. "
          f"running {args.secs:.0f}s (Ctrl-C to stop)… thresholds: fill>={args.min_fill} "
          f"edge_density>={args.min_edge_density} char_comps>={args.min_char_comps} aspect<={args.max_aspect}")
    t0 = time.time()
    last_status = 0.0
    frames = hits = 0
    saved_hit = False
    last_no = None

    try:
        while time.time() - t0 < args.secs:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            frames += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blocks = gate.find_label_blocks(gray)
            if blocks:
                hits += 1
                fw = frame.shape[1]
                x, y, w, h, comps = max(blocks, key=lambda b: b[2] * b[3])
                print(f"[{time.time()-t0:5.1f}s] LABEL block {w}x{h}px "
                      f"({100*w/fw:.0f}% of frame width, {comps} chars)  [{len(blocks)} block(s)]")
                if args.show or not saved_hit:
                    for bx, by, bw, bh, _ in blocks:
                        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 200, 0), 2)
                if not saved_hit:
                    cv2.imwrite("label_hit.png", frame)
                    saved_hit = True
                    print("           saved frame -> ./label_hit.png")
            else:
                last_no = frame

            if args.show:
                cv2.imshow("label test (Esc to quit)", frame)
                if (cv2.waitKey(1) & 0xFF) == 27:
                    break

            now = time.time()
            if now - last_status >= 1.0:
                last_status = now
                h, w = frame.shape[:2]
                print(f"[{now-t0:5.1f}s] frames={frames} res={w}x{h} detections={hits}")
    except KeyboardInterrupt:
        print("\ninterrupted.")
    finally:
        cap.release()
        if args.show:
            cv2.destroyAllWindows()

    if last_no is not None and not saved_hit:
        cv2.imwrite("label_nodetect.png", last_no)
        print("saved a no-detection sample frame -> ./label_nodetect.png (inspect it)")

    print(f"\ndone. frames={frames}, detections={hits}")
    if hits == 0:
        print("=> NO label detected. If a real label was in view, loosen thresholds: lower --min-fill / "
              "--min-char-comps / --min-edge-density, or hold the label larger/steadier. A plain bag "
              "SHOULD read 0 — that's correct.")
    else:
        print("=> Detection WORKS. Note the '% of frame width' — that's how big the label must appear on "
              "the head camera. If wrinkles/tape/glare also trip it (false positives), RAISE --min-fill "
              "/ --min-char-comps.")


if __name__ == "__main__":
    main()
