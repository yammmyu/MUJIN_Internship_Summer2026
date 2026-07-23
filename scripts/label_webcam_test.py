#!/usr/bin/env python3
"""Standalone webcam tester for the no-flip YOLO detector (real_world/no_flip_place.YoloGate).

Checks whether the no-flip detection fires on a LIVE camera on this machine, and lets you sanity-check
the trained model (and its confidence threshold) on real packages before touching the robot. It uses
the SAME YoloGate the robot uses (loaded from real_world/no_flip_place.py), so what works here works
there.

Usage:
    python scripts/label_webcam_test.py --weights path/to/best.pt            # camera 0, 30 s
    python scripts/label_webcam_test.py --weights best.pt --show             # draw boxes live
    python scripts/label_webcam_test.py --weights best.pt --conf 0.4         # tune the threshold

Hold a labelled package (and a plain/wrinkled bag) in front of the camera and watch which one trips it.
On the first detection it saves ./label_hit.png (with boxes drawn); if nothing ever detects it saves
./label_nodetect.png so you can eyeball what the camera saw. Requires `pip install ultralytics` and a
trained .pt — without either, the gate loads but never fires (it prints why).
"""
import argparse
import importlib.util
import os
import time

import cv2

# Load YoloGate directly from the module file (no package import -> no SDK/ROS deps pulled in).
_NFP = os.path.join(os.path.dirname(__file__), "..", "real_world", "no_flip_place.py")
_spec = importlib.util.spec_from_file_location("no_flip_place", _NFP)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
YoloGate = _mod.YoloGate
DEFAULT_WEIGHTS = _mod.DEFAULT_WEIGHTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--secs", type=float, default=30.0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--show", action="store_true", help="draw detected boxes live (needs a display)")
    # YoloGate params (calibrate on real packages)
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS, help="trained .pt model to load")
    ap.add_argument("--conf", type=float, default=0.25, help="min box confidence to count")
    ap.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    ap.add_argument("--imgsz", type=int, default=640, help="inference image size (multiple of 32)")
    ap.add_argument("--device", default=None, help="e.g. 'cpu' or '0' (default: auto)")
    args = ap.parse_args()

    gate = YoloGate(weights=args.weights, conf=args.conf, iou=args.iou, imgsz=args.imgsz,
                    device=args.device)
    if not gate.available:
        raise SystemExit(f"YOLO model not loaded: {gate._load_error}\n"
                         f"(install ultralytics and/or point --weights at a trained .pt)")

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise SystemExit(f"cannot open camera {args.cam} (try --cam 1)")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    aw, ah = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"camera {args.cam} open at {aw}x{ah}. Hold a LABELLED package in front of it. "
          f"running {args.secs:.0f}s (Ctrl-C to stop)… weights={args.weights} "
          f"conf>={args.conf} iou={args.iou} imgsz={args.imgsz}")
    t0 = time.time()
    last_status = 0.0
    frames = hits = 0
    saved_hit = False
    last_no = None

    try:
        while time.time() - t0 < args.secs:
            ok, frame = cap.read()                       # BGR from OpenCV
            if not ok:
                time.sleep(0.05)
                continue
            frames += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # YoloGate's contract is RGB in
            dets = gate.analyze(rgb)                       # [{x,y,w,h,conf,text,reason}, ...]
            if dets:
                hits += 1
                fw = frame.shape[1]
                best = max(dets, key=lambda d: d["conf"])
                print(f"[{time.time()-t0:5.1f}s] {best['text']}  {best['w']}x{best['h']}px "
                      f"({100*best['w']/fw:.0f}% of frame width)  [{len(dets)} box(es)]")
                if args.show or not saved_hit:
                    for d in dets:
                        cv2.rectangle(frame, (d["x"], d["y"]), (d["x"] + d["w"], d["y"] + d["h"]),
                                      (0, 200, 0), 2)
                        cv2.putText(frame, d["text"], (d["x"], max(14, d["y"] - 5)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1, cv2.LINE_AA)
                if not saved_hit:
                    cv2.imwrite("label_hit.png", frame)
                    saved_hit = True
                    print("           saved frame -> ./label_hit.png")
            else:
                last_no = frame

            if args.show:
                cv2.imshow("yolo test (Esc to quit)", frame)
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
        print("=> NO detection. If a real label was in view, lower --conf, hold the target larger/"
              "steadier, or retrain — a plain bag SHOULD read 0, that's correct.")
    else:
        print("=> Detection WORKS. Note the '% of frame width' — that's how big the target must appear "
              "on the head camera. If wrinkles/tape/glare also trip it (false positives), RAISE --conf.")


if __name__ == "__main__":
    main()
