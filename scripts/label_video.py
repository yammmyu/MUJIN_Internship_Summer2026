#!/usr/bin/env python3
"""Annotate a video file with the no-flip YOLO detector's boxes (real_world/assets/head_yolo.pt).

Offline counterpart to scripts/label_webcam_test.py: instead of a live camera it runs the SAME trained
detector over every frame of an input video and writes a new video with the detections drawn on it
(class name + confidence). Use it to review recorded head-camera footage and to pick the confidence
threshold the gate should run at before touching the robot.

Usage:
    python scripts/label_video.py in.mp4                        # -> in_labeled.mp4, defaults from YoloGate
    python scripts/label_video.py in.mp4 -o out.mp4 --conf 0.35 # tune the threshold
    python scripts/label_video.py in.mp4 --classes barcode      # only draw the barcode class
    python scripts/label_video.py in.mp4 --device cuda:0        # GPU inference
    python scripts/label_video.py in.mp4 --show                 # also preview while writing (q quits)

The defaults (conf/iou/imgsz) mirror YoloGate's so the boxes you see here are the ones the gate would
fire on. At the end it prints how many frames contained a detection and a per-class box count — the
"would the gate have fired?" summary for that clip. Requires `pip install ultralytics`.
"""
import argparse
import os
import sys

import cv2

# Same weights the no-flip gate loads by default (real_world/assets/head_yolo.pt).
DEFAULT_WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "..", "real_world", "assets", "right_yolo.pt")

# Per-class box colours (BGR), cycled by class id so every class stays visually distinct.
COLORS = [(0, 200, 255), (0, 220, 60), (255, 120, 0), (200, 0, 255), (0, 0, 255), (255, 255, 0)]


def draw_box(frame, xyxy, label, color):
    """Draw one detection: rectangle + a filled caption bar that flips inside the box near the top."""
    x1, y1, x2, y2 = (int(round(v)) for v in xyxy)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = y1 - th - base - 3
    if top < 0:                                   # box hugs the top edge -> caption goes inside it
        top = y1 + 2
    cv2.rectangle(frame, (x1, top), (x1 + tw + 4, top + th + base + 2), color, -1)
    cv2.putText(frame, label, (x1 + 2, top + th + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
                cv2.LINE_AA)


def resolve_classes(names, wanted):
    """Map the --classes filter (ids or case-insensitive names) to a list of class ids, or None."""
    if not wanted:
        return None
    by_name = {str(n).lower(): i for i, n in names.items()}
    ids = []
    for w in wanted:
        if w.isdigit() and int(w) in names:
            ids.append(int(w))
        elif w.lower() in by_name:
            ids.append(by_name[w.lower()])
        else:
            sys.exit(f"unknown class {w!r}; model classes are {names}")
    return ids


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("video", help="input video file")
    ap.add_argument("-o", "--out", default=None, help="output path (default: <input>_labeled.mp4)")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--conf", type=float, default=0.5, help="min box confidence (YoloGate default)")
    ap.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    ap.add_argument("--imgsz", type=int, default=640, help="inference size (longest side)")
    ap.add_argument("--classes", nargs="*", default=None, help="restrict to these class ids/names")
    ap.add_argument("--device", default=None, help="e.g. cpu, cuda:0 (default: ultralytics auto)")
    ap.add_argument("--show", action="store_true", help="preview while writing (press q to stop)")
    args = ap.parse_args()

    if not os.path.exists(args.video):
        sys.exit(f"input video not found: {args.video}")
    if not os.path.exists(args.weights):
        sys.exit(f"weights not found: {args.weights}")

    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("ultralytics not installed (`pip install ultralytics`)")

    model = YOLO(args.weights)
    if args.device:
        model.to(args.device)
    names = model.names
    class_ids = resolve_classes(names, args.classes)
    print(f"model {args.weights} classes={names}"
          + (f" -> filtering to {class_ids}" if class_ids is not None else ""))

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"could not open video: {args.video}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    out_path = args.out or os.path.splitext(args.video)[0] + "_labeled.mp4"
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        cap.release()
        sys.exit(f"could not open output video for writing: {out_path}")
    print(f"input {w}x{h} @ {fps:.2f} fps, {total or '?'} frames -> {out_path}")

    n_frames = 0            # frames read
    n_hit_frames = 0        # frames with >= 1 kept detection ("the gate would have fired")
    per_class = {}          # class name -> total boxes drawn
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            n_frames += 1
            res = model.predict(frame, conf=args.conf, iou=args.iou, imgsz=args.imgsz,
                                classes=class_ids, verbose=False)[0]
            boxes = res.boxes
            n_boxes = 0 if boxes is None else len(boxes)
            for i in range(n_boxes):
                cls_id = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())
                name = str(names.get(cls_id, cls_id))
                draw_box(frame, boxes.xyxy[i].tolist(), f"{name} {conf:.2f}",
                         COLORS[cls_id % len(COLORS)])
                per_class[name] = per_class.get(name, 0) + 1
            if n_boxes:
                n_hit_frames += 1
            # Burnt-in HUD so the output video is self-describing when reviewed later.
            cv2.putText(frame, f"frame {n_frames}  det {n_boxes}  conf>={args.conf:.2f}", (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            writer.write(frame)

            if args.show:
                cv2.imshow("label_video", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("stopped early (q)")
                    break
            if total and n_frames % 50 == 0:
                print(f"  {n_frames}/{total} frames ({100.0 * n_frames / total:.0f}%)", flush=True)
    except KeyboardInterrupt:
        print("interrupted — flushing what was written so far")
    finally:
        cap.release()
        writer.release()
        if args.show:
            cv2.destroyAllWindows()

    pct = 100.0 * n_hit_frames / n_frames if n_frames else 0.0
    print(f"\nwrote {out_path}")
    print(f"frames: {n_frames}   with detections: {n_hit_frames} ({pct:.1f}%)")
    print("boxes per class: " + (", ".join(f"{k}={v}" for k, v in sorted(per_class.items()))
                                 or "none — try a lower --conf"))


if __name__ == "__main__":
    main()
