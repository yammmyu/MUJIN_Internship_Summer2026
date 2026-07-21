#!/usr/bin/env python3
"""Standalone webcam ArUco-marker tester (also reads 1-D/QR barcodes if zxing-cpp is present).

Checks whether the no-flip detection can even SEE a marker from a LIVE camera on this machine,
independent of the robot head camera. Mirrors real_world/no_flip_place.ArucoGate (same cv2.aruco
dictionary + detectMarkers call), so what works here works on the robot.

First print/display a marker:
    python scripts/barcode_webcam_test.py --make-aruco        # saves ./aruco_marker.png (id 7)
    python scripts/barcode_webcam_test.py --make-aruco --id 3

Then hold it up to the webcam:
    python scripts/barcode_webcam_test.py                     # camera 0, 30 s
    python scripts/barcode_webcam_test.py --cam 1 --secs 60
    python scripts/barcode_webcam_test.py --show              # live preview (needs a display)

On each detection it prints the marker id and how big it was in the frame (px + % of width — the
number that tells you how close/large it must be). First hit is saved to ./marker_hit.png; if nothing
is ever detected it saves ./marker_nodetect.png so you can eyeball what the camera actually saw.
"""
import argparse
import time

import cv2
import numpy as np

ARUCO_DICT = "DICT_4X4_50"

try:
    import zxingcpp                       # optional: also report 1-D/QR barcodes for comparison
except ImportError:
    zxingcpp = None


def _aruco_detector(dict_name=ARUCO_DICT):
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(getattr(aruco, dict_name))
    return aruco.ArucoDetector(dictionary, aruco.DetectorParameters())


def _marker_width_px(corner):
    """Marker side length in px (top-left -> top-right of its 4x2 corner array)."""
    tl, tr = corner[0][0], corner[0][1]
    return float(np.hypot(tr[0] - tl[0], tr[1] - tl[1]))


def make_marker(marker_id, dict_name=ARUCO_DICT, px=600, path="aruco_marker.png"):
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(getattr(aruco, dict_name))
    img = aruco.generateImageMarker(dictionary, marker_id, px)
    border = px // 5                                  # white quiet zone (required for detection)
    canvas = np.full((px + 2 * border, px + 2 * border), 255, np.uint8)
    canvas[border:border + px, border:border + px] = img
    cv2.imwrite(path, canvas)
    print(f"saved {dict_name} marker id={marker_id} -> ./{path} ({canvas.shape[1]}x{canvas.shape[0]}px). "
          f"Print or display it, then run this script without --make-aruco.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0, help="camera index (default 0)")
    ap.add_argument("--secs", type=float, default=30.0, help="run duration in seconds")
    ap.add_argument("--width", type=int, default=1280, help="requested capture width (default 1280)")
    ap.add_argument("--height", type=int, default=720, help="requested capture height (default 720)")
    ap.add_argument("--show", action="store_true", help="open a live preview window (needs a display)")
    ap.add_argument("--make-aruco", action="store_true", help="generate a marker PNG and exit")
    ap.add_argument("--id", type=int, default=7, help="marker id for --make-aruco (default 7)")
    args = ap.parse_args()

    if args.make_aruco:
        make_marker(args.id)
        return

    det = _aruco_detector()
    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise SystemExit(f"cannot open camera {args.cam} (try --cam 1)")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)       # higher res = detect a smaller/farther marker
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"camera {args.cam} open at {aw}x{ah} ({ARUCO_DICT}). Hold an ArUco marker in front of it. "
          f"running {args.secs:.0f}s (Ctrl-C to stop)…" + ("" if zxingcpp else "  [zxing not installed]"))
    t0 = time.time()
    last_status = 0.0
    frames = hits = 0
    saved_hit = False
    last_no = None
    seen_ids = set()

    try:
        while time.time() - t0 < args.secs:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            frames += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _rej = det.detectMarkers(gray)
            found = [] if ids is None else list(ids.flatten())
            if found:
                hits += 1
                fw = frame.shape[1]
                for corner, mid in zip(corners, found):
                    mid = int(mid)
                    if mid not in seen_ids:
                        seen_ids.add(mid)
                        wpx = _marker_width_px(corner)
                        print(f"[{time.time()-t0:5.1f}s] MARKER id={mid}  "
                              f"size={wpx:.0f}px ({100*wpx/fw:.0f}% of frame width)")
                if not saved_hit:
                    cv2.imwrite("marker_hit.png", frame)
                    saved_hit = True
                    print("           saved frame -> ./marker_hit.png")
            else:
                last_no = frame
                if zxingcpp is not None:               # a marker isn't visible; note any barcode instead
                    codes = [r.text for r in zxingcpp.read_barcodes(gray) if r.text]
                    if codes:
                        print(f"[{time.time()-t0:5.1f}s] (no marker, but a BARCODE decoded: {codes})")

            if args.show:
                cv2.imshow("aruco test (Esc to quit)", frame)
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
        cv2.imwrite("marker_nodetect.png", last_no)
        print("saved a no-detection sample frame -> ./marker_nodetect.png (inspect it)")

    print(f"\ndone. frames={frames}, detections={hits}, unique ids={sorted(seen_ids)}")
    if hits == 0:
        print("=> NO marker detected. Generate one with --make-aruco, display/print it, ensure it has a "
              "white border (quiet zone), and try again. If it still fails up close, the camera frame is "
              "too blurry/low-contrast.")
    else:
        print("=> Detection WORKS here. Note the smallest '% of frame width' above — that is how big the "
              "marker must appear in the robot head view. ArUco should read at a few % of the frame.")


if __name__ == "__main__":
    main()
