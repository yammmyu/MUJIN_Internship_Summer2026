"""Isolate WHY RobotDds joint_states are frozen at rest.

Run each mode in a FRESH process (rclpy/Slam side effects persist within a
process, so don't combine them). While it polls, SLOWLY HAND-MOVE the left arm
so we can see whether the reported joints actually change.

    python -m scripts.diag_joint_states bare    # RobotDds() only  (matches exampleSDK.py — should be LIVE)
    python -m scripts.diag_joint_states slam     # + Slam()
    python -m scripts.diag_joint_states rclpy     # + rclpy.init() AFTER Slam (the robot_control_gui.py order)

The first mode whose joints stay CONSTANT while you move the arm is the culprit
step. Compare `bare` (expected live) against the others.
"""
import sys
import time

import numpy as np

from a2d_sdk.robot import RobotDds as Robot


def _poll(robot, label, seconds=8.0, hz=2.0):
    print(f"\n=== [{label}] polling arm_joint_states for {seconds:.0f}s — "
          f"MOVE THE LEFT ARM NOW ===")
    prev = None
    moved = False
    n = int(seconds * hz)
    for _ in range(n):
        try:
            vals, ts = robot.arm_joint_states()
        except Exception as e:
            print(f"  read error: {e}")
            time.sleep(1.0 / hz)
            continue
        arr = np.asarray(vals, dtype=float)
        left7 = np.round(arr[:7], 4) if arr.size >= 7 else arr
        delta = None if prev is None else float(np.max(np.abs(arr - prev)))
        if delta is not None and delta > 1e-4:
            moved = True
        print(f"  ts={ts}  left7={left7}  max|Δ|={'-' if delta is None else f'{delta:.5f}'}")
        prev = arr
        time.sleep(1.0 / hz)
    verdict = "LIVE ✅ (joints changed)" if moved else "FROZEN ❌ (no change while moving)"
    print(f"=== [{label}] verdict: {verdict} ===")
    return moved


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "bare"

    if mode == "rclpy_first":
        import rclpy
        rclpy.init(args=None)
        from a2d_sdk.robot import Slam
        slam = Slam()                      # noqa: F841  (side effect only)
        print("[rclpy_first] rclpy.init() THEN Slam()")
        robot = Robot()
        time.sleep(2.0)
        _poll(robot, mode)
        return

    robot = Robot()
    time.sleep(1.0)

    if mode == "bare":
        pass

    elif mode == "slam":
        from a2d_sdk.robot import Slam
        slam = Slam()                      # noqa: F841
        print("[slam] Slam() created")
        time.sleep(2.0)

    elif mode == "rclpy":
        from a2d_sdk.robot import Slam
        slam = Slam()                      # noqa: F841 — Slam BEFORE rclpy.init (GUI order)
        print("[rclpy] Slam() then rclpy.init()")
        import rclpy
        rclpy.init(args=None)
        time.sleep(2.0)

    elif mode == "full":
        import threading
        import rclpy
        from a2d_sdk.robot import Slam
        from control_wheel_example import WheelController
        slam = Slam()                      # noqa: F841
        rclpy.init(args=None)
        wc = WheelController()
        threading.Thread(target=rclpy.spin, args=(wc,), daemon=True).start()
        print("[full] Slam + rclpy.init + WheelController + spin (full GUI setup)")
        time.sleep(2.0)

    else:
        print(f"unknown mode: {mode!r}")
        sys.exit(2)

    _poll(robot, mode)


if __name__ == "__main__":
    main()
