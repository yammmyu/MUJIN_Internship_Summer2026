#!/usr/bin/env python3
"""Offline tests for the data-driven, partial retreat (real_world/retreat.py).

The missed-grasp recovery no longer always homes to the fixed start pose: it retreats to the
NEAREST pre-grasp approach waypoint that is NOT AHEAD of the arm's current approach phase, so it
backs off only as far as needed. These tests pin that selection logic and the retreat wiring with
plain fakes (no robot / sim / torch).

  W1  selection at a waypoint   -> an arm sitting on waypoint k retreats to waypoint k (no jump).
  W2  never advances (not-ahead) -> an arm between k and k+1 never selects k+1 (that would advance
                                    toward the grasp); it picks the nearest waypoint at/behind it.
  W3  endpoints                 -> beyond the last waypoint -> last (minimal retreat); before the
                                    first -> first (full retreat).
  W4  retreat_to_nearest wiring -> reads the live pose, clears the queue, opens the right gripper,
                                    and commands move_to_joints with the selected waypoint.
  W5  shipped config            -> config/retreat_waypoints.json loads as an (n>=2, 14) array whose
                                    row 0 (average start) is what the auto run homes to.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from real_world.retreat import nearest_waypoint_not_ahead, retreat_to_nearest


def _ordered_waypoints():
    """A synthetic ordered approach: 5 monotone 14-vecs (deterministic, config-independent)."""
    base = np.linspace(0.0, 1.0, 5)[:, None] * np.ones((5, 14))   # row k = k/4 on every joint
    return base


def test_selection_at_waypoint():
    W = _ordered_waypoints()
    for k in range(len(W)):
        assert nearest_waypoint_not_ahead(W[k], W) == k, f"W1: at WP{k} did not select itself"


def test_never_advances():
    W = _ordered_waypoints()
    # A pose between WP2 and WP3, biased toward WP3, must still pick WP2 (never the ahead WP3).
    for a in (0.4, 0.6, 0.9):
        q = W[2] * (1 - a) + W[3] * a
        idx = nearest_waypoint_not_ahead(q, W)
        assert idx <= 2, f"W2: between WP2/WP3 (bias {a}) advanced to WP{idx+1}"
        assert idx == 2, f"W2: expected the nearest-behind WP3(idx2), got WP{idx+1}"


def test_endpoints():
    W = _ordered_waypoints()
    beyond = W[-1] + (W[-1] - W[-2]) * 3.0          # past the grasp end of the approach
    assert nearest_waypoint_not_ahead(beyond, W) == len(W) - 1, "W3: beyond-last should be the last"
    before = W[0] - (W[1] - W[0]) * 2.0             # behind the start
    assert nearest_waypoint_not_ahead(before, W) == 0, "W3: before-start should be the first"
    assert nearest_waypoint_not_ahead(W[0], [W[0]]) == 0, "W3: single waypoint -> index 0"


class _FakeLock:
    def __enter__(self): return self
    def __exit__(self, *a): return False


class _FakePipeline:
    def reset_merge(self): self.reset = True


class _FakeEnv:
    """Minimal env exposing exactly what retreat_to_home / retreat_to_nearest touch."""
    def __init__(self, live_q14):
        self._live = np.asarray(live_q14, float)
        self._lock = _FakeLock()
        self._robot_q = [1, 2, 3]
        self._staged_release = [1]
        self._queued_through = 7
        self.pipeline = _FakePipeline()
        self.grip_reset = False
        self.gr_cmd = None
        self.moved_to = None

    def _read_arm14(self): return self._live.copy()
    def reset_grip_latch(self): self.grip_reset = True
    def command_gripper(self, gr=None): self.gr_cmd = gr
    def move_to_joints(self, q14, joint_step=None): self.moved_to = np.asarray(q14, float)


def test_retreat_to_nearest_wiring():
    W = _ordered_waypoints()
    # Arm sits at WP3 -> should retreat to WP3, having cleared the queue and opened the right grip.
    env = _FakeEnv(W[3])
    ok = retreat_to_nearest(env, W, open_grip=0.0)
    assert ok is True, "W4: retreat should report the move was issued"
    assert env.moved_to is not None and np.allclose(env.moved_to, W[3]), "W4: wrong retreat target"
    assert env._robot_q == [] and env._staged_release == [], "W4: queue/staged not cleared"
    assert env.grip_reset and env.gr_cmd == 0.0, "W4: right gripper not opened / latch not reset"


def test_shipped_config():
    from real_world.postprocess import load_retreat_waypoints
    W = load_retreat_waypoints()
    assert W is not None, "W5: config/retreat_waypoints.json missing"
    assert W.ndim == 2 and W.shape[1] == 14 and len(W) >= 2, f"W5: bad shape {W.shape}"
    # An arm parked at the shipped start (row 0) retreats to row 0 (the auto-run home).
    assert nearest_waypoint_not_ahead(W[0], W) == 0, "W5: start pose should select waypoint 1"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"[retreat] {name} OK")
    print("[retreat] ALL PASS")
