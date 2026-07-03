"""Sim-only action-preview loop for HumanoidEnv.

Extracted from HumanoidEnv. A SimPreview drains a submitted policy action chunk and drives the
PyBullet preview: decode the LEFT arm, workspace-guard (H4) + smooth (position EMA + orientation
SLERP, H3), IK-solve (warm-started from the last solution), and command the sim. It NEVER touches
the real robot — that is the release loop. It owns its own action queue, smoothing state, and the
LEFT IK warm-start seed (``last_q`` / ``set_seed``), all behind its own lock.

The env owns the thread + pacing (its exec loop calls ``tick(sim)`` once per dt) and passes the
current sim in, since the GUI attaches/detaches the sim dynamically.
"""

import copy
import threading

import numpy as np

from real_world.ik import IKQuery, decode_action_row, smooth_quat_step
from real_world.postprocess import pos_in_workspace, QUAT_ALPHA


class SimPreview:
    """Drains an action chunk into the sim preview (sim-only; never commands the robot)."""

    def __init__(self, solver, seed_q=None, pos_alpha=0.5):
        self.solver = solver                       # LEFT-arm IK solver (shared with the env)
        self.pos_alpha = pos_alpha                 # position low-pass EMA (1.0 = no smoothing)
        self._lock = threading.Lock()
        self._action_queue = []                    # most recent submitted chunk, FIFO drained
        self._smoothed_pos = None                  # running position EMA state
        self._quat_prev = None                     # previous target quat for SLERP smoothing (H3)
        # LEFT IK warm-start seed (7,): the recording's first arm joints when given, else the
        # limit-clipped zero config. Re-seeded to the live left arm on preview launch / resync.
        self._last_q = (np.asarray(seed_q, dtype=np.float64).copy()
                        if seed_q is not None else self.solver.m.clip(np.zeros(7)))

    @property
    def last_q(self):
        """The current LEFT IK warm-start seed (7,)."""
        return self._last_q

    def set_seed(self, q7):
        """Set the IK warm-start seed (e.g. to the robot's current left-arm joints)."""
        self._last_q = np.asarray(q7, dtype=np.float64).copy()

    def submit(self, action_chunk):
        """Hand a chunk to the preview queue (auto-run / replay). Sim only — this path can NEVER
        reach the robot. The hardware path is the env's validate_and_stage() + release_to_robot()."""
        with self._lock:
            self._action_queue = copy.deepcopy(list(action_chunk)) if action_chunk else []

    def queue_empty(self):
        """True when no actions are pending (used by runners to detect drain)."""
        with self._lock:
            return not self._action_queue

    def _solve_ik(self, pos, quat):
        """Target EE pose -> 7 left-arm joint angles via our URDF IK (warm-started from the last
        solution). Returns None when the target is unreachable, leaving the seed intact so the next
        target plans from the last good config."""
        q7 = self.solver.solve(IKQuery(
            target_pos=np.asarray(pos, dtype=np.float64),
            target_quat=np.asarray(quat, dtype=np.float64),
            current_joints=self._last_q))
        if not self.solver.last_reachable:
            print(f"[SimPreview] IK unreachable (pos err "
                  f"{self.solver.last_pos_err * 1000:.1f} mm); skipping action")
            return None
        self._last_q = q7
        return q7

    def tick(self, sim):
        """One preview iteration (called each dt by the env's exec thread): pop one queued action,
        decode the LEFT arm, workspace-guard + smooth, IK-solve, and command the sim. No-op when the
        queue is empty. Applies the same workspace + smoothing guards as validation so the preview
        matches what would be validated."""
        with self._lock:
            action = self._action_queue.pop(0) if self._action_queue else None
        if action is None:
            return

        pos, quat, grip = decode_action_row(action)
        pos = np.asarray(pos, dtype=np.float64)
        if not pos_in_workspace(pos):                               # H4
            print("[SimPreview] target outside workspace; skipping")
            return

        # Smooth the target: position EMA, orientation SLERP toward the new quat (H3).
        if self._smoothed_pos is None:
            self._smoothed_pos = pos
        else:
            self._smoothed_pos = self.pos_alpha * pos + (1.0 - self.pos_alpha) * self._smoothed_pos
        self._quat_prev = smooth_quat_step(self._quat_prev, quat, QUAT_ALPHA)

        q7 = self._solve_ik(self._smoothed_pos, self._quat_prev)
        if q7 is None:                     # unreachable target -> skip (seed not advanced)
            return
        if sim is not None:
            sim.command(q7, grip)
