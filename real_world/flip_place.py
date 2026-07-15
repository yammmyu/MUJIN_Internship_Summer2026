"""Scripted release macro that plugs in after the policy's grab+lift+flip.

The dual-arm policy (trained on MDM_data_collection/recordings 001-100) does grab -> lift ->
flip the grasped object ~180 deg, but has no reliable "move out and release" of its own. This
module watches for the flip completing and then, INLINE in the auto loop, takes over:

    1. stop predicting + clear ALL queues (so nothing stale runs)
    2. move the RIGHT arm from its LIVE flip-complete pose to a FIXED release pose, following the
       shape recorded in recording107 (real_world/assets/flip_release_path.npy), OPEN the gripper
       there (release), then reverse the exact same path back to the live start
    3. clear everything again and hand back -> auto inference resumes from the live pose

Design decisions (see the plan discussion):
  * JOINT space, not EE: the flip is a ~180 deg wrist swing; a fixed END joint config guarantees the
    same release EE point via FK with NO IK (no redundancy branch-flip / wrong-direction wrist).
  * The start ADAPTS to wherever the flip ended up (varies), the END is FIXED: a decaying-offset warp
    re-anchors recording107's shape so out[0]=live pose and out[-1]=the recorded release config.
  * Slow + smooth: streamed at vel_frac of max joint velocity, subdivided, from the live pose (zero
    seam); recording lightly pre-smoothed offline.
  * No snap on return: the robot queue + staging + merge buffer + grip latch are ALL cleared before
    and after, and the arm ends at the live start, so the first resumed inference is fresh.

Integration (one call-in in InferenceController._run_auto_inference, top of loop):
    fp = getattr(self, "flip_place", None)
    if fp is not None and fp.maybe_trigger(env):
        continue                      # macro ran this cycle; skip predicting
"""

import logging
import os
import time

import numpy as np

log = logging.getLogger(__name__)

DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "assets", "flip_release_path.npy")

RJ7_INDEX = 13          # right wrist-roll joint in the 14-vec arm_joints [left7, right7]
R_GRIP_IDX = 1          # right channel in a [gl, gr] gripper command


class FlipPlaceMacro:
    def __init__(self, path=DEFAULT_PATH, *,
                 rj7_change=2.5,       # fire when right wrist-roll has ROTATED this far since the grab
                                       # (|Δrj7|); the flip swings ~3.0-3.5 rad, so ~2.5 is near-complete
                 settle_s=0.3,         # |Δrj7| must stay >= threshold this long (never fire mid-flip)
                 vel_frac=0.5,         # streaming speed = fraction of MAX joint velocity
                 release_settle_s=0.4, # pause after opening the gripper before withdrawing
                 open_grip=0.0):       # 0 = open (release)
        self.path = np.load(path).astype(np.float64)      # (M, 14) absolute-joint waypoints
        if self.path.ndim != 2 or self.path.shape[1] != 14:
            raise ValueError(f"flip_release_path must be (M,14); got {self.path.shape}")
        self.rj7_change = float(rj7_change)
        self.settle_s = float(settle_s)
        self.vel_frac = float(vel_frac)
        self.release_settle_s = float(release_settle_s)
        self.open_grip = float(open_grip)
        self.enabled = True           # live on/off: False -> maybe_trigger is a no-op (macro never runs)
        self._fired = False           # already ran for the current closed episode?
        self._above_since = None      # monotonic time |Δrj7| first went >= threshold (settle timer)
        self._prev_closed = False     # right gripper closed last check (open->close edge = the grab)
        self._rj7_at_grab = None      # wrist-roll captured at the grab; the flip is measured FROM here
        log.info("FlipPlaceMacro ready: %d waypoints, rj7_change>=%.2f settle=%.1fs vel=%.0f%% max",
                 len(self.path), self.rj7_change, self.settle_s, self.vel_frac * 100)

    def maybe_trigger(self, env) -> bool:
        """Loop hook: True => the macro ran this cycle (caller should skip predicting). Fires ONCE per
        closed grasp, when the right gripper is commanded closed AND the wrist-roll has ROTATED by
        rj7_change (|Δrj7|) from its value AT THE GRAB, held for settle_s. Measuring the CHANGE (not an
        absolute angle) makes the trigger independent of the grasp orientation. Resets on release.
        No-op (returns False) while disabled, so the operator can turn the post-flip macro off live."""
        if not self.enabled:
            return False
        grip = getattr(env, "_last_grip_cmd", None)
        grip_closed = grip is not None and grip[R_GRIP_IDX] >= 1
        if not grip_closed:                          # released -> disarm; ready for the next grasp
            self._prev_closed = False
            self._fired = False
            self._above_since = None
            self._rj7_at_grab = None
            return False
        arm14 = env._read_arm14()
        if arm14 is None:
            return False
        rj7 = float(arm14[RJ7_INDEX])
        if not self._prev_closed:                    # open->close edge = the grab: capture the baseline
            self._prev_closed = True
            self._rj7_at_grab = rj7
        if self._fired or self._rj7_at_grab is None:
            return False
        change = abs(rj7 - self._rj7_at_grab)        # how far the wrist has rotated since the grab
        if change < self.rj7_change:
            self._above_since = None                 # fell back -> restart the settle timer
            return False
        now = time.monotonic()
        if self._above_since is None:
            self._above_since = now
            return False
        if now - self._above_since < self.settle_s:
            return False
        self._fired = True
        log.warning("[flip-place] flip complete (|Δrj7|=%.2f >= %.2f since grab) -> stop auto, run macro",
                    change, self.rj7_change)
        self._run(env, np.asarray(arm14, dtype=np.float64))
        return True

    # -- internals --------------------------------------------------------------
    def _clear(self, env):
        """Drop EVERY source of stale motion so nothing snaps: robot queue, manual staging, streaming
        cursor, and the temporal-ensemble merge buffer."""
        with env._lock:
            env._robot_q.clear()
            env._staged_release.clear()
            env._queued_through = -1
        if getattr(env, "pipeline", None) is not None:
            env.pipeline.reset_merge()

    def _run(self, env, q_now):
        self._clear(env)                                  # 1. stop/clear before moving
        M = len(self.path)
        # 2. warp: out[i] = rec[i] + (q_now - rec[0]) * (1 - i/(M-1)) -> out[0]=q_now, out[-1]=release
        decay = 1.0 - np.arange(M) / (M - 1)
        fwd = self.path + np.outer(decay, q_now - self.path[0])
        fwd[:, :7] = q_now[:7]                             # HOLD the left arm at its current pose
        # forward: move out to the fixed release pose, gripper untouched (stays closed on the object)
        env.play_joint_path(fwd, vel_frac=self.vel_frac, grip=None)
        # release at the fixed point, let it settle, then withdraw along the same path
        if hasattr(env, "command_gripper"):
            env.command_gripper(gr=self.open_grip)
        time.sleep(self.release_settle_s)
        env.play_joint_path(fwd[::-1], vel_frac=self.vel_frac, grip=None)   # 3. come back to q_now
        self._clear(env)                                  # 4. clear again -> auto resumes fresh
        if hasattr(env, "reset_grip_latch"):
            env.reset_grip_latch()
        log.info("[flip-place] macro complete -> auto resumes from the live pose")

    def reset(self):
        """Disarm (e.g. on auto stop / E-stop) so a restart doesn't immediately re-fire."""
        self._fired = False
        self._above_since = None
        self._prev_closed = False
        self._rj7_at_grab = None
