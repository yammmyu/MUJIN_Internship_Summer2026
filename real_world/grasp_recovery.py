"""Standalone CCDP-style grasp-failure recovery for the dual-arm policy.

Self-contained: loads the wrist-camera grasp detector (data/grasp_detector/detector.pt,
trained in the diffusion_policy repo) and drives failure recovery, with NO GUI coupling
and only three small call-ins from InferenceController. Needs only torch + torchvision +
the .pt checkpoint on the client machine (detection runs LOCALLY, not on the policy server).

WHAT IT DOES
------------
Watches the RIGHT gripper command in each policy chunk. On an open->close transition it
arms a grasp attempt and, after `settle_sec` (default 5 s — deliberately late so the check
never perturbs the grasp itself), runs the detector on the right wrist frame
(obs['handr_imgs'][-1]). If the gripper closed on nothing it enters RECOVERY:

    1. clear the robot queue (drop the policy's "I grasped, now lift" rows)
    2. STREAM a scripted retreat -- an interpolated right-EE lift to (current + offset) with
       the gripper opened, left arm held -- fed to env.append_actions ~2 rows/cycle from the
       auto loop, exactly like the policy is streamed (append_actions only commits
       APPEND_AHEAD_ROWS ahead of the master clock, so a one-shot append would NOT run it).
    3. hand control back -> the stochastic policy re-plans a fresh approach on its own.

No failure demos, no high-level planner: each grasp attempt is a sub-problem and the
retreat simply clears the failed grasp so the stochastic policy can try again.

The env solves the retreat: we only emit EE-space target rows; env.append_actions runs
Pinocchio IK + sim validation + the velocity-matched seam ramp per row.

INTEGRATION (see the three call-ins in InferenceController):
    __init__: self.recovery = GraspRecoveryMonitor("data/grasp_detector/detector.pt",
                                                    open_grip=..., closed_grip_min=...)
    _run_auto_inference (top of loop):
        if rec and rec.is_retreating:
            rec.pump(env); time.sleep(0.02); continue     # stream retreat, skip the server
    _run_inference (after obs, before post_predict):
        if rec and rec.maybe_start(env, obs): return True  # miss -> entered recovery
    _run_inference (after `action`):
        if rec: rec.note_action(action)                    # track the grip command

(ros2) mujin@PF3784S4:~/workspaces/humanoid$ python robot_control_gui.py 
Init glog with processor name:python3.10, pid:550046
/home/mujin/miniconda3/envs/ros2/lib/python3.10/site-packages/google/protobuf/__init__.py:37: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
  __import__('pkg_resources').declare_namespace(__name__)
pybullet build time: Jan 29 2025 23:16:28
[startup] running safety pre-flight (tests/test_safety_invariants.py)…
b3Warning[examples/Importers/ImportURDFDemo/BulletUrdfImporter.cpp,126]:
No inertial data for link, using mass=1, localinertiadiagonal = 1,1,1, identity local inertial frameb3Warning[examples/Importers/ImportURDFDemo/BulletUrdfImporter.cpp,126]:
link-armb3Warning[examples/Importers/ImportURDFDemo/BulletUrdfImporter.cpp,126]:
No inertial data for link, using mass=1, localinertiadiagonal = 1,1,1, identity local inertial frameb3Warning[examples/Importers/ImportURDFDemo/BulletUrdfImporter.cpp,126]:
gripper_centerb3Warning[examples/Importers/ImportURDFDemo/BulletUrdfImporter.cpp,126]:
No inertial data for link, using mass=1, localinertiadiagonal = 1,1,1, identity local inertial frameb3Warning[examples/Importers/ImportURDFDemo/BulletUrdfImporter.cpp,126]:
right_gripper_center[HumanoidEnv]: started (collect=off, exec=on, real=on).
[HumanoidEnv] release refused: nothing sim-validated (run 执行 first).
[HumanoidEnv] sim-validated 1381 points (id 1); 1381 substep(s) staged for release.
[HumanoidEnv] released 1381 validated pts (+89 ramp-in) to robot.
[HumanoidEnv] release refused: nothing sim-validated (run 执行 first).
[HumanoidEnv] sim-validated 1381 points (id 2); 2762 substep(s) staged for release.
[HumanoidEnv] released 1381 validated pts (+91 ramp-in) to robot.
[HumanoidEnv] dispatch-ramp: |Δq|=0.632 rad exceeds cap 0.033 -> streaming 19 bounded substeps
[HumanoidEnv] E-STOP: latched; dropped 4229 pending/staged cmds; holding pose.
[HumanoidEnv] release refused: E-stop latched (press 复位 to reset).
[HumanoidEnv] E-stop reset; release re-enabled (run 执行 then 释放).
[safety] ALL INVARIANTS PASS (C1 C2 C3 C4 C5 H1)
[startup] safety pre-flight passed.

SLAM 模块初始化成功（已解冻关节状态）
[INFO] [1783483424.279491321] [wheel_controller_example]: Wheel Controller Example node started. Publishing a target pose for the robot base.
Exception in thread Thread-3 (_run):
Traceback (most recent call last):
  File "/home/mujin/miniconda3/envs/ros2/lib/python3.10/threading.py", line 1016, in _bootstrap_inner
    self.run()
  File "/home/mujin/miniconda3/envs/ros2/lib/python3.10/threading.py", line 953, in run
    self._target(*self._args, **self._kwargs)
  File "/home/mujin/workspaces/humanoid/examples/control_wheel_example.py", line 33, in _run
    self.slam = Slam()
  File "/home/mujin/miniconda3/envs/ros2/lib/python3.10/site-packages/a2d_sdk/robot.py", line 167, in __init__
    self._slam = SlamCore()
  File "/home/mujin/miniconda3/envs/ros2/lib/python3.10/site-packages/a2d_sdk/core/slam/slam_core.py", line 53, in __init__
    self.node_ = agibotdds.Node("A2DRosSlam")
  File "/home/mujin/miniconda3/envs/ros2/lib/python3.10/site-packages/cosine_bus/agibotdds_py3/agibotdds.py", line 213, in __init__
    self.node = _AGIBOTDDS.new_PyNode(name)
SystemError: <built-in function new_PyNode> returned NULL without setting an exception
Exception ignored in: <function Node.__del__ at 0x701c995e3520>
Traceback (most recent call last):
  File "/home/mujin/miniconda3/envs/ros2/lib/python3.10/site-packages/cosine_bus/agibotdds_py3/agibotdds.py", line 225, in __del__
    for publisher in self.list_publisher:
AttributeError: 'Node' object has no attribute 'list_publisher'
HTTP/JSON listening on http://0.0.0.0:9000/robot_info  (Ctrl-C to stop)
12:03:48 INFO real_world.grasp_recovery: GraspRecoveryMonitor ready: settle=5.0s closed_grip_min=60.0 roi=(330, 265, 625, 480) thr=0.50 device=cpu
12:03:48 INFO real_world.postprocess: smoothing set: radius=6 sigma=1.09 m=0.120
12:03:48 INFO real_world.postprocess: te_buffer_len set: 8
[HumanoidEnv] speed_scale=3.000 -> substeps_per_row=4, ramp_joint_step=0.0333
[tuning] restored from /home/mujin/workspaces/humanoid/tuning_config.json
12:03:49 INFO real_world.postprocess: smoothing set: radius=6 sigma=1.09 m=0.120
12:03:49 INFO real_world.postprocess: te_buffer_len set: 8
[HumanoidEnv] speed_scale=3.000 -> substeps_per_row=4, ramp_joint_step=0.0333
[HumanoidEnv]: started (collect=on, exec=on, real=on).
12:03:49 INFO real_world.inference_controller: InferenceController ready (env owned by caller).
12:03:49 INFO xr_examples.pico_vr_server.server: Downstream listening on 0.0.0.0:5555
12:03:49 INFO xr_examples.pico_vr_server.server: Upstream listening on 0.0.0.0:5556
^Z[2]   Killed                  python robot_control_gui.py
[3]   Killed                  python robot_control_gui.py

[5]+  Stopped                 python robot_control_gui.py
(ros2) mujin@PF3784S4:~/workspaces/humanoid$ python -c "import torch,torchvision; print("torch.__version__","torchvision.__version__")"
2.12.1+cpu 0.27.1+cpu
(ros2) mujin@PF3784S4:~/workspaces/humanoid$ 

"""

import logging
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18

log = logging.getLogger(__name__)

# 20-col dual_arm_ee_image action row: L[pos3,rot6d6,grip1] ++ R[pos3,rot6d6,grip1]
L_EE = slice(0, 9)      # left [pos3, rot6d6]  (held during retreat)
L_GRIP = 9
R_POS = slice(10, 13)   # right pos3
R_ROT = slice(13, 19)   # right rot6d6
R_GRIP = 19

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class _Detector:
    """Inlined GraspDetector: ResNet18 head on the fixed finger-gap ROI crop."""

    def __init__(self, ckpt_path, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        # weights_only=False: the checkpoint stores plain python values (roi/threshold/size) next to
        # the state_dict, which torch>=2.6's weights_only default rejects. It's our own trusted file.
        ck = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.roi = tuple(ck["roi"])
        self.empty_idx = ck["empty_idx"]
        self.threshold = ck["threshold"]
        m = resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, 2)
        m.load_state_dict(ck["state_dict"])
        self.model = m.to(self.device).eval()
        self.tf = transforms.Compose([
            transforms.Resize((ck["size"], ck["size"])),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    @torch.no_grad()
    def p_empty(self, frame) -> float:
        if not isinstance(frame, Image.Image):
            frame = Image.fromarray(np.asarray(frame)[..., :3].astype(np.uint8))
        x = self.tf(frame.convert("RGB").crop(self.roi)).unsqueeze(0).to(self.device)
        return torch.softmax(self.model(x), dim=1)[0, self.empty_idx].item()


class GraspRecoveryMonitor:
    def __init__(self, detector_path, *,
                 settle_sec=5.0,
                 retreat_offset=(0.0, 0.0, 0.05),   # world-frame dxyz on right EE; +Z = lift
                 retreat_rows=12,                    # interpolated rows in the scripted retreat
                 retreat_timeout_sec=10.0,           # hard cap on a single retreat
                 open_grip=0.0,                      # retreat feeds append_actions directly (bypasses
                                                     # binarize) -> downstream wants {0,1}; 0 = open
                 closed_grip_min=10.0,               # = postprocess.GRIPPER_CLOSE_THRESH: note_action
                                                     # reads the RAW server grip [0,~85] (pre-binarize)
                 device=None):
        self.det = _Detector(detector_path, device=device)
        self.settle_sec = settle_sec
        self.retreat_offset = np.asarray(retreat_offset, dtype=float)
        self.retreat_rows = int(retreat_rows)
        self.retreat_timeout_sec = retreat_timeout_sec
        self.open_grip = float(open_grip)
        self.closed_grip_min = float(closed_grip_min)

        # grasp-attempt tracking
        self._prev_closed = False
        self._close_t = None          # monotonic time of the last open->close transition
        self._checked = False         # detector already run for this closure?

        # retreat-streaming state
        self._retreating = False
        self._retreat_traj = None     # list of 20-col rows anchored at _anchor_id
        self._anchor_id = None        # master id of retreat row 0
        self._end_id = None           # master id of the last retreat row
        self._retreat_deadline = None
        self._log_ts = {}             # key -> last monotonic log time (rate-limits hot-loop logs)
        log.info("GraspRecoveryMonitor ready: settle=%.1fs closed_grip_min=%.1f roi=%s thr=%.2f "
                 "device=%s", settle_sec, self.closed_grip_min, self.det.roi, self.det.threshold,
                 self.det.device)

    def _throttled(self, key, interval=2.0):
        """True at most once per `interval` s for `key` — keeps the per-chunk diagnostics readable."""
        now = time.monotonic()
        if now - self._log_ts.get(key, 0.0) >= interval:
            self._log_ts[key] = now
            return True
        return False

    # -- hook 1: every policy chunk, to track the right-grip command ------------
    def note_action(self, action_chunk):
        a = np.asarray(action_chunk, dtype=float)
        if a.ndim != 2 or a.shape[1] <= R_GRIP:
            if self._throttled("shape", 5.0):
                log.warning("[grasp-check] action chunk shape %s has no right-grip col (need 2-D with "
                            ">%d cols, i.e. a 20-col dual-arm row) -> grip tracking OFF, detector will "
                            "never fire", a.shape, R_GRIP)
            return
        rgrip = float(a[-1, R_GRIP])
        closed_now = bool(rgrip >= self.closed_grip_min)
        if self._throttled("grip", 2.0):                  # heartbeat: proves note_action is running
            log.info("[grasp-check] right grip=%.1f (close>=%.1f) closed=%s armed=%s checked=%s",
                     rgrip, self.closed_grip_min, closed_now, self._close_t is not None, self._checked)
        if closed_now and not self._prev_closed:          # open -> close: attempt starts
            self._close_t = time.monotonic()
            self._checked = False
            log.info("[grasp-check] open->close (grip=%.1f) -> attempt ARMED, detector check in %.1fs",
                     rgrip, self.settle_sec)
        elif not closed_now and self._prev_closed and not self._checked:
            self._close_t = None                          # released before the check -> cancel it
            log.info("[grasp-check] released before the %.1fs check -> attempt cancelled", self.settle_sec)
        self._prev_closed = closed_now

    # -- hook 2: every loop BEFORE predict. True => entered recovery (skip predict)
    def maybe_start(self, env, obs) -> bool:
        if self._retreating or self._close_t is None or self._checked:
            return False
        waited = time.monotonic() - self._close_t
        if waited < self.settle_sec:
            if self._throttled("settle", 1.0):
                log.info("[grasp-check] grasp armed, settling %.1f/%.1fs before detector runs",
                         waited, self.settle_sec)
            return False
        self._checked = True
        log.info("[grasp-check] settle elapsed -> running detector on right wrist frame")

        frame = obs.get("handr_imgs")
        if isinstance(frame, (list, tuple)):
            frame = frame[-1] if frame else None
        if frame is None:
            log.warning("[grasp-check] no 'handr_imgs' right-wrist frame in obs (keys=%s) -> "
                        "cannot run detector", list(obs.keys()))
            return False

        p = self.det.p_empty(frame)
        if p < self.det.threshold:
            log.info("[grasp-check] detector P_empty=%.2f < thr=%.2f -> grasp HELD, no recovery",
                     p, self.det.threshold)
            return False

        self._begin_retreat(env, obs)
        log.warning("[recovery] missed grasp (P_empty=%.2f >= thr=%.2f) -> streaming retreat",
                    p, self.det.threshold)
        return True

    @property
    def is_retreating(self) -> bool:
        return self._retreating

    # -- hook 3: every loop WHILE retreating; streams the retreat, exits when done
    def pump(self, env):
        if not self._retreating:
            return
        if time.monotonic() > self._retreat_deadline:
            log.warning("[recovery] retreat timed out -> handing back to policy")
            return self._finish()
        # Top the queue up FIRST (a no-op once every row is queued), so the empty queue that
        # _begin_retreat just left behind can't read as "done" on the opening pump.
        ok, reason = env.append_actions(self._retreat_traj, self._anchor_id)
        if not ok:                                         # IK/validation hard-fail: don't spin
            log.warning("[recovery] retreat append refused (%s) -> handing back to policy", reason)
            return self._finish()
        # Retreat done when the clock has REACHED the last retreat row (ids run anchor.._end_id, so
        # the clock tops out AT _end_id — never past it) AND the robot queue has drained. robot_pending
        # counts _robot_q (what the retreat streams into); env.queue_empty() is the preview-sim queue,
        # which the retreat never touches, so it is NOT the drain signal here.
        cur, _ = env.queue_status()
        if cur >= self._end_id and env.robot_pending == 0:  # retreat fully executed
            log.info("[recovery] retreat complete -> policy re-approaches")
            self._finish()

    # -- internals --------------------------------------------------------------
    def _begin_retreat(self, env, obs):
        with env._lock:                                    # clear queue (non-estop half of lock_robot)
            env._robot_q.clear()
            env._staged_release.clear()
            env._queued_through = -1                       # next append re-anchors to the clock
        cur, _ = env.queue_status()
        left = np.asarray(obs["robotl_eef_pos"][-1], dtype=float)     # [pos3, rot6d6] held
        left_grip = float(obs["robot0_grip"][-1][0])
        right = np.asarray(obs["robotr_eef_pos"][-1], dtype=float)    # [pos3, rot6d6]
        r_pos0, r_rot = right[0:3], right[3:9]
        r_pos1 = r_pos0 + self.retreat_offset
        # interpolate current -> lifted target over retreat_rows; gripper OPEN from row 0
        traj = []
        for k in range(1, self.retreat_rows + 1):
            rp = r_pos0 + (k / self.retreat_rows) * (r_pos1 - r_pos0)
            traj.append([*left, left_grip, *rp, *r_rot, self.open_grip])
        self._retreat_traj = traj
        self._anchor_id = int(cur)
        self._end_id = int(cur) + self.retreat_rows - 1
        self._retreat_deadline = time.monotonic() + self.retreat_timeout_sec
        self._retreating = True

    def _finish(self):
        self._retreating = False
        self._retreat_traj = None
        # gripper is open now; re-arm grip tracking so the next close is a fresh attempt
        self._prev_closed = False
        self._close_t = None

    def reset(self):
        self._finish()
