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

python robot_control_gui.py 
Init glog with processor name:python3.10, pid:622872
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
[release-timing] 1 substeps/s | per-substep avg: total 8.9ms = recorder 0.3 + firmware 0.0 + dispatch 8.6 (STEP_TIME=8.3ms)
[release-timing] 113 substeps/s | per-substep avg: total 8.9ms = recorder 0.3 + firmware 0.0 + dispatch 8.6 (STEP_TIME=8.3ms)
[release-timing] 113 substeps/s | per-substep avg: total 8.9ms = recorder 0.3 + firmware 0.0 + dispatch 8.6 (STEP_TIME=8.3ms)
[release-timing] 113 substeps/s | per-substep avg: total 8.9ms = recorder 0.3 + firmware 0.0 + dispatch 8.6 (STEP_TIME=8.3ms)
[release-timing] 113 substeps/s | per-substep avg: total 8.9ms = recorder 0.3 + firmware 0.0 + dispatch 8.6 (STEP_TIME=8.3ms)
[release-timing] 114 substeps/s | per-substep avg: total 8.8ms = recorder 0.3 + firmware 0.0 + dispatch 8.6 (STEP_TIME=8.3ms)
[release-timing] 113 substeps/s | per-substep avg: total 8.9ms = recorder 0.3 + firmware 0.0 + dispatch 8.6 (STEP_TIME=8.3ms)
[release-timing] 113 substeps/s | per-substep avg: total 8.9ms = recorder 0.3 + firmware 0.0 + dispatch 8.6 (STEP_TIME=8.3ms)
[release-timing] 114 substeps/s | per-substep avg: total 8.8ms = recorder 0.2 + firmware 0.0 + dispatch 8.6 (STEP_TIME=8.3ms)
[release-timing] 114 substeps/s | per-substep avg: total 8.8ms = recorder 0.2 + firmware 0.0 + dispatch 8.6 (STEP_TIME=8.3ms)
[release-timing] 114 substeps/s | per-substep avg: total 8.8ms = recorder 0.3 + firmware 0.0 + dispatch 8.6 (STEP_TIME=8.3ms)
[release-timing] 112 substeps/s | per-substep avg: total 8.9ms = recorder 0.3 + firmware 0.0 + dispatch 8.6 (STEP_TIME=8.3ms)
[release-timing] 112 substeps/s | per-substep avg: total 8.9ms = recorder 0.4 + firmware 0.0 + dispatch 8.6 (STEP_TIME=8.3ms)
[HumanoidEnv] release refused: nothing sim-validated (run 执行 first).
[HumanoidEnv] sim-validated 1381 points (id 2); 2762 substep(s) staged for release.
[HumanoidEnv] released 1381 validated pts (+91 ramp-in) to robot.
[HumanoidEnv] dispatch-ramp: |Δq|=0.632 rad exceeds cap 0.033 -> streaming 19 bounded substeps
[release-timing] 112 substeps/s | per-substep avg: total 10.2ms = recorder 0.3 + firmware 0.0 + dispatch 10.0 (STEP_TIME=8.3ms)
[HumanoidEnv] E-STOP: latched; dropped 4228 pending/staged cmds; holding pose.
[HumanoidEnv] release refused: E-stop latched (press 复位 to reset).
[HumanoidEnv] E-stop reset; release re-enabled (run 执行 then 释放).
[safety] ALL INVARIANTS PASS (C1 C2 C3 C4 C5 H1)
[startup] safety pre-flight passed.

SLAM 模块初始化成功（已解冻关节状态）
[INFO] [1783498000.914322428] [wheel_controller_example]: Wheel Controller Example node started. Publishing a target pose for the robot base.
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
Exception ignored in: <function Node.__del__ at 0x7fb9017df520>
Traceback (most recent call last):
  File "/home/mujin/miniconda3/envs/ros2/lib/python3.10/site-packages/cosine_bus/agibotdds_py3/agibotdds.py", line 225, in __del__
    for publisher in self.list_publisher:
AttributeError: 'Node' object has no attribute 'list_publisher'
HTTP/JSON listening on http://0.0.0.0:9000/robot_info  (Ctrl-C to stop)
16:06:45 INFO real_world.grasp_recovery: GraspRecoveryMonitor ready: settle=5.0s closed_grip_min=60.0 roi=(330, 265, 625, 480) thr=0.50 device=cpu
16:06:45 INFO real_world.postprocess: smoothing set: radius=4 sigma=0.85 m=0.500
16:06:45 INFO real_world.postprocess: te_buffer_len set: 6
[HumanoidEnv] speed_scale=2.540 -> substeps_per_row=5, ramp_joint_step=0.0333
[tuning] restored from /home/mujin/workspaces/humanoid/tuning_config.json
16:06:46 INFO real_world.postprocess: smoothing set: radius=4 sigma=0.85 m=0.500
16:06:46 INFO real_world.postprocess: te_buffer_len set: 6
[HumanoidEnv] speed_scale=2.540 -> substeps_per_row=5, ramp_joint_step=0.0333
[HumanoidEnv]: started (collect=on, exec=on, real=on).
16:06:46 INFO real_world.inference_controller: InferenceController ready (env owned by caller).
16:06:46 INFO xr_examples.pico_vr_server.server: Downstream listening on 0.0.0.0:5555
16:06:46 INFO xr_examples.pico_vr_server.server: Upstream listening on 0.0.0.0:5556
startThreads creating 1 threads.
starting thread 0
started thread 0 
argc=2
argv[0] = --unused
argv[1] = --start_demo_name=Physics Server
ExampleBrowserThreadFunc started
X11 functions dynamically loaded using dlopen/dlsym OK!
X11 functions dynamically loaded using dlopen/dlsym OK!
Creating context
Created GL 3.3 context
Direct GLX rendering context obtained
Making context current
GL_VENDOR=Intel
GL_RENDERER=Mesa Intel(R) UHD Graphics (CML GT2)
GL_VERSION=4.6 (Core Profile) Mesa 23.2.1-1ubuntu3.1~22.04.4
GL_SHADING_LANGUAGE_VERSION=4.60
pthread_getconcurrency()=0
Version = 4.6 (Core Profile) Mesa 23.2.1-1ubuntu3.1~22.04.4
Vendor = Intel
Renderer = Mesa Intel(R) UHD Graphics (CML GT2)
b3Printf: Selected demo: Physics Server
startThreads creating 1 threads.
starting thread 0
started thread 0 
MotionThreadFunc thread started
b3Printf: b3Warning[examples/Importers/ImportURDFDemo/BulletUrdfImporter.cpp,126]:

b3Printf: No inertial data for link, using mass=1, localinertiadiagonal = 1,1,1, identity local inertial frame
b3Printf: b3Warning[examples/Importers/ImportURDFDemo/BulletUrdfImporter.cpp,126]:

b3Printf: link-arm
ven = Intel
Workaround for some crash in the Intel OpenGL driver on Linux/Ubuntu
b3Printf: b3Warning[examples/Importers/ImportURDFDemo/BulletUrdfImporter.cpp,126]:

b3Printf: No inertial data for link, using mass=1, localinertiadiagonal = 1,1,1, identity local inertial frame
b3Printf: b3Warning[examples/Importers/ImportURDFDemo/BulletUrdfImporter.cpp,126]:

b3Printf: gripper_center
b3Printf: b3Warning[examples/Importers/ImportURDFDemo/BulletUrdfImporter.cpp,126]:

b3Printf: No inertial data for link, using mass=1, localinertiadiagonal = 1,1,1, identity local inertial frame
b3Printf: b3Warning[examples/Importers/ImportURDFDemo/BulletUrdfImporter.cpp,126]:

b3Printf: right_gripper_center
ven = Intel
Workaround for some crash in the Intel OpenGL driver on Linux/Ubuntu
16:07:06 INFO real_world.inference_controller: [auto] START -> robot | server 10.12.11.144:9000 | uncapped (latency-bound) | temporal-ensemble ON | grasp-recovery ON
[CameraHub] camera ON: head -> ['head']
[CameraHub] camera ON: hand_left -> ['hand_left', 'head']
[CameraHub] camera ON: hand_right -> ['hand_left', 'hand_right', 'head']
[ObsCollector] warming up: waiting for cameras ['head', 'hand_left', 'hand_right']…
[CameraHub] camera subscribed (ON): hand_right
[CameraHub] camera subscribed (ON): head
[CameraHub] camera subscribed (ON): hand_left
16:07:06 INFO real_world.grasp_recovery: [grasp-check] right grip=0.4 (close>=60.0) closed=False armed=False checked=False
[pipeline] append: obs_row=0, clock=0, appended ids 0..6 (+0 catch-up) -> queue 35
16:07:06 INFO real_world.inference_controller: [infer] #1 | start 16:07:06.354 end 16:07:06.778 | took 423.5 ms | carried ids 0..13 | robot@id 0 (queued->6, lead 6) | srv 125 ms | L-gap 0.4cm R-gap 1.1cm
[release-timing] 1 substeps/s | per-substep avg: total 25.5ms = recorder 8.3 + firmware 0.7 + dispatch 16.4 (STEP_TIME=8.3ms)
16:07:07 INFO real_world.inference_controller: [infer] #2 | start 16:07:06.781 end 16:07:07.046 | took 265.1 ms | carried ids 3..16 | robot@id 2 (queued->7, lead 5) | srv 117 ms | anchor idx -1 OOR (buf (11, 20))
16:07:07 INFO real_world.inference_controller: [infer] #3 | start 16:07:07.047 end 16:07:07.369 | took 322.8 ms | carried ids 4..17 | robot@id 4 (queued->9, lead 5) | srv 109 ms | L-gap 0.1cm R-gap 2.2cm
[release-timing] 35 substeps/s | per-substep avg: total 28.7ms = recorder 11.8 + firmware 0.6 + dispatch 16.4 (STEP_TIME=8.3ms)
[pipeline] append: obs_row=6, clock=7, appended ids 10..12 (+0 catch-up) -> queue 28
16:07:07 INFO real_world.inference_controller: [infer] #4 | start 16:07:07.374 end 16:07:07.874 | took 499.8 ms | carried ids 6..19 | robot@id 7 (queued->12, lead 5) | srv 110 ms | L-gap 0.3cm R-gap 2.9cm
16:07:08 INFO real_world.inference_controller: [infer] #5 | start 16:07:07.875 end 16:07:08.239 | took 363.6 ms | carried ids 9..22 | robot@id 9 (queued->14, lead 5) | srv 110 ms | L-gap 0.4cm R-gap 3.4cm
16:07:08 INFO real_world.inference_controller: [infer] #6 | start 16:07:08.240 end 16:07:08.620 | took 380.2 ms | carried ids 11..24 | robot@id 11 (queued->16, lead 5) | srv 143 ms | L-gap 0.9cm R-gap 3.2cm
[release-timing] 28 substeps/s | per-substep avg: total 36.1ms = recorder 16.5 + firmware 1.8 + dispatch 17.8 (STEP_TIME=8.3ms)
16:07:08 INFO real_world.grasp_recovery: [grasp-check] right grip=1.2 (close>=60.0) closed=False armed=False checked=False
[pipeline] append: obs_row=13, clock=13, appended ids 17..19 (+0 catch-up) -> queue 32
16:07:09 INFO real_world.inference_controller: [infer] #7 | start 16:07:08.640 end 16:07:09.092 | took 452.0 ms | carried ids 13..26 | robot@id 13 (queued->19, lead 6) | srv 113 ms | L-gap 0.5cm R-gap 3.2cm
16:07:09 INFO real_world.inference_controller: [infer] #8 | start 16:07:09.105 end 16:07:09.487 | took 382.4 ms | carried ids 16..29 | robot@id 16 (queued->21, lead 5) | srv 128 ms | L-gap 0.3cm R-gap 2.8cm
[release-timing] 26 substeps/s | per-substep avg: total 40.1ms = recorder 20.5 + firmware 2.7 + dispatch 16.9 (STEP_TIME=8.3ms)
16:07:09 INFO real_world.inference_controller: [infer] #9 | start 16:07:09.493 end 16:07:09.915 | took 422.0 ms | carried ids 18..31 | robot@id 18 (queued->23, lead 5) | srv 119 ms | L-gap 0.1cm R-gap 3.0cm
[pipeline] append: obs_row=20, clock=19, appended ids 24..25 (+0 catch-up) -> queue 31
16:07:10 INFO real_world.inference_controller: [infer] #10 | start 16:07:09.918 end 16:07:10.284 | took 366.0 ms | carried ids 20..33 | robot@id 19 (queued->25, lead 6) | srv 107 ms | anchor idx -1 OOR (buf (12, 20))
16:07:10 INFO real_world.inference_controller: [infer] #11 | start 16:07:10.285 end 16:07:10.678 | took 393.1 ms | carried ids 22..35 | robot@id 21 (queued->27, lead 6) | srv 123 ms | anchor idx -1 OOR (buf (11, 20))
[release-timing] 26 substeps/s | per-substep avg: total 38.4ms = recorder 20.1 + firmware 1.0 + dispatch 17.3 (STEP_TIME=8.3ms)
16:07:10 INFO real_world.grasp_recovery: [grasp-check] right grip=0.1 (close>=60.0) closed=False armed=False checked=False
16:07:11 INFO real_world.inference_controller: [infer] #12 | start 16:07:10.678 end 16:07:11.079 | took 400.5 ms | carried ids 24..37 | robot@id 24 (queued->29, lead 5) | srv 137 ms | L-gap 0.5cm R-gap 3.2cm
[pipeline] append: obs_row=26, clock=26, appended ids 30..31 (+0 catch-up) -> queue 28
16:07:11 INFO real_world.inference_controller: [infer] #13 | start 16:07:11.080 end 16:07:11.511 | took 431.0 ms | carried ids 26..39 | robot@id 26 (queued->31, lead 5) | srv 142 ms | L-gap 0.5cm R-gap 3.4cm
16:07:11 INFO real_world.inference_controller: [infer] #14 | start 16:07:11.513 end 16:07:11.902 | took 389.0 ms | carried ids 28..41 | robot@id 28 (queued->33, lead 5) | srv 129 ms | L-gap 0.3cm R-gap 3.8cm
[release-timing] 26 substeps/s | per-substep avg: total 39.2ms = recorder 18.4 + firmware 2.0 + dispatch 18.8 (STEP_TIME=8.3ms)
16:07:12 INFO real_world.inference_controller: [infer] #15 | start 16:07:11.905 end 16:07:12.402 | took 496.7 ms | carried ids 30..43 | robot@id 31 (queued->36, lead 5) | srv 102 ms | L-gap 0.5cm R-gap 3.6cm
[pipeline] append: obs_row=33, clock=33, appended ids 37..38 (+0 catch-up) -> queue 29
16:07:12 INFO real_world.inference_controller: [infer] #16 | start 16:07:12.404 end 16:07:12.782 | took 378.1 ms | carried ids 33..46 | robot@id 33 (queued->38, lead 5) | srv 126 ms | L-gap 0.6cm R-gap 3.5cm
[release-timing] 25 substeps/s | per-substep avg: total 40.3ms = recorder 22.7 + firmware 0.8 + dispatch 16.7 (STEP_TIME=8.3ms)
16:07:13 INFO real_world.grasp_recovery: [grasp-check] right grip=0.4 (close>=60.0) closed=False armed=False checked=False
16:07:13 INFO real_world.inference_controller: [infer] #17 | start 16:07:12.784 end 16:07:13.329 | took 545.1 ms | carried ids 35..48 | robot@id 35 (queued->41, lead 6) | srv 174 ms | L-gap 0.7cm R-gap 3.2cm
[pipeline] append: obs_row=38, clock=37, appended ids 42..43 (+0 catch-up) -> queue 30
16:07:13 INFO real_world.inference_controller: [infer] #18 | start 16:07:13.331 end 16:07:13.787 | took 455.5 ms | carried ids 38..51 | robot@id 37 (queued->43, lead 6) | srv 133 ms | anchor idx -1 OOR (buf (11, 20))
[release-timing] 26 substeps/s | per-substep avg: total 39.1ms = recorder 18.8 + firmware 1.4 + dispatch 18.9 (STEP_TIME=8.3ms)
16:07:14 INFO real_world.inference_controller: [infer] #19 | start 16:07:13.790 end 16:07:14.306 | took 515.9 ms | carried ids 40..53 | robot@id 40 (queued->45, lead 5) | srv 146 ms | L-gap 0.3cm R-gap 3.3cm
[pipeline] append: obs_row=42, clock=42, appended ids 46..48 (+0 catch-up) -> queue 30
16:07:14 INFO real_world.inference_controller: [infer] #20 | start 16:07:14.309 end 16:07:14.836 | took 527.3 ms | carried ids 42..55 | robot@id 42 (queued->48, lead 6) | srv 128 ms | L-gap 0.1cm R-gap 3.3cm
[release-timing] 24 substeps/s | per-substep avg: total 41.8ms = recorder 21.2 + firmware 3.0 + dispatch 17.6 (STEP_TIME=8.3ms)
16:07:15 INFO real_world.inference_controller: [infer] #21 | start 16:07:14.837 end 16:07:15.265 | took 428.7 ms | carried ids 45..58 | robot@id 45 (queued->50, lead 5) | srv 142 ms | L-gap 0.3cm R-gap 3.2cm
16:07:15 INFO real_world.grasp_recovery: [grasp-check] right grip=11.7 (close>=60.0) closed=False armed=False checked=False
16:07:15 INFO real_world.inference_controller: [infer] #22 | start 16:07:15.266 end 16:07:15.711 | took 445.5 ms | carried ids 47..60 | robot@id 47 (queued->52, lead 5) | srv 158 ms | L-gap 0.4cm R-gap 3.7cm
[release-timing] 26 substeps/s | per-substep avg: total 39.7ms = recorder 19.3 + firmware 1.5 + dispatch 18.8 (STEP_TIME=8.3ms)
[pipeline] append: obs_row=49, clock=49, appended ids 53..55 (+0 catch-up) -> queue 30
16:07:16 INFO real_world.inference_controller: [infer] #23 | start 16:07:15.712 end 16:07:16.292 | took 579.9 ms | carried ids 49..62 | robot@id 49 (queued->55, lead 6) | srv 186 ms | L-gap 0.6cm R-gap 3.5cm
16:07:16 INFO real_world.inference_controller: [infer] #24 | start 16:07:16.294 end 16:07:16.892 | took 598.5 ms | carried ids 52..65 | robot@id 53 (queued->58, lead 5) | srv 215 ms | L-gap 0.2cm R-gap 3.3cm
[release-timing] 26 substeps/s | per-substep avg: total 39.5ms = recorder 19.1 + firmware 1.7 + dispatch 18.6 (STEP_TIME=8.3ms)
[pipeline] append: obs_row=55, clock=55, appended ids 59..61 (+0 catch-up) -> queue 30
16:07:17 INFO real_world.inference_controller: [infer] #25 | start 16:07:16.894 end 16:07:17.417 | took 522.2 ms | carried ids 55..68 | robot@id 56 (queued->61, lead 5) | srv 157 ms | L-gap 0.2cm R-gap 3.3cm
16:07:17 INFO real_world.grasp_recovery: [grasp-check] right grip=0.7 (close>=60.0) closed=False armed=False checked=False
16:07:17 INFO real_world.inference_controller: [infer] #26 | start 16:07:17.422 end 16:07:17.991 | took 569.4 ms | carried ids 58..71 | robot@id 58 (queued->64, lead 6) | srv 154 ms | L-gap 0.9cm R-gap 3.5cm
[release-timing] 26 substeps/s | per-substep avg: total 42.0ms = recorder 21.0 + firmware 1.4 + dispatch 19.7 (STEP_TIME=8.3ms)
[pipeline] append: obs_row=61, clock=61, appended ids 65..66 (+0 catch-up) -> queue 28
16:07:18 INFO real_world.inference_controller: [infer] #27 | start 16:07:17.994 end 16:07:18.522 | took 528.1 ms | carried ids 61..74 | robot@id 61 (queued->66, lead 5) | srv 127 ms | L-gap 0.5cm R-gap 3.7cm
16:07:19 INFO real_world.inference_controller: [infer] #28 | start 16:07:18.525 end 16:07:19.001 | took 476.0 ms | carried ids 63..76 | robot@id 63 (queued->68, lead 5) | srv 108 ms | L-gap 0.3cm R-gap 3.2cm
[release-timing] 24 substeps/s | per-substep avg: total 43.3ms = recorder 22.3 + firmware 4.0 + dispatch 16.9 (STEP_TIME=8.3ms)
16:07:19 INFO real_world.inference_controller: [infer] #29 | start 16:07:19.003 end 16:07:19.468 | took 465.4 ms | carried ids 65..78 | robot@id 65 (queued->70, lead 5) | srv 132 ms | L-gap 0.2cm R-gap 3.6cm
[pipeline] append: obs_row=67, clock=68, appended ids 71..73 (+0 catch-up) -> queue 28
16:07:19 INFO real_world.inference_controller: [infer] #30 | start 16:07:19.474 end 16:07:19.984 | took 510.4 ms | carried ids 67..80 | robot@id 68 (queued->73, lead 5) | srv 111 ms | L-gap 0.4cm R-gap 3.8cm
[release-timing] 26 substeps/s | per-substep avg: total 39.2ms = recorder 19.8 + firmware 3.0 + dispatch 16.4 (STEP_TIME=8.3ms)
16:07:20 INFO real_world.grasp_recovery: [grasp-check] right grip=0.3 (close>=60.0) closed=False armed=False checked=False
16:07:20 INFO real_world.inference_controller: [infer] #31 | start 16:07:19.985 end 16:07:20.432 | took 447.7 ms | carried ids 70..83 | robot@id 70 (queued->75, lead 5) | srv 125 ms | L-gap 0.4cm R-gap 3.5cm
[pipeline] append: obs_row=72, clock=72, appended ids 76..77 (+0 catch-up) -> queue 29
16:07:21 INFO real_world.inference_controller: [infer] #32 | start 16:07:20.433 end 16:07:21.020 | took 587.2 ms | carried ids 72..85 | robot@id 72 (queued->77, lead 5) | srv 130 ms | L-gap 1.0cm R-gap 3.1cm
[release-timing] 20 substeps/s | per-substep avg: total 51.0ms = recorder 27.8 + firmware 2.1 + dispatch 21.1 (STEP_TIME=8.3ms)
16:07:21 INFO real_world.inference_controller: [infer] #33 | start 16:07:21.021 end 16:07:21.586 | took 565.2 ms | carried ids 74..87 | robot@id 74 (queued->80, lead 6) | srv 205 ms | L-gap 0.2cm R-gap 2.9cm
16:07:21 INFO real_world.grasp_recovery: [grasp-check] open->close (grip=87.4) -> attempt ARMED, detector check in 5.0s
[pipeline] append: obs_row=77, clock=77, appended ids 81..82 (+0 catch-up) -> queue 29
16:07:22 INFO real_world.inference_controller: [infer] #34 | start 16:07:21.594 end 16:07:22.144 | took 549.8 ms | carried ids 77..90 | robot@id 77 (queued->82, lead 5) | srv 132 ms | L-gap 0.8cm R-gap 3.2cm
16:07:22 INFO real_world.grasp_recovery: [grasp-check] grasp armed, settling 0.3/5.0s before detector runs
[release-timing] 22 substeps/s | per-substep avg: total 46.9ms = recorder 20.8 + firmware 4.1 + dispatch 22.0 (STEP_TIME=8.3ms)
16:07:22 INFO real_world.grasp_recovery: [grasp-check] right grip=0.2 (close>=60.0) closed=False armed=True checked=False
16:07:22 INFO real_world.grasp_recovery: [grasp-check] released before the 5.0s check -> attempt cancelled
16:07:22 INFO real_world.inference_controller: [infer] #35 | start 16:07:22.162 end 16:07:22.691 | took 529.2 ms | carried ids 79..92 | robot@id 79 (queued->84, lead 5) | srv 134 ms | L-gap 0.1cm R-gap 3.5cm
[pipeline] append: obs_row=81, clock=82, appended ids 85..87 (+0 catch-up) -> queue 29
16:07:23 INFO real_world.inference_controller: [infer] #36 | start 16:07:22.691 end 16:07:23.167 | took 475.6 ms | carried ids 81..94 | robot@id 82 (queued->87, lead 5) | srv 103 ms | L-gap 0.4cm R-gap 3.6cm
[release-timing] 26 substeps/s | per-substep avg: total 39.5ms = recorder 16.5 + firmware 3.5 + dispatch 19.6 (STEP_TIME=8.3ms)
16:07:23 INFO real_world.inference_controller: [infer] #37 | start 16:07:23.169 end 16:07:23.697 | took 527.8 ms | carried ids 84..97 | robot@id 84 (queued->90, lead 6) | srv 123 ms | L-gap 0.2cm R-gap 3.5cm
[pipeline] append: obs_row=87, clock=87, appended ids 91..92 (+0 catch-up) -> queue 28
16:07:24 INFO real_world.inference_controller: [infer] #38 | start 16:07:23.700 end 16:07:24.210 | took 510.3 ms | carried ids 87..100 | robot@id 87 (queued->92, lead 5) | srv 111 ms | L-gap 0.2cm R-gap 3.5cm
[release-timing] 25 substeps/s | per-substep avg: total 39.8ms = recorder 19.5 + firmware 1.5 + dispatch 18.8 (STEP_TIME=8.3ms)
16:07:24 INFO real_world.inference_controller: [infer] #39 | start 16:07:24.212 end 16:07:24.567 | took 355.4 ms | carried ids 89..102 | robot@id 88 (queued->94, lead 6) | srv 116 ms | anchor idx -1 OOR (buf (12, 20))
16:07:24 INFO real_world.grasp_recovery: [grasp-check] right grip=0.1 (close>=60.0) closed=False armed=False checked=False
16:07:25 INFO real_world.inference_controller: [infer] #40 | start 16:07:24.571 end 16:07:25.090 | took 519.0 ms | carried ids 91..104 | robot@id 90 (queued->96, lead 6) | srv 162 ms | anchor idx -1 OOR (buf (11, 20))
[release-timing] 21 substeps/s | per-substep avg: total 48.2ms = recorder 26.7 + firmware 1.2 + dispatch 20.3 (STEP_TIME=8.3ms)
[pipeline] append: obs_row=93, clock=93, appended ids 97..98 (+0 catch-up) -> queue 28
16:07:25 INFO real_world.inference_controller: [infer] #41 | start 16:07:25.091 end 16:07:25.569 | took 477.7 ms | carried ids 93..106 | robot@id 93 (queued->98, lead 5) | srv 131 ms | L-gap 0.6cm R-gap 3.6cm
16:07:26 INFO real_world.inference_controller: [infer] #42 | start 16:07:25.573 end 16:07:26.166 | took 593.4 ms | carried ids 95..108 | robot@id 96 (queued->101, lead 5) | srv 127 ms | L-gap 0.5cm R-gap 3.6cm
[release-timing] 26 substeps/s | per-substep avg: total 38.9ms = recorder 15.9 + firmware 3.3 + dispatch 19.6 (STEP_TIME=8.3ms)
[pipeline] append: obs_row=98, clock=98, appended ids 102..103 (+0 catch-up) -> queue 27
16:07:26 INFO real_world.inference_controller: [infer] #43 | start 16:07:26.168 end 16:07:26.688 | took 520.0 ms | carried ids 98..111 | robot@id 98 (queued->103, lead 5) | srv 109 ms | L-gap 0.2cm R-gap 3.8cm
16:07:26 INFO real_world.grasp_recovery: [grasp-check] right grip=1.1 (close>=60.0) closed=False armed=False checked=False
16:07:27 INFO real_world.inference_controller: [infer] #44 | start 16:07:26.715 end 16:07:27.224 | took 509.2 ms | carried ids 100..113 | robot@id 100 (queued->106, lead 6) | srv 117 ms | L-gap 0.5cm R-gap 3.8cm
[release-timing] 22 substeps/s | per-substep avg: total 46.3ms = recorder 24.2 + firmware 2.3 + dispatch 19.8 (STEP_TIME=8.3ms)
[pipeline] append: obs_row=103, clock=103, appended ids 107..108 (+0 catch-up) -> queue 29
16:07:27 INFO real_world.inference_controller: [infer] #45 | start 16:07:27.232 end 16:07:27.717 | took 484.8 ms | carried ids 103..116 | robot@id 103 (queued->108, lead 5) | srv 128 ms | L-gap 0.5cm R-gap 3.1cm
16:07:28 INFO real_world.inference_controller: [infer] #46 | start 16:07:27.718 end 16:07:28.098 | took 379.7 ms | carried ids 105..118 | robot@id 104 (queued->110, lead 6) | srv 119 ms | anchor idx -1 OOR (buf (11, 20))
[release-timing] 23 substeps/s | per-substep avg: total 43.4ms = recorder 20.1 + firmware 2.7 + dispatch 20.7 (STEP_TIME=8.3ms)
16:07:28 INFO real_world.inference_controller: [infer] #47 | start 16:07:28.100 end 16:07:28.675 | took 574.5 ms | carried ids 107..120 | robot@id 108 (queued->113, lead 5) | srv 148 ms | L-gap 0.3cm R-gap 2.9cm
[pipeline] append: obs_row=110, clock=111, appended ids 114..116 (+0 catch-up) -> queue 28
16:07:29 INFO real_world.inference_controller: [infer] #48 | start 16:07:28.676 end 16:07:29.325 | took 649.2 ms | carried ids 110..123 | robot@id 111 (queued->116, lead 5) | srv 129 ms | L-gap 0.4cm R-gap 2.9cm
[release-timing] 27 substeps/s | per-substep avg: total 37.4ms = recorder 13.5 + firmware 2.1 + dispatch 21.9 (STEP_TIME=8.3ms)
16:07:29 INFO real_world.grasp_recovery: [grasp-check] right grip=0.0 (close>=60.0) closed=False armed=False checked=False
16:07:29 INFO real_world.inference_controller: [infer] #49 | start 16:07:29.329 end 16:07:29.933 | took 603.7 ms | carried ids 113..126 | robot@id 113 (queued->119, lead 6) | srv 108 ms | L-gap 0.6cm R-gap 3.1cm
16:07:30 INFO real_world.grasp_recovery: [grasp-check] open->close (grip=84.2) -> attempt ARMED, detector check in 5.0s
[release-timing] 21 substeps/s | per-substep avg: total 48.9ms = recorder 23.2 + firmware 3.3 + dispatch 22.3 (STEP_TIME=8.3ms)
[pipeline] append: obs_row=116, clock=115, appended ids 120..121 (+0 catch-up) -> queue 30
16:07:30 INFO real_world.inference_controller: [infer] #50 | start 16:07:29.940 end 16:07:30.493 | took 553.4 ms | carried ids 116..129 | robot@id 115 (queued->121, lead 6) | srv 115 ms | anchor idx -1 OOR (buf (11, 20))
16:07:30 INFO real_world.grasp_recovery: [grasp-check] grasp armed, settling 0.3/5.0s before detector runs
16:07:30 INFO real_world.grasp_recovery: [grasp-check] released before the 5.0s check -> attempt cancelled
16:07:31 INFO real_world.inference_controller: [infer] #51 | start 16:07:30.494 end 16:07:31.084 | took 589.5 ms | carried ids 118..131 | robot@id 118 (queued->123, lead 5) | srv 168 ms | L-gap 0.7cm R-gap 3.2cm
16:07:31 INFO real_world.grasp_recovery: [grasp-check] open->close (grip=99.6) -> attempt ARMED, detector check in 5.0s
[release-timing] 25 substeps/s | per-substep avg: total 41.4ms = recorder 18.1 + firmware 3.7 + dispatch 19.6 (STEP_TIME=8.3ms)
[pipeline] append: obs_row=120, clock=121, appended ids 124..126 (+0 catch-up) -> queue 28
16:07:31 INFO real_world.inference_controller: [infer] #52 | start 16:07:31.085 end 16:07:31.653 | took 568.6 ms | carried ids 120..133 | robot@id 121 (queued->126, lead 5) | srv 117 ms | L-gap 0.2cm R-gap 3.6cm
16:07:31 INFO real_world.grasp_recovery: [grasp-check] grasp armed, settling 0.4/5.0s before detector runs
[HumanoidEnv] arm_joint_states bad read (shape (0,)); using last good.
16:07:31 INFO real_world.grasp_recovery: [grasp-check] right grip=1.2 (close>=60.0) closed=False armed=True checked=False
16:07:31 INFO real_world.grasp_recovery: [grasp-check] released before the 5.0s check -> attempt cancelled
16:07:32 INFO real_world.inference_controller: [infer] #53 | start 16:07:31.667 end 16:07:32.134 | took 467.0 ms | carried ids 123..136 | robot@id 123 (queued->128, lead 5) | srv 116 ms | L-gap 0.3cm R-gap 3.5cm
16:07:32 INFO real_world.inference_controller: [auto] STOP requested — draining queue (53 inferences this run).
[release-timing] 21 substeps/s | per-substep avg: total 48.4ms = recorder 23.4 + firmware 2.7 + dispatch 22.3 (STEP_TIME=8.3ms)
[pipeline] append: obs_row=125, clock=125, appended ids 129..131 (+0 catch-up) -> queue 30
16:07:32 INFO real_world.inference_controller: [infer] #54 | start 16:07:32.153 end 16:07:32.753 | took 600.6 ms | carried ids 125..138 | robot@id 125 (queued->131, lead 6) | srv 134 ms | L-gap 0.4cm R-gap 3.5cm
[release-timing] 31 substeps/s | per-substep avg: total 32.9ms = recorder 12.7 + firmware 1.4 + dispatch 18.7 (STEP_TIME=8.3ms)



""

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
        # The retreat streams straight to append_actions, BYPASSING pipeline.merge(), so its master
        # ids never enter the smoothed buffer. Clear that buffer now so the FIRST post-retreat merge
        # re-anchors from the live clock instead of materializing a stale pre-miss run that stops at
        # the retreat-id gap — which would starve append_actions and freeze the arm until an E-stop
        # re-anchor (exactly the "only lock_robot+reset revives it" symptom).
        env.pipeline.reset_merge()
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
