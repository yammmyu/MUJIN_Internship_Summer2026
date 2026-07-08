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
Init glog with processor name:python3.10, pid:619818
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
[HumanoidEnv] E-STOP: latched; dropped 4228 pending/staged cmds; holding pose.
[HumanoidEnv] release refused: E-stop latched (press 复位 to reset).
[HumanoidEnv] E-stop reset; release re-enabled (run 执行 then 释放).
[safety] ALL INVARIANTS PASS (C1 C2 C3 C4 C5 H1)
[startup] safety pre-flight passed.

SLAM 模块初始化成功（已解冻关节状态）
[INFO] [1783496618.202571282] [wheel_controller_example]: Wheel Controller Example node started. Publishing a target pose for the robot base.
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
Exception ignored in: <function Node.__del__ at 0x7a5e71cdb520>
Traceback (most recent call last):
  File "/home/mujin/miniconda3/envs/ros2/lib/python3.10/site-packages/cosine_bus/agibotdds_py3/agibotdds.py", line 225, in __del__
    for publisher in self.list_publisher:
AttributeError: 'Node' object has no attribute 'list_publisher'
HTTP/JSON listening on http://0.0.0.0:9000/robot_info  (Ctrl-C to stop)
15:43:42 INFO real_world.grasp_recovery: GraspRecoveryMonitor ready: settle=5.0s closed_grip_min=60.0 roi=(330, 265, 625, 480) thr=0.50 device=cpu
15:43:42 INFO real_world.postprocess: smoothing set: radius=4 sigma=0.85 m=0.500
15:43:42 INFO real_world.postprocess: te_buffer_len set: 6
[HumanoidEnv] speed_scale=2.540 -> substeps_per_row=5, ramp_joint_step=0.0333
[tuning] restored from /home/mujin/workspaces/humanoid/tuning_config.json
15:43:43 INFO real_world.postprocess: smoothing set: radius=4 sigma=0.85 m=0.500
15:43:43 INFO real_world.postprocess: te_buffer_len set: 6
[HumanoidEnv] speed_scale=2.540 -> substeps_per_row=5, ramp_joint_step=0.0333
[HumanoidEnv]: started (collect=on, exec=on, real=on).
15:43:43 INFO real_world.inference_controller: InferenceController ready (env owned by caller).
15:43:43 INFO xr_examples.pico_vr_server.server: Downstream listening on 0.0.0.0:5555
15:43:43 INFO xr_examples.pico_vr_server.server: Upstream listening on 0.0.0.0:5556
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
15:44:44 INFO real_world.inference_controller: [auto] START -> robot | server 10.12.11.144:9000 | uncapped (latency-bound) | temporal-ensemble ON | grasp-recovery ON
[CameraHub] camera ON: head -> ['head']
[CameraHub] camera ON: hand_left -> ['hand_left', 'head']
[CameraHub] camera ON: hand_right -> ['hand_left', 'hand_right', 'head']
[ObsCollector] warming up: waiting for cameras ['head', 'hand_left', 'hand_right']…
[CameraHub] camera subscribed (ON): head
[CameraHub] camera subscribed (ON): hand_left
[CameraHub] camera subscribed (ON): hand_right
15:44:45 INFO real_world.grasp_recovery: [grasp-check] right grip=0.6 (close>=60.0) closed=False armed=False checked=False
[pipeline] append: obs_row=0, clock=0, appended ids 0..6 (+0 catch-up) -> queue 45
15:44:45 INFO real_world.inference_controller: [infer] #1 | start 15:44:45.161 end 15:44:45.582 | took 420.9 ms | carried ids 0..13 | robot@id 0 (queued->6, lead 6) | srv 121 ms | L-gap 2.6cm R-gap 0.9cm
15:44:45 INFO real_world.inference_controller: [infer] #2 | start 15:44:45.588 end 15:44:45.789 | took 201.1 ms | carried ids 3..16 | robot@id 0 (queued->6, lead 6) | srv 98 ms | anchor idx -3 OOR (buf (11, 20))
15:44:46 INFO real_world.inference_controller: [infer] #3 | start 15:44:45.790 end 15:44:46.102 | took 311.5 ms | carried ids 3..16 | robot@id 2 (queued->7, lead 5) | srv 118 ms | anchor idx -1 OOR (buf (11, 20))
[pipeline] append: obs_row=4, clock=5, appended ids 8..9 (+0 catch-up) -> queue 71
15:44:46 INFO real_world.inference_controller: [infer] #4 | start 15:44:46.103 end 15:44:46.918 | took 815.2 ms | carried ids 4..17 | robot@id 5 (queued->9, lead 4) | srv 100 ms | L-gap 4.0cm R-gap 1.8cm
15:44:47 INFO real_world.inference_controller: [infer] #5 | start 15:44:46.923 end 15:44:47.456 | took 533.6 ms | carried ids 6..19 | robot@id 8 (queued->13, lead 5) | srv 116 ms | L-gap 2.1cm R-gap 2.3cm
15:44:47 INFO real_world.grasp_recovery: [grasp-check] right grip=0.5 (close>=60.0) closed=False armed=False checked=False
15:44:47 INFO real_world.inference_controller: [infer] #6 | start 15:44:47.460 end 15:44:47.762 | took 301.8 ms | carried ids 10..23 | robot@id 8 (queued->14, lead 6) | srv 111 ms | anchor idx -2 OOR (buf (12, 20))
15:44:47 INFO real_world.inference_controller: [infer] #7 | start 15:44:47.763 end 15:44:47.981 | took 218.6 ms | carried ids 11..24 | robot@id 8 (queued->14, lead 6) | srv 98 ms | anchor idx -3 OOR (buf (11, 20))
15:44:48 INFO real_world.inference_controller: [infer] #8 | start 15:44:47.983 end 15:44:48.281 | took 297.6 ms | carried ids 11..24 | robot@id 8 (queued->14, lead 6) | srv 129 ms | anchor idx -3 OOR (buf (11, 20))
15:44:48 INFO real_world.inference_controller: [infer] #9 | start 15:44:48.281 end 15:44:48.532 | took 250.7 ms | carried ids 11..24 | robot@id 8 (queued->14, lead 6) | srv 115 ms | anchor idx -3 OOR (buf (11, 20))
15:44:48 INFO real_world.inference_controller: [infer] #10 | start 15:44:48.532 end 15:44:48.812 | took 279.4 ms | carried ids 11..24 | robot@id 8 (queued->14, lead 6) | srv 118 ms | anchor idx -3 OOR (buf (11, 20))
15:44:49 INFO real_world.inference_controller: [infer] #11 | start 15:44:48.814 end 15:44:49.084 | took 270.1 ms | carried ids 11..24 | robot@id 8 (queued->14, lead 6) | srv 122 ms | anchor idx -3 OOR (buf (11, 20))
[pipeline] append: obs_row=11, clock=9, appended ids 15..15 (+0 catch-up) -> queue 31
15:44:49 INFO real_world.inference_controller: [infer] #12 | start 15:44:49.085 end 15:44:49.468 | took 383.0 ms | carried ids 11..24 | robot@id 9 (queued->15, lead 6) | srv 97 ms | anchor idx -2 OOR (buf (11, 20))
15:44:49 INFO real_world.grasp_recovery: [grasp-check] right grip=2.3 (close>=60.0) closed=False armed=False checked=False
15:44:49 INFO real_world.inference_controller: [infer] #13 | start 15:44:49.469 end 15:44:49.902 | took 432.7 ms | carried ids 12..25 | robot@id 12 (queued->17, lead 5) | srv 94 ms | L-gap 1.7cm R-gap 2.3cm
15:44:50 INFO real_world.inference_controller: [infer] #14 | start 15:44:49.903 end 15:44:50.384 | took 481.9 ms | carried ids 14..27 | robot@id 14 (queued->20, lead 6) | srv 126 ms | L-gap 2.6cm R-gap 2.3cm
[pipeline] append: obs_row=17, clock=17, appended ids 21..22 (+0 catch-up) -> queue 29
15:44:50 INFO real_world.inference_controller: [infer] #15 | start 15:44:50.385 end 15:44:50.801 | took 415.6 ms | carried ids 17..30 | robot@id 17 (queued->22, lead 5) | srv 118 ms | L-gap 3.2cm R-gap 2.5cm
15:44:51 INFO real_world.inference_controller: [infer] #16 | start 15:44:50.802 end 15:44:51.308 | took 506.8 ms | carried ids 19..32 | robot@id 20 (queued->25, lead 5) | srv 126 ms | L-gap 1.6cm R-gap 3.3cm
15:44:51 INFO real_world.inference_controller: [infer] #17 | start 15:44:51.314 end 15:44:51.710 | took 396.2 ms | carried ids 22..35 | robot@id 22 (queued->27, lead 5) | srv 141 ms | L-gap 1.1cm R-gap 3.6cm
15:44:52 INFO real_world.grasp_recovery: [grasp-check] right grip=5.0 (close>=60.0) closed=False armed=False checked=False
[pipeline] append: obs_row=24, clock=24, appended ids 28..30 (+0 catch-up) -> queue 30
15:44:52 INFO real_world.inference_controller: [infer] #18 | start 15:44:51.732 end 15:44:52.229 | took 497.1 ms | carried ids 24..37 | robot@id 24 (queued->30, lead 6) | srv 127 ms | L-gap 1.5cm R-gap 3.4cm
15:44:52 INFO real_world.inference_controller: [infer] #19 | start 15:44:52.246 end 15:44:52.676 | took 430.3 ms | carried ids 27..40 | robot@id 27 (queued->32, lead 5) | srv 134 ms | L-gap 1.7cm R-gap 3.3cm
[pipeline] append: obs_row=29, clock=30, appended ids 33..35 (+0 catch-up) -> queue 26
15:44:53 INFO real_world.inference_controller: [infer] #20 | start 15:44:52.682 end 15:44:53.245 | took 563.4 ms | carried ids 29..42 | robot@id 30 (queued->35, lead 5) | srv 154 ms | L-gap 2.0cm R-gap 4.2cm
15:44:53 INFO real_world.inference_controller: [infer] #21 | start 15:44:53.246 end 15:44:53.644 | took 397.6 ms | carried ids 32..45 | robot@id 32 (queued->37, lead 5) | srv 122 ms | L-gap 0.5cm R-gap 3.9cm
15:44:54 INFO real_world.inference_controller: [infer] #22 | start 15:44:53.649 end 15:44:54.156 | took 506.4 ms | carried ids 34..47 | robot@id 34 (queued->40, lead 6) | srv 119 ms | L-gap 0.9cm R-gap 4.0cm
15:44:54 INFO real_world.grasp_recovery: [grasp-check] right grip=3.2 (close>=60.0) closed=False armed=False checked=False
[pipeline] append: obs_row=37, clock=36, appended ids 41..42 (+0 catch-up) -> queue 30
15:44:54 INFO real_world.inference_controller: [infer] #23 | start 15:44:54.159 end 15:44:54.587 | took 428.1 ms | carried ids 37..50 | robot@id 36 (queued->42, lead 6) | srv 130 ms | anchor idx -1 OOR (buf (11, 20))
15:44:55 INFO real_world.inference_controller: [infer] #24 | start 15:44:54.588 end 15:44:55.041 | took 452.2 ms | carried ids 39..52 | robot@id 39 (queued->44, lead 5) | srv 112 ms | L-gap 1.8cm R-gap 3.7cm
15:44:55 INFO real_world.inference_controller: [infer] #25 | start 15:44:55.041 end 15:44:55.535 | took 493.7 ms | carried ids 41..54 | robot@id 42 (queued->47, lead 5) | srv 119 ms | L-gap 1.2cm R-gap 4.5cm
[pipeline] append: obs_row=44, clock=44, appended ids 48..49 (+0 catch-up) -> queue 29
15:44:56 INFO real_world.inference_controller: [infer] #26 | start 15:44:55.541 end 15:44:56.019 | took 478.3 ms | carried ids 44..57 | robot@id 44 (queued->49, lead 5) | srv 127 ms | L-gap 1.1cm R-gap 4.8cm
15:44:56 INFO real_world.inference_controller: [infer] #27 | start 15:44:56.023 end 15:44:56.474 | took 451.1 ms | carried ids 46..59 | robot@id 46 (queued->51, lead 5) | srv 113 ms | L-gap 1.0cm R-gap 5.0cm
15:44:56 INFO real_world.grasp_recovery: [grasp-check] right grip=6.5 (close>=60.0) closed=False armed=False checked=False
15:44:57 INFO real_world.inference_controller: [infer] #28 | start 15:44:56.476 end 15:44:56.979 | took 502.7 ms | carried ids 48..61 | robot@id 49 (queued->54, lead 5) | srv 121 ms | L-gap 1.1cm R-gap 4.8cm
[pipeline] append: obs_row=51, clock=51, appended ids 55..57 (+0 catch-up) -> queue 30
15:44:57 INFO real_world.inference_controller: [infer] #29 | start 15:44:57.008 end 15:44:57.575 | took 566.5 ms | carried ids 51..64 | robot@id 51 (queued->57, lead 6) | srv 118 ms | L-gap 1.3cm R-gap 5.2cm
15:44:58 INFO real_world.inference_controller: [infer] #30 | start 15:44:57.579 end 15:44:58.055 | took 475.9 ms | carried ids 54..67 | robot@id 54 (queued->59, lead 5) | srv 108 ms | L-gap 1.4cm R-gap 5.4cm
15:44:58 INFO real_world.inference_controller: [infer] #31 | start 15:44:58.110 end 15:44:58.570 | took 459.8 ms | carried ids 56..69 | robot@id 57 (queued->62, lead 5) | srv 121 ms | L-gap 1.8cm R-gap 5.6cm
15:44:58 INFO real_world.grasp_recovery: [grasp-check] right grip=1.1 (close>=60.0) closed=False armed=False checked=False
[pipeline] append: obs_row=59, clock=59, appended ids 63..65 (+0 catch-up) -> queue 30
15:44:59 INFO real_world.inference_controller: [infer] #32 | start 15:44:58.584 end 15:44:59.120 | took 536.2 ms | carried ids 59..72 | robot@id 59 (queued->65, lead 6) | srv 119 ms | L-gap 1.5cm R-gap 5.8cm
15:44:59 INFO real_world.inference_controller: [infer] #33 | start 15:44:59.128 end 15:44:59.550 | took 422.4 ms | carried ids 62..75 | robot@id 62 (queued->67, lead 5) | srv 118 ms | L-gap 1.5cm R-gap 5.7cm
15:45:00 INFO real_world.inference_controller: [infer] #34 | start 15:44:59.558 end 15:45:00.039 | took 481.5 ms | carried ids 64..77 | robot@id 64 (queued->69, lead 5) | srv 119 ms | L-gap 2.3cm R-gap 5.3cm
[pipeline] append: obs_row=66, clock=67, appended ids 70..72 (+0 catch-up) -> queue 28
15:45:00 INFO real_world.inference_controller: [infer] #35 | start 15:45:00.042 end 15:45:00.608 | took 566.3 ms | carried ids 66..79 | robot@id 67 (queued->72, lead 5) | srv 128 ms | L-gap 0.7cm R-gap 6.0cm
15:45:00 INFO real_world.grasp_recovery: [grasp-check] right grip=0.7 (close>=60.0) closed=False armed=False checked=False
15:45:01 INFO real_world.inference_controller: [infer] #36 | start 15:45:00.624 end 15:45:01.264 | took 639.9 ms | carried ids 69..82 | robot@id 69 (queued->75, lead 6) | srv 143 ms | L-gap 1.3cm R-gap 5.9cm
[pipeline] append: obs_row=72, clock=72, appended ids 76..77 (+0 catch-up) -> queue 29
15:45:01 INFO real_world.inference_controller: [infer] #37 | start 15:45:01.271 end 15:45:01.745 | took 474.0 ms | carried ids 72..85 | robot@id 72 (queued->77, lead 5) | srv 139 ms | L-gap 2.6cm R-gap 6.1cm
15:45:02 INFO real_world.inference_controller: [infer] #38 | start 15:45:01.754 end 15:45:02.354 | took 600.4 ms | carried ids 74..87 | robot@id 75 (queued->80, lead 5) | srv 105 ms | L-gap 2.2cm R-gap 7.1cm
[pipeline] append: obs_row=77, clock=77, appended ids 81..82 (+0 catch-up) -> queue 27
15:45:02 INFO real_world.inference_controller: [infer] #39 | start 15:45:02.356 end 15:45:02.815 | took 458.9 ms | carried ids 77..90 | robot@id 77 (queued->82, lead 5) | srv 155 ms | L-gap 1.7cm R-gap 5.3cm
15:45:02 INFO real_world.inference_controller: [auto] STOP requested — draining queue (39 inferences this run).
15:45:03 INFO real_world.grasp_recovery: [grasp-check] right grip=0.8 (close>=60.0) closed=False armed=False checked=False
15:45:03 INFO real_world.inference_controller: [infer] #40 | start 15:45:02.819 end 15:45:03.448 | took 628.9 ms | carried ids 79..92 | robot@id 80 (queued->85, lead 5) | srv 119 ms | L-gap 1.2cm R-gap 5.5cm
[CameraHub] camera unsubscribed (OFF): head
[CameraHub] camera unsubscribed (OFF): hand_left
[CameraHub] camera unsubscribed (OFF): hand_right
[HumanoidEnv] E-STOP: latched; dropped 0 pending/staged cmds; holding pose.
[HumanoidEnv] E-stop reset; release re-enabled (run 执行 then 释放).
15:45:15 INFO real_world.inference_controller: [auto] START -> robot | server 10.12.11.144:9000 | uncapped (latency-bound) | temporal-ensemble ON | grasp-recovery ON
[CameraHub] camera ON: head -> ['head']
[CameraHub] camera ON: hand_left -> ['hand_left', 'head']
[CameraHub] camera ON: hand_right -> ['hand_left', 'hand_right', 'head']
[ObsCollector] warming up: waiting for cameras ['head', 'hand_left', 'hand_right']…
[CameraHub] camera subscribed (ON): head
[CameraHub] camera subscribed (ON): hand_left
[CameraHub] camera subscribed (ON): hand_right
15:45:15 INFO real_world.grasp_recovery: [grasp-check] right grip=1.1 (close>=60.0) closed=False armed=False checked=False
[pipeline] append: obs_row=85, clock=85, appended ids 85..91 (+0 catch-up) -> queue 35
15:45:16 INFO real_world.inference_controller: [infer] #41 | start 15:45:15.749 end 15:45:16.353 | took 603.7 ms | carried ids 85..98 | robot@id 85 (queued->91, lead 6) | srv 134 ms | L-gap 0.8cm R-gap 2.2cm
15:45:16 INFO real_world.inference_controller: [infer] #42 | start 15:45:16.353 end 15:45:16.726 | took 372.3 ms | carried ids 88..101 | robot@id 86 (queued->92, lead 6) | srv 118 ms | anchor idx -2 OOR (buf (11, 20))
15:45:17 INFO real_world.inference_controller: [infer] #43 | start 15:45:16.726 end 15:45:17.271 | took 545.0 ms | carried ids 89..102 | robot@id 88 (queued->94, lead 6) | srv 127 ms | anchor idx -1 OOR (buf (11, 20))
[pipeline] append: obs_row=91, clock=91, appended ids 95..96 (+0 catch-up) -> queue 29
15:45:17 INFO real_world.inference_controller: [infer] #44 | start 15:45:17.272 end 15:45:17.750 | took 477.2 ms | carried ids 91..104 | robot@id 91 (queued->96, lead 5) | srv 135 ms | L-gap 2.3cm R-gap 4.3cm
15:45:18 INFO real_world.grasp_recovery: [grasp-check] right grip=1.1 (close>=60.0) closed=False armed=False checked=False
15:45:18 INFO real_world.inference_controller: [infer] #45 | start 15:45:17.751 end 15:45:18.232 | took 482.0 ms | carried ids 93..106 | robot@id 93 (queued->98, lead 5) | srv 112 ms | L-gap 0.5cm R-gap 4.1cm
[pipeline] append: obs_row=95, clock=96, appended ids 99..101 (+0 catch-up) -> queue 29
15:45:18 INFO real_world.inference_controller: [infer] #46 | start 15:45:18.237 end 15:45:18.864 | took 627.1 ms | carried ids 95..108 | robot@id 96 (queued->101, lead 5) | srv 174 ms | L-gap 0.8cm R-gap 4.8cm
15:45:19 INFO real_world.inference_controller: [infer] #47 | start 15:45:18.864 end 15:45:19.525 | took 660.7 ms | carried ids 98..111 | robot@id 99 (queued->104, lead 5) | srv 124 ms | L-gap 1.5cm R-gap 4.6cm
[pipeline] append: obs_row=101, clock=101, appended ids 105..106 (+0 catch-up) -> queue 26
15:45:20 INFO real_world.inference_controller: [infer] #48 | start 15:45:19.526 end 15:45:20.092 | took 565.8 ms | carried ids 101..114 | robot@id 101 (queued->106, lead 5) | srv 135 ms | L-gap 0.9cm R-gap 4.5cm
15:45:20 INFO real_world.grasp_recovery: [grasp-check] right grip=0.5 (close>=60.0) closed=False armed=False checked=False
15:45:20 INFO real_world.inference_controller: [infer] #49 | start 15:45:20.097 end 15:45:20.712 | took 614.6 ms | carried ids 103..116 | robot@id 104 (queued->109, lead 5) | srv 149 ms | L-gap 0.6cm R-gap 5.0cm
[pipeline] append: obs_row=106, clock=107, appended ids 110..112 (+0 catch-up) -> queue 29
15:45:21 INFO real_world.inference_controller: [infer] #50 | start 15:45:20.717 end 15:45:21.216 | took 499.6 ms | carried ids 106..119 | robot@id 107 (queued->112, lead 5) | srv 146 ms | L-gap 0.7cm R-gap 4.1cm
15:45:21 INFO real_world.inference_controller: [infer] #51 | start 15:45:21.217 end 15:45:21.697 | took 480.7 ms | carried ids 109..122 | robot@id 108 (queued->114, lead 6) | srv 144 ms | anchor idx -1 OOR (buf (11, 20))
15:45:22 INFO real_world.inference_controller: [infer] #52 | start 15:45:21.699 end 15:45:22.176 | took 476.8 ms | carried ids 111..124 | robot@id 110 (queued->116, lead 6) | srv 128 ms | anchor idx -1 OOR (buf (11, 20))
15:45:22 INFO real_world.grasp_recovery: [grasp-check] right grip=0.7 (close>=60.0) closed=False armed=False checked=False
[pipeline] append: obs_row=113, clock=113, appended ids 117..118 (+0 catch-up) -> queue 29
15:45:22 INFO real_world.inference_controller: [infer] #53 | start 15:45:22.177 end 15:45:22.662 | took 484.7 ms | carried ids 113..126 | robot@id 113 (queued->118, lead 5) | srv 97 ms | L-gap 0.3cm R-gap 4.4cm
15:45:23 INFO real_world.inference_controller: [infer] #54 | start 15:45:22.663 end 15:45:23.143 | took 479.9 ms | carried ids 115..128 | robot@id 115 (queued->120, lead 5) | srv 100 ms | L-gap 0.2cm R-gap 4.9cm
[pipeline] append: obs_row=117, clock=118, appended ids 121..123 (+0 catch-up) -> queue 28
15:45:23 INFO real_world.inference_controller: [infer] #55 | start 15:45:23.159 end 15:45:23.718 | took 558.7 ms | carried ids 117..130 | robot@id 118 (queued->123, lead 5) | srv 157 ms | L-gap 0.4cm R-gap 4.7cm
15:45:24 INFO real_world.inference_controller: [infer] #56 | start 15:45:23.728 end 15:45:24.201 | took 472.8 ms | carried ids 120..133 | robot@id 120 (queued->125, lead 5) | srv 126 ms | L-gap 0.1cm R-gap 3.9cm
15:45:24 INFO real_world.grasp_recovery: [grasp-check] right grip=0.7 (close>=60.0) closed=False armed=False checked=False
[pipeline] append: obs_row=122, clock=124, appended ids 126..129 (+0 catch-up) -> queue 29
15:45:24 INFO real_world.inference_controller: [infer] #57 | start 15:45:24.230 end 15:45:24.881 | took 651.3 ms | carried ids 122..135 | robot@id 124 (queued->129, lead 5) | srv 115 ms | L-gap 0.5cm R-gap 4.0cm
15:45:25 INFO real_world.inference_controller: [infer] #58 | start 15:45:24.882 end 15:45:25.372 | took 489.5 ms | carried ids 126..139 | robot@id 126 (queued->131, lead 5) | srv 112 ms | L-gap 0.6cm R-gap 4.4cm
15:45:25 INFO real_world.inference_controller: [infer] #59 | start 15:45:25.372 end 15:45:25.792 | took 420.0 ms | carried ids 128..141 | robot@id 128 (queued->133, lead 5) | srv 110 ms | L-gap 0.3cm R-gap 5.2cm
[pipeline] append: obs_row=130, clock=131, appended ids 134..136 (+0 catch-up) -> queue 29
15:45:26 INFO real_world.inference_controller: [infer] #60 | start 15:45:25.798 end 15:45:26.430 | took 632.6 ms | carried ids 130..143 | robot@id 131 (queued->136, lead 5) | srv 136 ms | L-gap 0.3cm R-gap 4.8cm
15:45:26 INFO real_world.grasp_recovery: [grasp-check] right grip=0.5 (close>=60.0) closed=False armed=False checked=False
15:45:26 INFO real_world.inference_controller: [infer] #61 | start 15:45:26.440 end 15:45:26.888 | took 448.4 ms | carried ids 133..146 | robot@id 133 (queued->138, lead 5) | srv 127 ms | L-gap 0.5cm R-gap 4.5cm
[pipeline] append: obs_row=135, clock=135, appended ids 139..141 (+0 catch-up) -> queue 30
15:45:27 INFO real_world.inference_controller: [infer] #62 | start 15:45:26.889 end 15:45:27.445 | took 556.3 ms | carried ids 135..148 | robot@id 135 (queued->141, lead 6) | srv 136 ms | L-gap 0.1cm R-gap 4.0cm
15:45:28 INFO real_world.inference_controller: [infer] #63 | start 15:45:27.459 end 15:45:28.079 | took 620.6 ms | carried ids 138..151 | robot@id 139 (queued->144, lead 5) | srv 139 ms | L-gap 0.2cm R-gap 4.4cm
[pipeline] append: obs_row=141, clock=141, appended ids 145..146 (+0 catch-up) -> queue 27
15:45:28 INFO real_world.inference_controller: [infer] #64 | start 15:45:28.080 end 15:45:28.581 | took 500.8 ms | carried ids 141..154 | robot@id 141 (queued->146, lead 5) | srv 117 ms | L-gap 0.3cm R-gap 4.9cm
15:45:28 INFO real_world.grasp_recovery: [grasp-check] right grip=0.8 (close>=60.0) closed=False armed=False checked=False
15:45:29 INFO real_world.inference_controller: [infer] #65 | start 15:45:28.582 end 15:45:29.109 | took 526.6 ms | carried ids 143..156 | robot@id 143 (queued->149, lead 6) | srv 124 ms | L-gap 0.7cm R-gap 5.5cm
15:45:29 INFO real_world.inference_controller: [auto] STOP requested — draining queue (65 inferences this run).
[pipeline] append: obs_row=146, clock=146, appended ids 150..151 (+0 catch-up) -> queue 27
15:45:29 INFO real_world.inference_controller: [infer] #66 | start 15:45:29.110 end 15:45:29.584 | took 474.0 ms | carried ids 146..159 | robot@id 146 (queued->151, lead 5) | srv 128 ms | L-gap 0.1cm R-gap 5.0cm
[CameraHub] camera unsubscribed (OFF): head
[CameraHub] camera unsubscribed (OFF): hand_left
[CameraHub] camera unsubscribed (OFF): hand_right



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
