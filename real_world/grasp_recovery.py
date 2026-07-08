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
Init glog with processor name:python3.10, pid:639495
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
[release-timing] 1 substeps/s | per-substep avg: total 8.6ms = recorder 0.0 + firmware 0.0 + dispatch 8.6 (STEP_TIME=8.3ms)
[release-timing] 116 substeps/s | per-substep avg: total 8.7ms = recorder 0.0 + firmware 0.0 + dispatch 8.6 (STEP_TIME=8.3ms)
[release-timing] 116 substeps/s | per-substep avg: total 8.6ms = recorder 0.0 + firmware 0.0 + dispatch 8.6 (STEP_TIME=8.3ms)
[release-timing] 116 substeps/s | per-substep avg: total 8.6ms = recorder 0.0 + firmware 0.0 + dispatch 8.6 (STEP_TIME=8.3ms)
[release-timing] 116 substeps/s | per-substep avg: total 8.6ms = recorder 0.0 + firmware 0.0 + dispatch 8.6 (STEP_TIME=8.3ms)
[release-timing] 115 substeps/s | per-substep avg: total 8.7ms = recorder 0.0 + firmware 0.0 + dispatch 8.7 (STEP_TIME=8.3ms)
[release-timing] 116 substeps/s | per-substep avg: total 8.6ms = recorder 0.0 + firmware 0.0 + dispatch 8.6 (STEP_TIME=8.3ms)
[release-timing] 115 substeps/s | per-substep avg: total 8.7ms = recorder 0.0 + firmware 0.0 + dispatch 8.7 (STEP_TIME=8.3ms)
[release-timing] 116 substeps/s | per-substep avg: total 8.7ms = recorder 0.0 + firmware 0.0 + dispatch 8.7 (STEP_TIME=8.3ms)
[release-timing] 116 substeps/s | per-substep avg: total 8.6ms = recorder 0.0 + firmware 0.0 + dispatch 8.6 (STEP_TIME=8.3ms)
[release-timing] 116 substeps/s | per-substep avg: total 8.6ms = recorder 0.0 + firmware 0.0 + dispatch 8.6 (STEP_TIME=8.3ms)
[release-timing] 116 substeps/s | per-substep avg: total 8.6ms = recorder 0.0 + firmware 0.0 + dispatch 8.6 (STEP_TIME=8.3ms)
[release-timing] 115 substeps/s | per-substep avg: total 8.7ms = recorder 0.0 + firmware 0.0 + dispatch 8.7 (STEP_TIME=8.3ms)
[HumanoidEnv] release refused: nothing sim-validated (run 执行 first).
[HumanoidEnv] sim-validated 1381 points (id 2); 2762 substep(s) staged for release.
[HumanoidEnv] released 1381 validated pts (+91 ramp-in) to robot.
[HumanoidEnv] dispatch-ramp: |Δq|=0.632 rad exceeds cap 0.033 -> streaming 19 bounded substeps
[release-timing] 81 substeps/s | per-substep avg: total 10.6ms = recorder 0.0 + firmware 0.0 + dispatch 10.6 (STEP_TIME=8.3ms)
[HumanoidEnv] E-STOP: latched; dropped 4228 pending/staged cmds; holding pose.
[HumanoidEnv] release refused: E-stop latched (press 复位 to reset).
[HumanoidEnv] E-stop reset; release re-enabled (run 执行 then 释放).
[safety] ALL INVARIANTS PASS (C1 C2 C3 C4 C5 H1)
[startup] safety pre-flight passed.

SLAM 模块初始化成功（已解冻关节状态）
[INFO] [1783500772.158754407] [wheel_controller_example]: Wheel Controller Example node started. Publishing a target pose for the robot base.
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
Exception ignored in: <function Node.__del__ at 0x73273f7e3520>
Traceback (most recent call last):
  File "/home/mujin/miniconda3/envs/ros2/lib/python3.10/site-packages/cosine_bus/agibotdds_py3/agibotdds.py", line 225, in __del__
    for publisher in self.list_publisher:
AttributeError: 'Node' object has no attribute 'list_publisher'
HTTP/JSON listening on http://0.0.0.0:9000/robot_info  (Ctrl-C to stop)
16:52:56 INFO real_world.grasp_recovery: GraspRecoveryMonitor ready: settle=5.0s closed_grip_min=50.0 roi=(330, 265, 625, 480) thr=0.50 device=cpu
16:52:56 INFO real_world.postprocess: smoothing set: radius=6 sigma=1.40 m=0.123
16:52:56 INFO real_world.postprocess: te_buffer_len set: 8
[HumanoidEnv] speed_scale=2.794 -> substeps_per_row=4, ramp_joint_step=0.0333
[tuning] restored from /home/mujin/workspaces/humanoid/tuning_config.json
16:52:57 INFO real_world.postprocess: smoothing set: radius=6 sigma=1.40 m=0.123
16:52:57 INFO real_world.postprocess: te_buffer_len set: 8
[HumanoidEnv] speed_scale=2.790 -> substeps_per_row=4, ramp_joint_step=0.0333
[HumanoidEnv]: started (collect=on, exec=on, real=on).
16:52:57 INFO real_world.inference_controller: InferenceController ready (env owned by caller).
16:52:57 INFO xr_examples.pico_vr_server.server: Downstream listening on 0.0.0.0:5555
16:52:57 INFO xr_examples.pico_vr_server.server: Upstream listening on 0.0.0.0:5556
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
16:53:20 INFO real_world.inference_controller: [auto] START -> robot | server 10.12.11.144:9000 | uncapped (latency-bound) | temporal-ensemble ON | grasp-recovery ON
[CameraHub] camera ON: head -> ['head']
[CameraHub] camera ON: hand_left -> ['hand_left', 'head']
[CameraHub] camera ON: hand_right -> ['hand_left', 'hand_right', 'head']
[ObsCollector] warming up: waiting for cameras ['head', 'hand_left', 'hand_right']…
[CameraHub] camera subscribed (ON): head
[CameraHub] camera subscribed (ON): hand_left
[CameraHub] camera subscribed (ON): hand_right
16:53:20 INFO real_world.grasp_recovery: [grasp-check] right grip=118.3 (close>=50.0) closed=True armed=False checked=False
16:53:20 INFO real_world.grasp_recovery: [grasp-check] open->close (grip=118.3) -> attempt ARMED, detector check in 5.0s
[pipeline] append: obs_row=0, clock=0, appended ids 0..3 (+0 catch-up) -> queue 20
16:53:20 INFO real_world.inference_controller: [infer] #1 | start 16:53:20.346 end 16:53:20.665 | took 318.3 ms | carried ids 0..13 | robot@id 0 (queued->3, lead 3) | srv 113 ms | L-gap 2.6cm R-gap 0.9cm
16:53:20 INFO real_world.grasp_recovery: [grasp-check] grasp armed, settling 0.2/5.0s before detector runs
[release-timing] 1 substeps/s | per-substep avg: total 18.9ms = recorder 0.0 + firmware 0.4 + dispatch 18.5 (STEP_TIME=8.3ms)
16:53:20 INFO real_world.inference_controller: [infer] #2 | start 16:53:20.665 end 16:53:20.934 | took 269.3 ms | carried ids 0..13 | robot@id 3 (queued->5, lead 2) | srv 117 ms | L-gap 5.1cm R-gap 2.3cm
16:53:21 INFO real_world.inference_controller: [infer] #3 | start 16:53:20.939 end 16:53:21.240 | took 300.7 ms | carried ids 0..13 | robot@id 5 (queued->8, lead 3) | srv 103 ms | L-gap 3.7cm R-gap 4.1cm
16:53:21 INFO real_world.inference_controller: [infer] #4 | start 16:53:21.241 end 16:53:21.555 | took 314.5 ms | carried ids 3..16 | robot@id 8 (queued->11, lead 3) | srv 119 ms | L-gap 0.2cm R-gap 6.1cm
[release-timing] 48 substeps/s | per-substep avg: total 15.0ms = recorder 0.0 + firmware 1.0 + dispatch 13.9 (STEP_TIME=8.3ms)
[pipeline] append: obs_row=6, clock=11, appended ids 12..14 (+0 catch-up) -> queue 12
16:53:21 INFO real_world.inference_controller: [infer] #5 | start 16:53:21.556 end 16:53:21.866 | took 310.2 ms | carried ids 6..19 | robot@id 11 (queued->14, lead 3) | srv 130 ms | L-gap 0.5cm R-gap 6.3cm
16:53:21 INFO real_world.grasp_recovery: [grasp-check] grasp armed, settling 1.4/5.0s before detector runs
16:53:22 INFO real_world.inference_controller: [infer] #6 | start 16:53:21.868 end 16:53:22.210 | took 342.5 ms | carried ids 9..22 | robot@id 14 (queued->17, lead 3) | srv 122 ms | L-gap 0.3cm R-gap 6.1cm
16:53:22 INFO real_world.inference_controller: [infer] #7 | start 16:53:22.211 end 16:53:22.563 | took 351.7 ms | carried ids 12..25 | robot@id 17 (queued->20, lead 3) | srv 131 ms | L-gap 0.4cm R-gap 6.7cm
[release-timing] 33 substeps/s | per-substep avg: total 17.9ms = recorder 0.0 + firmware 1.6 + dispatch 16.3 (STEP_TIME=8.3ms)
16:53:22 INFO real_world.grasp_recovery: [grasp-check] right grip=119.8 (close>=50.0) closed=True armed=True checked=False
[pipeline] append: obs_row=15, clock=20, appended ids 21..23 (+0 catch-up) -> queue 12
16:53:22 INFO real_world.inference_controller: [infer] #8 | start 16:53:22.567 end 16:53:22.941 | took 374.1 ms | carried ids 15..28 | robot@id 20 (queued->23, lead 3) | srv 114 ms | L-gap 0.2cm R-gap 7.7cm
16:53:22 INFO real_world.grasp_recovery: [grasp-check] grasp armed, settling 2.4/5.0s before detector runs
16:53:23 INFO real_world.inference_controller: [infer] #9 | start 16:53:22.941 end 16:53:23.268 | took 326.8 ms | carried ids 18..31 | robot@id 23 (queued->26, lead 3) | srv 100 ms | L-gap 0.5cm R-gap 8.3cm
16:53:23 INFO real_world.inference_controller: [infer] #10 | start 16:53:23.271 end 16:53:23.598 | took 327.7 ms | carried ids 21..34 | robot@id 26 (queued->29, lead 3) | srv 97 ms | L-gap 0.3cm R-gap 7.8cm
[release-timing] 36 substeps/s | per-substep avg: total 17.7ms = recorder 0.0 + firmware 1.6 + dispatch 16.0 (STEP_TIME=8.3ms)
[pipeline] append: obs_row=24, clock=29, appended ids 30..32 (+0 catch-up) -> queue 12
16:53:23 INFO real_world.inference_controller: [infer] #11 | start 16:53:23.605 end 16:53:23.975 | took 369.7 ms | carried ids 24..37 | robot@id 29 (queued->32, lead 3) | srv 129 ms | L-gap 0.3cm R-gap 7.9cm
16:53:23 INFO real_world.grasp_recovery: [grasp-check] grasp armed, settling 3.5/5.0s before detector runs
16:53:24 INFO real_world.inference_controller: [infer] #12 | start 16:53:23.976 end 16:53:24.345 | took 368.7 ms | carried ids 27..40 | robot@id 32 (queued->35, lead 3) | srv 142 ms | L-gap 0.3cm R-gap 8.0cm
16:53:24 INFO real_world.inference_controller: [infer] #13 | start 16:53:24.350 end 16:53:24.686 | took 336.7 ms | carried ids 30..43 | robot@id 35 (queued->38, lead 3) | srv 121 ms | L-gap 0.2cm R-gap 7.2cm
[release-timing] 32 substeps/s | per-substep avg: total 18.0ms = recorder 0.0 + firmware 1.1 + dispatch 16.9 (STEP_TIME=8.3ms)
16:53:24 INFO real_world.grasp_recovery: [grasp-check] right grip=118.2 (close>=50.0) closed=True armed=True checked=False
[pipeline] append: obs_row=33, clock=38, appended ids 39..41 (+0 catch-up) -> queue 12
16:53:25 INFO real_world.inference_controller: [infer] #14 | start 16:53:24.687 end 16:53:25.082 | took 395.0 ms | carried ids 33..46 | robot@id 38 (queued->41, lead 3) | srv 126 ms | L-gap 0.2cm R-gap 6.8cm
16:53:25 INFO real_world.grasp_recovery: [grasp-check] grasp armed, settling 4.6/5.0s before detector runs
16:53:25 INFO real_world.inference_controller: [infer] #15 | start 16:53:25.088 end 16:53:25.412 | took 324.1 ms | carried ids 36..49 | robot@id 41 (queued->44, lead 3) | srv 106 ms | L-gap 0.3cm R-gap 6.9cm
16:53:25 INFO real_world.inference_controller: [infer] #16 | start 16:53:25.413 end 16:53:25.783 | took 370.3 ms | carried ids 39..52 | robot@id 44 (queued->47, lead 3) | srv 124 ms | L-gap 0.2cm R-gap 7.3cm
16:53:25 INFO real_world.grasp_recovery: [grasp-check] settle elapsed -> running detector on right wrist frame
[release-timing] 35 substeps/s | per-substep avg: total 18.1ms = recorder 0.0 + firmware 2.8 + dispatch 15.3 (STEP_TIME=8.3ms)
16:53:26 INFO real_world.grasp_recovery: [grasp-check] detector P_empty=0.02 < thr=0.50 -> grasp HELD, no recovery
[pipeline] append: obs_row=42, clock=47, appended ids 48..50 (+0 catch-up) -> queue 12
16:53:26 INFO real_world.inference_controller: [infer] #17 | start 16:53:25.785 end 16:53:26.745 | took 959.8 ms | carried ids 42..55 | robot@id 47 (queued->50, lead 3) | srv 116 ms | L-gap 0.3cm R-gap 7.6cm
[release-timing] 15 substeps/s | per-substep avg: total 24.4ms = recorder 0.0 + firmware 1.7 + dispatch 22.7 (STEP_TIME=8.3ms)
16:53:27 INFO real_world.grasp_recovery: [grasp-check] right grip=117.2 (close>=50.0) closed=True armed=True checked=True
16:53:27 INFO real_world.inference_controller: [infer] #18 | start 16:53:26.752 end 16:53:27.139 | took 387.3 ms | carried ids 45..58 | robot@id 50 (queued->53, lead 3) | srv 115 ms | L-gap 0.2cm R-gap 7.3cm
16:53:27 INFO real_world.inference_controller: [infer] #19 | start 16:53:27.139 end 16:53:27.503 | took 363.3 ms | carried ids 48..61 | robot@id 53 (queued->56, lead 3) | srv 106 ms | L-gap 0.2cm R-gap 6.8cm
[pipeline] append: obs_row=51, clock=56, appended ids 57..59 (+0 catch-up) -> queue 12
16:53:27 INFO real_world.inference_controller: [infer] #20 | start 16:53:27.511 end 16:53:27.907 | took 396.7 ms | carried ids 51..64 | robot@id 56 (queued->59, lead 3) | srv 109 ms | L-gap 0.3cm R-gap 7.1cm
[release-timing] 33 substeps/s | per-substep avg: total 19.0ms = recorder 0.0 + firmware 2.2 + dispatch 16.8 (STEP_TIME=8.3ms)
16:53:28 INFO real_world.inference_controller: [infer] #21 | start 16:53:27.908 end 16:53:28.272 | took 363.7 ms | carried ids 54..67 | robot@id 59 (queued->62, lead 3) | srv 124 ms | L-gap 0.3cm R-gap 7.4cm
16:53:28 INFO real_world.inference_controller: [infer] #22 | start 16:53:28.274 end 16:53:28.671 | took 396.2 ms | carried ids 57..70 | robot@id 62 (queued->65, lead 3) | srv 107 ms | L-gap 0.2cm R-gap 7.4cm
[pipeline] append: obs_row=60, clock=65, appended ids 66..68 (+0 catch-up) -> queue 12
16:53:29 INFO real_world.inference_controller: [infer] #23 | start 16:53:28.673 end 16:53:29.045 | took 372.0 ms | carried ids 60..73 | robot@id 65 (queued->68, lead 3) | srv 109 ms | L-gap 0.2cm R-gap 7.2cm
[release-timing] 36 substeps/s | per-substep avg: total 20.2ms = recorder 0.0 + firmware 3.2 + dispatch 17.1 (STEP_TIME=8.3ms)
16:53:29 INFO real_world.grasp_recovery: [grasp-check] right grip=119.0 (close>=50.0) closed=True armed=True checked=True
16:53:29 INFO real_world.inference_controller: [infer] #24 | start 16:53:29.046 end 16:53:29.434 | took 388.7 ms | carried ids 63..76 | robot@id 68 (queued->71, lead 3) | srv 93 ms | L-gap 0.1cm R-gap 7.5cm
16:53:29 INFO real_world.inference_controller: [infer] #25 | start 16:53:29.441 end 16:53:29.889 | took 447.6 ms | carried ids 66..79 | robot@id 71 (queued->74, lead 3) | srv 130 ms | L-gap 0.2cm R-gap 7.6cm
[release-timing] 33 substeps/s | per-substep avg: total 20.1ms = recorder 0.0 + firmware 2.6 + dispatch 17.5 (STEP_TIME=8.3ms)
[pipeline] append: obs_row=69, clock=74, appended ids 75..77 (+0 catch-up) -> queue 12
16:53:30 INFO real_world.inference_controller: [infer] #26 | start 16:53:29.890 end 16:53:30.264 | took 373.5 ms | carried ids 69..82 | robot@id 74 (queued->77, lead 3) | srv 111 ms | L-gap 0.0cm R-gap 8.0cm
16:53:30 INFO real_world.inference_controller: [infer] #27 | start 16:53:30.265 end 16:53:30.625 | took 359.3 ms | carried ids 72..85 | robot@id 77 (queued->80, lead 3) | srv 116 ms | L-gap 0.2cm R-gap 8.2cm
16:53:31 INFO real_world.inference_controller: [infer] #28 | start 16:53:30.625 end 16:53:31.049 | took 424.0 ms | carried ids 75..88 | robot@id 80 (queued->83, lead 3) | srv 138 ms | L-gap 0.2cm R-gap 7.8cm
[release-timing] 27 substeps/s | per-substep avg: total 19.3ms = recorder 0.0 + firmware 1.3 + dispatch 18.0 (STEP_TIME=8.3ms)
[pipeline] append: obs_row=78, clock=83, appended ids 84..86 (+0 catch-up) -> queue 12
16:53:31 INFO real_world.inference_controller: [infer] #29 | start 16:53:31.049 end 16:53:31.397 | took 347.2 ms | carried ids 78..91 | robot@id 83 (queued->86, lead 3) | srv 106 ms | L-gap 0.1cm R-gap 7.5cm
16:53:31 INFO real_world.grasp_recovery: [grasp-check] right grip=119.8 (close>=50.0) closed=True armed=True checked=True
16:53:31 INFO real_world.inference_controller: [infer] #30 | start 16:53:31.397 end 16:53:31.816 | took 419.1 ms | carried ids 81..94 | robot@id 86 (queued->89, lead 3) | srv 112 ms | L-gap 0.1cm R-gap 7.9cm
16:53:32 INFO real_world.inference_controller: [infer] #31 | start 16:53:31.817 end 16:53:32.216 | took 398.8 ms | carried ids 84..97 | robot@id 89 (queued->92, lead 3) | srv 112 ms | L-gap 0.1cm R-gap 7.3cm
[release-timing] 36 substeps/s | per-substep avg: total 18.6ms = recorder 0.0 + firmware 2.2 + dispatch 16.4 (STEP_TIME=8.3ms)
[pipeline] append: obs_row=87, clock=92, appended ids 93..95 (+0 catch-up) -> queue 12
16:53:32 INFO real_world.inference_controller: [infer] #32 | start 16:53:32.218 end 16:53:32.597 | took 378.7 ms | carried ids 87..100 | robot@id 93 (queued->95, lead 2) | srv 124 ms | L-gap 0.1cm R-gap 6.6cm
16:53:32 INFO real_world.inference_controller: [infer] #33 | start 16:53:32.599 end 16:53:32.974 | took 375.1 ms | carried ids 90..103 | robot@id 95 (queued->98, lead 3) | srv 116 ms | L-gap 0.1cm R-gap 6.7cm
16:53:33 INFO real_world.inference_controller: [infer] #34 | start 16:53:32.976 end 16:53:33.389 | took 412.2 ms | carried ids 93..106 | robot@id 98 (queued->101, lead 3) | srv 115 ms | L-gap 0.0cm R-gap 7.0cm
[release-timing] 36 substeps/s | per-substep avg: total 19.6ms = recorder 0.0 + firmware 2.8 + dispatch 16.7 (STEP_TIME=8.3ms)
[pipeline] append: obs_row=96, clock=101, appended ids 102..104 (+0 catch-up) -> queue 12
16:53:33 INFO real_world.inference_controller: [infer] #35 | start 16:53:33.390 end 16:53:33.757 | took 367.4 ms | carried ids 96..109 | robot@id 101 (queued->104, lead 3) | srv 97 ms | L-gap 0.2cm R-gap 7.5cm
16:53:33 INFO real_world.grasp_recovery: [grasp-check] right grip=118.0 (close>=50.0) closed=True armed=True checked=True
16:53:34 INFO real_world.inference_controller: [infer] #36 | start 16:53:33.759 end 16:53:34.110 | took 351.1 ms | carried ids 99..112 | robot@id 104 (queued->107, lead 3) | srv 116 ms | L-gap 0.1cm R-gap 7.9cm
16:53:34 INFO real_world.inference_controller: [infer] #37 | start 16:53:34.116 end 16:53:34.496 | took 379.9 ms | carried ids 102..115 | robot@id 107 (queued->110, lead 3) | srv 108 ms | L-gap 0.1cm R-gap 7.4cm
[release-timing] 36 substeps/s | per-substep avg: total 17.9ms = recorder 0.0 + firmware 1.7 + dispatch 16.3 (STEP_TIME=8.3ms)
[pipeline] append: obs_row=105, clock=110, appended ids 111..113 (+0 catch-up) -> queue 12
16:53:34 INFO real_world.inference_controller: [infer] #38 | start 16:53:34.500 end 16:53:34.877 | took 376.8 ms | carried ids 105..118 | robot@id 110 (queued->113, lead 3) | srv 113 ms | L-gap 0.1cm R-gap 7.2cm
16:53:35 INFO real_world.inference_controller: [infer] #39 | start 16:53:34.877 end 16:53:35.264 | took 386.4 ms | carried ids 108..121 | robot@id 113 (queued->116, lead 3) | srv 101 ms | L-gap 0.3cm R-gap 7.6cm
16:53:35 INFO real_world.inference_controller: [infer] #40 | start 16:53:35.264 end 16:53:35.674 | took 409.2 ms | carried ids 111..124 | robot@id 116 (queued->119, lead 3) | srv 117 ms | L-gap 0.1cm R-gap 7.9cm
[release-timing] 36 substeps/s | per-substep avg: total 18.6ms = recorder 0.0 + firmware 1.4 + dispatch 17.2 (STEP_TIME=8.3ms)
[pipeline] append: obs_row=114, clock=119, appended ids 120..122 (+0 catch-up) -> queue 12
16:53:36 INFO real_world.inference_controller: [infer] #41 | start 16:53:35.679 end 16:53:36.046 | took 366.4 ms | carried ids 114..127 | robot@id 119 (queued->122, lead 3) | srv 127 ms | L-gap 0.1cm R-gap 7.6cm
16:53:36 INFO real_world.grasp_recovery: [grasp-check] right grip=119.8 (close>=50.0) closed=True armed=True checked=True
16:53:36 INFO real_world.inference_controller: [infer] #42 | start 16:53:36.048 end 16:53:36.436 | took 387.9 ms | carried ids 117..130 | robot@id 122 (queued->125, lead 3) | srv 111 ms | L-gap 0.4cm R-gap 7.6cm
16:53:36 INFO real_world.inference_controller: [infer] #43 | start 16:53:36.437 end 16:53:36.827 | took 390.5 ms | carried ids 120..133 | robot@id 125 (queued->128, lead 3) | srv 153 ms | L-gap 0.1cm R-gap 7.5cm
[release-timing] 36 substeps/s | per-substep avg: total 21.0ms = recorder 0.0 + firmware 1.9 + dispatch 19.1 (STEP_TIME=8.3ms)
[pipeline] append: obs_row=123, clock=128, appended ids 129..131 (+0 catch-up) -> queue 12
16:53:37 INFO real_world.inference_controller: [infer] #44 | start 16:53:36.829 end 16:53:37.231 | took 402.0 ms | carried ids 123..136 | robot@id 128 (queued->131, lead 3) | srv 103 ms | L-gap 0.0cm R-gap 7.6cm
16:53:37 INFO real_world.inference_controller: [infer] #45 | start 16:53:37.233 end 16:53:37.634 | took 401.1 ms | carried ids 126..139 | robot@id 131 (queued->134, lead 3) | srv 112 ms | L-gap 0.1cm R-gap 7.9cm
[release-timing] 34 substeps/s | per-substep avg: total 19.4ms = recorder 0.0 + firmware 3.2 + dispatch 16.2 (STEP_TIME=8.3ms)
16:53:38 INFO real_world.inference_controller: [infer] #46 | start 16:53:37.634 end 16:53:38.051 | took 416.6 ms | carried ids 129..142 | robot@id 134 (queued->137, lead 3) | srv 117 ms | L-gap 0.2cm R-gap 8.1cm
16:53:38 INFO real_world.grasp_recovery: [grasp-check] right grip=117.9 (close>=50.0) closed=True armed=True checked=True
[pipeline] append: obs_row=132, clock=137, appended ids 138..140 (+0 catch-up) -> queue 12
16:53:38 INFO real_world.inference_controller: [infer] #47 | start 16:53:38.051 end 16:53:38.486 | took 434.9 ms | carried ids 132..145 | robot@id 137 (queued->140, lead 3) | srv 131 ms | L-gap 0.2cm R-gap 7.3cm
16:53:38 INFO real_world.inference_controller: [infer] #48 | start 16:53:38.490 end 16:53:38.865 | took 374.4 ms | carried ids 135..148 | robot@id 140 (queued->143, lead 3) | srv 110 ms | L-gap 0.2cm R-gap 7.2cm
[release-timing] 26 substeps/s | per-substep avg: total 20.9ms = recorder 0.0 + firmware 2.2 + dispatch 18.7 (STEP_TIME=8.3ms)
16:53:39 INFO real_world.inference_controller: [infer] #49 | start 16:53:38.866 end 16:53:39.264 | took 398.4 ms | carried ids 138..151 | robot@id 143 (queued->146, lead 3) | srv 123 ms | L-gap 0.3cm R-gap 7.2cm
[pipeline] append: obs_row=141, clock=146, appended ids 147..149 (+0 catch-up) -> queue 12
16:53:39 INFO real_world.inference_controller: [infer] #50 | start 16:53:39.266 end 16:53:39.671 | took 404.8 ms | carried ids 141..154 | robot@id 146 (queued->149, lead 3) | srv 128 ms | L-gap 0.1cm R-gap 7.5cm
16:53:40 INFO real_world.inference_controller: [infer] #51 | start 16:53:39.675 end 16:53:40.091 | took 415.2 ms | carried ids 144..157 | robot@id 149 (queued->152, lead 3) | srv 144 ms | L-gap 0.2cm R-gap 7.1cm
[release-timing] 36 substeps/s | per-substep avg: total 19.3ms = recorder 0.0 + firmware 1.9 + dispatch 17.3 (STEP_TIME=8.3ms)
16:53:40 INFO real_world.grasp_recovery: [grasp-check] right grip=118.9 (close>=50.0) closed=True armed=True checked=True
16:53:40 INFO real_world.inference_controller: [infer] #52 | start 16:53:40.099 end 16:53:40.474 | took 374.4 ms | carried ids 147..160 | robot@id 152 (queued->155, lead 3) | srv 118 ms | L-gap 0.2cm R-gap 6.9cm
16:53:40 INFO real_world.inference_controller: [auto] STOP requested — draining queue (52 inferences this run).
[pipeline] append: obs_row=150, clock=155, appended ids 156..158 (+0 catch-up) -> queue 12
16:53:40 INFO real_world.inference_controller: [infer] #53 | start 16:53:40.479 end 16:53:40.909 | took 429.4 ms | carried ids 150..163 | robot@id 155 (queued->158, lead 3) | srv 125 ms | L-gap 0.1cm R-gap 7.1cm
[release-timing] 35 substeps/s | per-substep avg: total 19.3ms = recorder 0.0 + firmware 2.1 + dispatch 17.2 (STEP_TIME=8.3ms)
16:53:42 INFO real_world.grasp_recovery: [grasp-check] open->close (grip=115.8) -> attempt ARMED, detector check in 5.0s
[HumanoidEnv] sim-validated 53 points (id 1); 53 substep(s) staged for release.
[HumanoidEnv] queued 60 cmd(s) to robot (0 substep(s) still staged).
[release-timing] 1 substeps/s | per-substep avg: total 14.8ms = recorder 0.0 + firmware 0.4 + dispatch 14.4 (STEP_TIME=8.3ms)
[CameraHub] camera unsubscribed (OFF): head
[CameraHub] camera unsubscribed (OFF): hand_left
[CameraHub] camera unsubscribed (OFF): hand_right
[CameraHub] camera ON: head -> ['head']
[CameraHub] camera ON: hand_left -> ['hand_left', 'head']
[CameraHub] camera ON: hand_right -> ['hand_left', 'hand_right', 'head']
[ObsCollector] warming up: waiting for cameras ['head', 'hand_left', 'hand_right']…
[CameraHub] camera subscribed (ON): head
[CameraHub] camera subscribed (ON): hand_left
[CameraHub] camera subscribed (ON): hand_right
16:53:48 INFO real_world.grasp_recovery: [grasp-check] settle elapsed -> running detector on right wrist frame
16:53:49 INFO real_world.grasp_recovery: [grasp-check] detector P_empty=0.03 < thr=0.50 -> grasp HELD, no recovery
16:53:49 INFO real_world.grasp_recovery: [grasp-check] right grip=118.0 (close>=50.0) closed=True armed=True checked=True
[HumanoidEnv] sim-validated 53 points (id 2); 53 substep(s) staged for release.
[HumanoidEnv] queued 64 cmd(s) to robot (0 substep(s) still staged).
[release-timing] 60 substeps/s | per-substep avg: total 14.9ms = recorder 0.0 + firmware 1.3 + dispatch 13.6 (STEP_TIME=8.3ms)
[CameraHub] camera unsubscribed (OFF): head
[CameraHub] camera unsubscribed (OFF): hand_left
[CameraHub] camera unsubscribed (OFF): hand_right
[CameraHub] camera ON: head -> ['head']
[CameraHub] camera ON: hand_left -> ['hand_left', 'head']
[CameraHub] camera ON: hand_right -> ['hand_left', 'hand_right', 'head']
[ObsCollector] warming up: waiting for cameras ['head', 'hand_left', 'hand_right']…
[CameraHub] camera subscribed (ON): head
[CameraHub] camera subscribed (ON): hand_left
[CameraHub] camera subscribed (ON): hand_right
16:54:01 INFO real_world.grasp_recovery: [grasp-check] right grip=117.5 (close>=50.0) closed=True armed=True checked=True
[HumanoidEnv] sim-validated 53 points (id 3); 53 substep(s) staged for release.
[HumanoidEnv] queued 62 cmd(s) to robot (0 substep(s) still staged).
[release-timing] 64 substeps/s | per-substep avg: total 13.9ms = recorder 0.0 + firmware 0.8 + dispatch 13.1 (STEP_TIME=8.3ms)
[CameraHub] camera unsubscribed (OFF): head
[CameraHub] camera unsubscribed (OFF): hand_left
[CameraHub] camera unsubscribed (OFF): hand_right
16:54:10 INFO real_world.inference_controller: [auto] START -> robot | server 10.12.11.144:9000 | uncapped (latency-bound) | temporal-ensemble ON | grasp-recovery ON
[CameraHub] camera ON: head -> ['head']
[CameraHub] camera ON: hand_left -> ['hand_left', 'head']
[CameraHub] camera ON: hand_right -> ['hand_left', 'hand_right', 'head']
[ObsCollector] warming up: waiting for cameras ['head', 'hand_left', 'hand_right']…
[CameraHub] camera subscribed (ON): head
[CameraHub] camera subscribed (ON): hand_left
[CameraHub] camera subscribed (ON): hand_right
16:54:10 INFO real_world.grasp_recovery: [grasp-check] right grip=115.8 (close>=50.0) closed=True armed=True checked=True
[pipeline] append: obs_row=153, clock=158, appended ids 159..161 (+0 catch-up) -> queue 16
16:54:11 INFO real_world.inference_controller: [infer] #57 | start 16:54:10.748 end 16:54:11.085 | took 336.7 ms | carried ids 153..166 | robot@id 158 (queued->161, lead 3) | srv 114 ms | L-gap 0.3cm R-gap 6.8cm
[release-timing] 62 substeps/s | per-substep avg: total 13.1ms = recorder 0.0 + firmware 0.6 + dispatch 12.5 (STEP_TIME=8.3ms)
16:54:11 INFO real_world.inference_controller: [infer] #58 | start 16:54:11.086 end 16:54:11.486 | took 399.9 ms | carried ids 156..169 | robot@id 161 (queued->164, lead 3) | srv 133 ms | L-gap 0.4cm R-gap 6.1cm
16:54:11 INFO real_world.inference_controller: [infer] #59 | start 16:54:11.496 end 16:54:11.863 | took 367.3 ms | carried ids 159..172 | robot@id 164 (queued->167, lead 3) | srv 113 ms | L-gap 0.2cm R-gap 6.9cm
[release-timing] 37 substeps/s | per-substep avg: total 20.2ms = recorder 0.0 + firmware 1.8 + dispatch 18.4 (STEP_TIME=8.3ms)
[pipeline] append: obs_row=162, clock=167, appended ids 168..170 (+0 catch-up) -> queue 12
16:54:12 INFO real_world.inference_controller: [infer] #60 | start 16:54:11.877 end 16:54:12.314 | took 437.2 ms | carried ids 162..175 | robot@id 168 (queued->170, lead 2) | srv 127 ms | L-gap 0.1cm R-gap 7.3cm
16:54:12 INFO real_world.inference_controller: [infer] #61 | start 16:54:12.318 end 16:54:12.731 | took 413.5 ms | carried ids 165..178 | robot@id 170 (queued->173, lead 3) | srv 150 ms | L-gap 0.1cm R-gap 7.5cm
16:54:13 INFO real_world.grasp_recovery: [grasp-check] right grip=119.2 (close>=50.0) closed=True armed=True checked=True
16:54:13 INFO real_world.inference_controller: [infer] #62 | start 16:54:12.739 end 16:54:13.123 | took 383.6 ms | carried ids 168..181 | robot@id 173 (queued->176, lead 3) | srv 128 ms | L-gap 0.1cm R-gap 7.2cm
[release-timing] 27 substeps/s | per-substep avg: total 19.3ms = recorder 0.0 + firmware 2.0 + dispatch 17.3 (STEP_TIME=8.3ms)
[pipeline] append: obs_row=171, clock=176, appended ids 177..179 (+0 catch-up) -> queue 12
16:54:13 INFO real_world.inference_controller: [infer] #63 | start 16:54:13.124 end 16:54:13.530 | took 406.3 ms | carried ids 171..184 | robot@id 176 (queued->179, lead 3) | srv 114 ms | L-gap 0.1cm R-gap 7.3cm
16:54:13 INFO real_world.inference_controller: [infer] #64 | start 16:54:13.535 end 16:54:13.927 | took 391.1 ms | carried ids 174..187 | robot@id 179 (queued->182, lead 3) | srv 96 ms | L-gap 0.2cm R-gap 7.4cm
[release-timing] 32 substeps/s | per-substep avg: total 24.2ms = recorder 0.0 + firmware 4.9 + dispatch 19.4 (STEP_TIME=8.3ms)
16:54:14 INFO real_world.inference_controller: [infer] #65 | start 16:54:13.939 end 16:54:14.342 | took 403.1 ms | carried ids 177..190 | robot@id 182 (queued->185, lead 3) | srv 128 ms | L-gap 0.1cm R-gap 7.3cm
[pipeline] append: obs_row=180, clock=185, appended ids 186..188 (+0 catch-up) -> queue 12
16:54:14 INFO real_world.inference_controller: [infer] #66 | start 16:54:14.347 end 16:54:14.761 | took 414.4 ms | carried ids 180..193 | robot@id 185 (queued->188, lead 3) | srv 123 ms | L-gap 0.1cm R-gap 7.6cm
16:54:15 INFO real_world.grasp_recovery: [grasp-check] right grip=117.9 (close>=50.0) closed=True armed=True checked=True
16:54:15 INFO real_world.inference_controller: [infer] #67 | start 16:54:14.763 end 16:54:15.167 | took 404.1 ms | carried ids 183..196 | robot@id 188 (queued->191, lead 3) | srv 119 ms | L-gap 0.2cm R-gap 7.2cm
[release-timing] 28 substeps/s | per-substep avg: total 20.1ms = recorder 0.0 + firmware 2.5 + dispatch 17.6 (STEP_TIME=8.3ms)
16:54:15 INFO real_world.inference_controller: [auto] STOP requested — draining queue (67 inferences this run).
16:54:15 INFO real_world.inference_controller: [infer] #68 | start 16:54:15.168 end 16:54:15.583 | took 415.5 ms | carried ids 186..199 | robot@id 191 (queued->194, lead 3) | srv 134 ms | L-gap 0.1cm R-gap 7.4cm
16:54:18 INFO real_world.inference_controller: [auto] START -> robot | server 10.12.11.144:9000 | uncapped (latency-bound) | temporal-ensemble ON | grasp-recovery ON
16:54:18 INFO real_world.grasp_recovery: [grasp-check] right grip=117.7 (close>=50.0) closed=True armed=False checked=True
16:54:18 INFO real_world.grasp_recovery: [grasp-check] open->close (grip=117.7) -> attempt ARMED, detector check in 5.0s
[pipeline] append: obs_row=189, clock=194, appended ids 195..197 (+0 catch-up) -> queue 12
16:54:18 INFO real_world.inference_controller: [infer] #69 | start 16:54:18.234 end 16:54:18.571 | took 336.9 ms | carried ids 189..202 | robot@id 194 (queued->197, lead 3) | srv 132 ms | L-gap 0.0cm R-gap 7.6cm
16:54:18 INFO real_world.grasp_recovery: [grasp-check] grasp armed, settling 0.1/5.0s before detector runs
[release-timing] 24 substeps/s | per-substep avg: total 18.4ms = recorder 0.0 + firmware 1.4 + dispatch 17.0 (STEP_TIME=8.3ms)
16:54:19 INFO real_world.inference_controller: [infer] #70 | start 16:54:18.575 end 16:54:19.000 | took 425.1 ms | carried ids 192..205 | robot@id 197 (queued->200, lead 3) | srv 129 ms | L-gap 0.2cm R-gap 6.8cm
16:54:19 INFO real_world.inference_controller: [infer] #71 | start 16:54:19.002 end 16:54:19.372 | took 369.9 ms | carried ids 195..208 | robot@id 200 (queued->203, lead 3) | srv 129 ms | L-gap 0.4cm R-gap 6.5cm
[pipeline] append: obs_row=198, clock=203, appended ids 204..206 (+0 catch-up) -> queue 12
16:54:19 INFO real_world.inference_controller: [infer] #72 | start 16:54:19.384 end 16:54:19.764 | took 380.3 ms | carried ids 198..211 | robot@id 203 (queued->206, lead 3) | srv 104 ms | L-gap 0.4cm R-gap 6.2cm
[release-timing] 36 substeps/s | per-substep avg: total 19.3ms = recorder 0.0 + firmware 2.5 + dispatch 16.7 (STEP_TIME=8.3ms)
16:54:19 INFO real_world.grasp_recovery: [grasp-check] grasp armed, settling 1.3/5.0s before detector runs
16:54:20 INFO real_world.inference_controller: [infer] #73 | start 16:54:19.775 end 16:54:20.198 | took 422.9 ms | carried ids 201..214 | robot@id 206 (queued->209, lead 3) | srv 127 ms | L-gap 0.2cm R-gap 6.1cm
16:54:20 INFO real_world.grasp_recovery: [grasp-check] right grip=117.5 (close>=50.0) closed=True armed=True checked=False
16:54:20 INFO real_world.inference_controller: [infer] #74 | start 16:54:20.200 end 16:54:20.614 | took 413.8 ms | carried ids 204..217 | robot@id 209 (queued->212, lead 3) | srv 125 ms | L-gap 0.1cm R-gap 6.6cm
[release-timing] 31 substeps/s | per-substep avg: total 22.5ms = recorder 0.0 + firmware 1.9 + dispatch 20.7 (STEP_TIME=8.3ms)
[pipeline] append: obs_row=207, clock=212, appended ids 213..215 (+0 catch-up) -> queue 12
16:54:21 INFO real_world.inference_controller: [infer] #75 | start 16:54:20.615 end 16:54:21.013 | took 397.8 ms | carried ids 207..220 | robot@id 212 (queued->215, lead 3) | srv 110 ms | L-gap 0.6cm R-gap 7.1cm
16:54:21 INFO real_world.grasp_recovery: [grasp-check] grasp armed, settling 2.6/5.0s before detector runs
[HumanoidEnv] arm_joint_states bad read (shape (0,)); using last good.
16:54:21 INFO real_world.inference_controller: [infer] #76 | start 16:54:21.017 end 16:54:21.421 | took 404.3 ms | carried ids 210..223 | robot@id 215 (queued->218, lead 3) | srv 95 ms | L-gap 0.3cm R-gap 7.1cm
16:54:21 INFO real_world.inference_controller: [infer] #77 | start 16:54:21.422 end 16:54:21.805 | took 383.8 ms | carried ids 213..226 | robot@id 218 (queued->221, lead 3) | srv 105 ms | L-gap 0.1cm R-gap 7.1cm
[release-timing] 29 substeps/s | per-substep avg: total 20.7ms = recorder 0.0 + firmware 2.4 + dispatch 18.2 (STEP_TIME=8.3ms)
[pipeline] append: obs_row=216, clock=221, appended ids 222..224 (+0 catch-up) -> queue 12
16:54:22 INFO real_world.inference_controller: [infer] #78 | start 16:54:21.811 end 16:54:22.168 | took 356.7 ms | carried ids 216..229 | robot@id 222 (queued->224, lead 2) | srv 107 ms | L-gap 0.3cm R-gap 6.8cm
16:54:22 INFO real_world.grasp_recovery: [grasp-check] grasp armed, settling 3.7/5.0s before detector runs
16:54:22 INFO real_world.grasp_recovery: [grasp-check] right grip=117.5 (close>=50.0) closed=True armed=True checked=False
16:54:22 INFO real_world.inference_controller: [infer] #79 | start 16:54:22.170 end 16:54:22.603 | took 433.7 ms | carried ids 219..232 | robot@id 224 (queued->227, lead 3) | srv 119 ms | L-gap 0.1cm R-gap 6.9cm
[release-timing] 33 substeps/s | per-substep avg: total 20.6ms = recorder 0.0 + firmware 1.7 + dispatch 18.9 (STEP_TIME=8.3ms)
16:54:23 INFO real_world.inference_controller: [infer] #80 | start 16:54:22.605 end 16:54:23.014 | took 409.4 ms | carried ids 222..235 | robot@id 227 (queued->230, lead 3) | srv 126 ms | L-gap 0.1cm R-gap 7.0cm
[pipeline] append: obs_row=225, clock=230, appended ids 231..233 (+0 catch-up) -> queue 12
16:54:23 INFO real_world.inference_controller: [infer] #81 | start 16:54:23.019 end 16:54:23.417 | took 398.1 ms | carried ids 225..238 | robot@id 230 (queued->233, lead 3) | srv 102 ms | L-gap 0.3cm R-gap 6.5cm
16:54:23 INFO real_world.grasp_recovery: [grasp-check] grasp armed, settling 5.0/5.0s before detector runs
16:54:23 INFO real_world.inference_controller: [infer] #82 | start 16:54:23.421 end 16:54:23.825 | took 404.2 ms | carried ids 228..241 | robot@id 233 (queued->236, lead 3) | srv 111 ms | L-gap 0.3cm R-gap 6.6cm
[release-timing] 27 substeps/s | per-substep avg: total 23.3ms = recorder 0.0 + firmware 3.5 + dispatch 19.8 (STEP_TIME=8.3ms)
16:54:23 INFO real_world.grasp_recovery: [grasp-check] settle elapsed -> running detector on right wrist frame
16:54:24 INFO real_world.grasp_recovery: [grasp-check] detector P_empty=0.11 < thr=0.50 -> grasp HELD, no recovery
16:54:24 INFO real_world.grasp_recovery: [grasp-check] right grip=118.9 (close>=50.0) closed=True armed=True checked=True
[pipeline] append: obs_row=231, clock=236, appended ids 237..239 (+0 catch-up) -> queue 12
16:54:24 INFO real_world.inference_controller: [infer] #83 | start 16:54:23.827 end 16:54:24.794 | took 966.9 ms | carried ids 231..244 | robot@id 236 (queued->239, lead 3) | srv 106 ms | L-gap 0.1cm R-gap 6.6cm
[release-timing] 13 substeps/s | per-substep avg: total 24.6ms = recorder 0.0 + firmware 2.7 + dispatch 21.9 (STEP_TIME=8.3ms)
16:54:25 INFO real_world.inference_controller: [infer] #84 | start 16:54:24.795 end 16:54:25.146 | took 351.6 ms | carried ids 234..247 | robot@id 239 (queued->242, lead 3) | srv 118 ms | L-gap 0.1cm R-gap 7.2cm
16:54:25 INFO real_world.inference_controller: [infer] #85 | start 16:54:25.154 end 16:54:25.540 | took 386.0 ms | carried ids 237..250 | robot@id 242 (queued->245, lead 3) | srv 108 ms | L-gap 0.1cm R-gap 7.5cm
[pipeline] append: obs_row=240, clock=245, appended ids 246..248 (+0 catch-up) -> queue 12
16:54:25 INFO real_world.inference_controller: [infer] #86 | start 16:54:25.542 end 16:54:25.932 | took 389.8 ms | carried ids 240..253 | robot@id 245 (queued->248, lead 3) | srv 113 ms | L-gap 0.1cm R-gap 7.0cm
[release-timing] 35 substeps/s | per-substep avg: total 18.9ms = recorder 0.0 + firmware 1.7 + dispatch 17.2 (STEP_TIME=8.3ms)
16:54:26 INFO real_world.inference_controller: [infer] #87 | start 16:54:25.933 end 16:54:26.356 | took 423.3 ms | carried ids 243..256 | robot@id 248 (queued->251, lead 3) | srv 110 ms | L-gap 0.2cm R-gap 5.9cm
16:54:26 INFO real_world.inference_controller: [infer] #88 | start 16:54:26.364 end 16:54:26.789 | took 424.8 ms | carried ids 246..259 | robot@id 251 (queued->254, lead 3) | srv 107 ms | L-gap 0.0cm R-gap 5.9cm
[release-timing] 32 substeps/s | per-substep avg: total 20.8ms = recorder 0.0 + firmware 2.5 + dispatch 18.3 (STEP_TIME=8.3ms)
16:54:27 INFO real_world.grasp_recovery: [grasp-check] right grip=118.4 (close>=50.0) closed=True armed=True checked=True
[pipeline] append: obs_row=249, clock=254, appended ids 255..257 (+0 catch-up) -> queue 12
16:54:27 INFO real_world.inference_controller: [infer] #89 | start 16:54:26.789 end 16:54:27.257 | took 468.1 ms | carried ids 249..262 | robot@id 255 (queued->257, lead 2) | srv 113 ms | L-gap 0.2cm R-gap 6.1cm
16:54:27 INFO real_world.inference_controller: [infer] #90 | start 16:54:27.258 end 16:54:27.663 | took 405.3 ms | carried ids 252..265 | robot@id 257 (queued->260, lead 3) | srv 124 ms | L-gap 0.4cm R-gap 6.2cm
16:54:28 INFO real_world.inference_controller: [infer] #91 | start 16:54:27.666 end 16:54:28.075 | took 409.2 ms | carried ids 255..268 | robot@id 260 (queued->263, lead 3) | srv 131 ms | L-gap 0.2cm R-gap 6.3cm
[release-timing] 28 substeps/s | per-substep avg: total 20.0ms = recorder 0.0 + firmware 1.9 + dispatch 18.1 (STEP_TIME=8.3ms)
[pipeline] append: obs_row=258, clock=263, appended ids 264..266 (+0 catch-up) -> queue 12
16:54:28 INFO real_world.inference_controller: [infer] #92 | start 16:54:28.076 end 16:54:28.485 | took 408.8 ms | carried ids 258..271 | robot@id 263 (queued->266, lead 3) | srv 155 ms | L-gap 0.4cm R-gap 6.3cm
16:54:28 INFO real_world.inference_controller: [infer] #93 | start 16:54:28.491 end 16:54:28.880 | took 389.3 ms | carried ids 261..274 | robot@id 266 (queued->269, lead 3) | srv 125 ms | L-gap 0.2cm R-gap 6.6cm
[release-timing] 34 substeps/s | per-substep avg: total 21.8ms = recorder 0.0 + firmware 2.4 + dispatch 19.3 (STEP_TIME=8.3ms)
16:54:29 INFO real_world.grasp_recovery: [grasp-check] right grip=117.0 (close>=50.0) closed=True armed=True checked=True
16:54:29 INFO real_world.inference_controller: [infer] #94 | start 16:54:28.884 end 16:54:29.311 | took 426.5 ms | carried ids 264..277 | robot@id 269 (queued->272, lead 3) | srv 111 ms | L-gap 0.1cm R-gap 6.7cm
[pipeline] append: obs_row=267, clock=272, appended ids 273..275 (+0 catch-up) -> queue 12
16:54:29 INFO real_world.inference_controller: [infer] #95 | start 16:54:29.318 end 16:54:29.740 | took 421.6 ms | carried ids 267..280 | robot@id 273 (queued->275, lead 2) | srv 120 ms | L-gap 0.2cm R-gap 6.4cm
16:54:30 INFO real_world.inference_controller: [infer] #96 | start 16:54:29.753 end 16:54:30.161 | took 407.9 ms | carried ids 270..283 | robot@id 275 (queued->278, lead 3) | srv 119 ms | L-gap 0.3cm R-gap 6.7cm
[release-timing] 26 substeps/s | per-substep avg: total 19.1ms = recorder 0.0 + firmware 2.0 + dispatch 17.2 (STEP_TIME=8.3ms)
16:54:30 INFO real_world.inference_controller: [infer] #97 | start 16:54:30.168 end 16:54:30.584 | took 415.3 ms | carried ids 273..286 | robot@id 278 (queued->281, lead 3) | srv 108 ms | L-gap 0.3cm R-gap 6.8cm
[pipeline] append: obs_row=276, clock=281, appended ids 282..284 (+0 catch-up) -> queue 12
16:54:31 INFO real_world.inference_controller: [infer] #98 | start 16:54:30.584 end 16:54:31.019 | took 434.7 ms | carried ids 276..289 | robot@id 281 (queued->284, lead 3) | srv 125 ms | L-gap 0.1cm R-gap 7.2cm
[release-timing] 30 substeps/s | per-substep avg: total 20.8ms = recorder 0.0 + firmware 2.6 + dispatch 18.2 (STEP_TIME=8.3ms)
16:54:31 INFO real_world.grasp_recovery: [grasp-check] right grip=118.4 (close>=50.0) closed=True armed=True checked=True
16:54:31 INFO real_world.inference_controller: [infer] #99 | start 16:54:31.019 end 16:54:31.463 | took 443.5 ms | carried ids 279..292 | robot@id 284 (queued->287, lead 3) | srv 117 ms | L-gap 0.1cm R-gap 6.4cm
16:54:31 INFO real_world.inference_controller: [infer] #100 | start 16:54:31.464 end 16:54:31.885 | took 421.2 ms | carried ids 282..295 | robot@id 287 (queued->290, lead 3) | srv 117 ms | L-gap 0.2cm R-gap 6.1cm
16:54:32 INFO real_world.inference_controller: [auto] STOP requested — draining queue (100 inferences this run).
[pipeline] append: obs_row=285, clock=290, appended ids 291..293 (+0 catch-up) -> queue 12
16:54:32 INFO real_world.inference_controller: [infer] #101 | start 16:54:31.886 end 16:54:32.297 | took 410.3 ms | carried ids 285..298 | robot@id 290 (queued->293, lead 3) | srv 123 ms | L-gap 0.1cm R-gap 6.4cm
[release-timing] 30 substeps/s | per-substep avg: total 21.0ms = recorder 0.0 + firmware 2.8 + dispatch 18.2 (STEP_TIME=8.3ms)
[HumanoidEnv] E-STOP: latched; dropped 0 pending/staged cmds; holding pose.
[HumanoidEnv] E-STOP: latched; dropped 0 pending/staged cmds; holding pose.
[HumanoidEnv] E-stop reset; release re-enabled (run 执行 then 释放).
[HumanoidEnv] E-stop reset; release re-enabled (run 执行 then 释放).
16:54:34 INFO real_world.inference_controller: [auto] START -> robot | server 10.12.11.144:9000 | uncapped (latency-bound) | temporal-ensemble ON | grasp-recovery ON
16:54:34 INFO real_world.grasp_recovery: [grasp-check] right grip=118.3 (close>=50.0) closed=True armed=False checked=True
16:54:34 INFO real_world.grasp_recovery: [grasp-check] open->close (grip=118.3) -> attempt ARMED, detector check in 5.0s
[pipeline] append: obs_row=293, clock=293, appended ids 293..296 (+0 catch-up) -> queue 20
16:54:34 INFO real_world.inference_controller: [infer] #102 | start 16:54:34.533 end 16:54:34.926 | took 392.4 ms | carried ids 293..306 | robot@id 293 (queued->296, lead 3) | srv 116 ms | L-gap 0.4cm R-gap 3.5cm
16:54:34 INFO real_world.grasp_recovery: [grasp-check] grasp armed, settling 0.2/5.0s before detector runs
[release-timing] 12 substeps/s | per-substep avg: total 22.8ms = recorder 0.0 + firmware 2.5 + dispatch 20.2 (STEP_TIME=8.3ms)
16:54:35 INFO real_world.inference_controller: [infer] #103 | start 16:54:34.926 end 16:54:35.376 | took 449.9 ms | carried ids 293..306 | robot@id 296 (queued->299, lead 3) | srv 135 ms | L-gap 0.4cm R-gap 5.2cm
16:54:35 INFO real_world.inference_controller: [infer] #104 | start 16:54:35.376 end 16:54:35.842 | took 465.5 ms | carried ids 294..307 | robot@id 299 (queued->302, lead 3) | srv 157 ms | L-gap 0.3cm R-gap 6.6cm
[release-timing] 37 substeps/s | per-substep avg: total 18.8ms = recorder 0.0 + firmware 2.3 + dispatch 16.5 (STEP_TIME=8.3ms)
[pipeline] append: obs_row=297, clock=302, appended ids 303..305 (+0 catch-up) -> queue 12
16:54:36 INFO real_world.inference_controller: [infer] #105 | start 16:54:35.842 end 16:54:36.271 | took 428.5 ms | carried ids 297..310 | robot@id 302 (queued->305, lead 3) | srv 117 ms | L-gap 0.2cm R-gap 6.8cm
16:54:36 INFO real_world.grasp_recovery: [grasp-check] grasp armed, settling 1.5/5.0s before detector runs
16:54:36 INFO real_world.inference_controller: [infer] #106 | start 16:54:36.278 end 16:54:36.707 | took 428.7 ms | carried ids 300..313 | robot@id 305 (queued->308, lead 3) | srv 123 ms | L-gap 0.1cm R-gap 6.5cm
[release-timing] 30 substeps/s | per-substep avg: total 22.2ms = recorder 0.0 + firmware 3.1 + dispatch 19.1 (STEP_TIME=8.3ms)
16:54:37 INFO real_world.grasp_recovery: [grasp-check] right grip=119.5 (close>=50.0) closed=True armed=True checked=False
16:54:37 INFO real_world.inference_controller: [infer] #107 | start 16:54:36.709 end 16:54:37.142 | took 432.7 ms | carried ids 303..316 | robot@id 308 (queued->311, lead 3) | srv 111 ms | L-gap 0.2cm R-gap 6.5cm
[pipeline] append: obs_row=306, clock=311, appended ids 312..314 (+0 catch-up) -> queue 12
16:54:37 INFO real_world.inference_controller: [infer] #108 | start 16:54:37.150 end 16:54:37.601 | took 450.8 ms | carried ids 306..319 | robot@id 311 (queued->314, lead 3) | srv 108 ms | L-gap 0.4cm R-gap 6.4cm
16:54:37 INFO real_world.grasp_recovery: [grasp-check] grasp armed, settling 2.9/5.0s before detector runs
16:54:38 INFO real_world.inference_controller: [infer] #109 | start 16:54:37.604 end 16:54:38.021 | took 417.4 ms | carried ids 309..322 | robot@id 314 (queued->317, lead 3) | srv 126 ms | L-gap 0.1cm R-gap 5.8cm
[release-timing] 25 substeps/s | per-substep avg: total 19.6ms = recorder 0.0 + firmware 2.7 + dispatch 16.9 (STEP_TIME=8.3ms)
16:54:38 INFO real_world.inference_controller: [infer] #110 | start 16:54:38.022 end 16:54:38.433 | took 410.5 ms | carried ids 312..325 | robot@id 317 (queued->320, lead 3) | srv 98 ms | L-gap 0.2cm R-gap 5.4cm
[pipeline] append: obs_row=315, clock=320, appended ids 321..323 (+0 catch-up) -> queue 12
16:54:38 INFO real_world.inference_controller: [infer] #111 | start 16:54:38.433 end 16:54:38.872 | took 438.8 ms | carried ids 315..328 | robot@id 320 (queued->323, lead 3) | srv 110 ms | L-gap 0.3cm R-gap 5.9cm
16:54:38 INFO real_world.grasp_recovery: [grasp-check] grasp armed, settling 4.2/5.0s before detector runs
[release-timing] 30 substeps/s | per-substep avg: total 22.1ms = recorder 0.0 + firmware 3.2 + dispatch 18.9 (STEP_TIME=8.3ms)
16:54:39 INFO real_world.grasp_recovery: [grasp-check] right grip=118.0 (close>=50.0) closed=True armed=True checked=False
16:54:39 INFO real_world.inference_controller: [infer] #112 | start 16:54:38.885 end 16:54:39.300 | took 415.0 ms | carried ids 318..331 | robot@id 323 (queued->326, lead 3) | srv 107 ms | L-gap 0.1cm R-gap 6.4cm
16:54:39 INFO real_world.inference_controller: [infer] #113 | start 16:54:39.308 end 16:54:39.747 | took 438.8 ms | carried ids 321..334 | robot@id 326 (queued->329, lead 3) | srv 137 ms | L-gap 0.1cm R-gap 6.1cm
16:54:39 INFO real_world.grasp_recovery: [grasp-check] settle elapsed -> running detector on right wrist frame
16:54:40 WARNING real_world.grasp_recovery: [recovery] missed grasp (P_empty=0.57 >= thr=0.50) -> streaming retreat
[pipeline] append: obs_row=329, clock=329, appended ids 329..332 (+0 catch-up) -> queue 24
[release-timing] 30 substeps/s | per-substep avg: total 21.3ms = recorder 0.0 + firmware 1.0 + dispatch 20.3 (STEP_TIME=8.3ms)
16:54:41 INFO real_world.inference_controller: [auto] STOP requested — draining queue (113 inferences this run).
[HumanoidEnv] E-STOP: latched; dropped 0 pending/staged cmds; holding pose.
[HumanoidEnv] E-stop reset; release re-enabled (run 执行 then 释放).
[CameraHub] camera unsubscribed (OFF): head
[CameraHub] camera unsubscribed (OFF): hand_left
[CameraHub] camera unsubscribed (OFF): hand_right
16:54:45 INFO real_world.inference_controller: [auto] START -> robot | server 10.12.11.144:9000 | uncapped (latency-bound) | temporal-ensemble ON | grasp-recovery ON
[CameraHub] camera ON: head -> ['head']
[CameraHub] camera ON: hand_left -> ['hand_left', 'head']
[CameraHub] camera ON: hand_right -> ['hand_left', 'hand_right', 'head']
[ObsCollector] warming up: waiting for cameras ['head', 'hand_left', 'hand_right']…
[CameraHub] camera subscribed (ON): head
[CameraHub] camera subscribed (ON): hand_left
[CameraHub] camera subscribed (ON): hand_right
16:54:45 INFO real_world.grasp_recovery: [grasp-check] right grip=112.8 (close>=50.0) closed=True armed=False checked=True
16:54:45 INFO real_world.grasp_recovery: [grasp-check] open->close (grip=112.8) -> attempt ARMED, detector check in 5.0s
[pipeline] append: obs_row=334, clock=334, appended ids 334..337 (+0 catch-up) -> queue 16
16:54:46 INFO real_world.inference_controller: [infer] #114 | start 16:54:45.687 end 16:54:46.012 | took 325.0 ms | carried ids 334..347 | robot@id 334 (queued->337, lead 3) | srv 105 ms | L-gap 0.9cm R-gap 2.9cm
[release-timing] 32 substeps/s | per-substep avg: total 16.5ms = recorder 0.0 + firmware 1.2 + dispatch 15.3 (STEP_TIME=8.3ms)
16:54:46 INFO real_world.grasp_recovery: [grasp-check] grasp armed, settling 0.2/5.0s before detector runs
16:54:46 INFO real_world.inference_controller: [infer] #115 | start 16:54:46.018 end 16:54:46.483 | took 464.3 ms | carried ids 334..347 | robot@id 337 (queued->340, lead 3) | srv 110 ms | L-gap 1.5cm R-gap 3.8cm
16:54:46 INFO real_world.inference_controller: [infer] #116 | start 16:54:46.490 end 16:54:46.878 | took 388.5 ms | carried ids 335..348 | robot@id 340 (queued->343, lead 3) | srv 109 ms | L-gap 0.3cm R-gap 4.7cm
[release-timing] 35 substeps/s | per-substep avg: total 19.6ms = recorder 0.0 + firmware 3.0 + dispatch 16.6 (STEP_TIME=8.3ms)
[pipeline] append: obs_row=338, clock=343, appended ids 344..346 (+0 catch-up) -> queue 12
16:54:47 INFO real_world.inference_controller: [infer] #117 | start 16:54:46.879 end 16:54:47.301 | took 422.5 ms | carried ids 338..351 | robot@id 343 (queued->346, lead 3) | srv 107 ms | L-gap 0.4cm R-gap 5.1cm
16:54:47 INFO real_world.grasp_recovery: [grasp-check] grasp armed, settling 1.5/5.0s before detector runs
16:54:47 INFO real_world.inference_controller: [infer] #118 | start 16:54:47.302 end 16:54:47.749 | took 447.7 ms | carried ids 341..354 | robot@id 346 (queued->349, lead 3) | srv 133 ms | L-gap 0.5cm R-gap 5.3cm
16:54:48 INFO real_world.grasp_recovery: [grasp-check] right grip=118.0 (close>=50.0) closed=True armed=True checked=False
16:54:48 INFO real_world.inference_controller: [infer] #119 | start 16:54:47.753 end 16:54:48.183 | took 429.7 ms | carried ids 344..357 | robot@id 349 (queued->352, lead 3) | srv 126 ms | L-gap 0.4cm R-gap 5.7cm
[release-timing] 29 substeps/s | per-substep avg: total 19.4ms = recorder 0.0 + firmware 2.8 + dispatch 16.7 (STEP_TIME=8.3ms)
[pipeline] append: obs_row=347, clock=352, appended ids 353..355 (+0 catch-up) -> queue 12
16:54:48 INFO real_world.inference_controller: [infer] #120 | start 16:54:48.185 end 16:54:48.579 | took 393.0 ms | carried ids 347..360 | robot@id 352 (queued->355, lead 3) | srv 104 ms | L-gap 0.8cm R-gap 5.9cm
16:54:48 INFO real_world.grasp_recovery: [grasp-check] grasp armed, settling 2.7/5.0s before detector runs
16:54:48 INFO real_world.inference_controller: [infer] #121 | start 16:54:48.580 end 16:54:48.970 | took 390.2 ms | carried ids 350..363 | robot@id 355 (queued->358, lead 3) | srv 123 ms | L-gap 0.9cm R-gap 5.7cm
[release-timing] 35 substeps/s | per-substep avg: total 20.2ms = recorder 0.0 + firmware 3.0 + dispatch 17.2 (STEP_TIME=8.3ms)
16:54:49 INFO real_world.inference_controller: [infer] #122 | start 16:54:48.974 end 16:54:49.425 | took 451.6 ms | carried ids 353..366 | robot@id 358 (queued->361, lead 3) | srv 136 ms | L-gap 0.7cm R-gap 5.4cm
[pipeline] append: obs_row=356, clock=361, appended ids 362..364 (+0 catch-up) -> queue 12
16:54:49 INFO real_world.inference_controller: [infer] #123 | start 16:54:49.426 end 16:54:49.824 | took 398.1 ms | carried ids 356..369 | robot@id 362 (queued->364, lead 2) | srv 148 ms | L-gap 0.2cm R-gap 4.9cm
16:54:49 INFO real_world.grasp_recovery: [grasp-check] grasp armed, settling 4.0/5.0s before detector runs
16:54:50 INFO real_world.grasp_recovery: [grasp-check] right grip=115.3 (close>=50.0) closed=True armed=True checked=False
16:54:50 INFO real_world.inference_controller: [infer] #124 | start 16:54:49.835 end 16:54:50.242 | took 406.5 ms | carried ids 359..372 | robot@id 364 (queued->367, lead 3) | srv 113 ms | L-gap 0.1cm R-gap 5.1cm
[release-timing] 25 substeps/s | per-substep avg: total 19.6ms = recorder 0.0 + firmware 2.3 + dispatch 17.2 (STEP_TIME=8.3ms)
16:54:50 INFO real_world.inference_controller: [infer] #125 | start 16:54:50.242 end 16:54:50.668 | took 425.8 ms | carried ids 362..375 | robot@id 367 (queued->370, lead 3) | srv 134 ms | L-gap 0.4cm R-gap 5.2cm
[pipeline] append: obs_row=365, clock=370, appended ids 371..373 (+0 catch-up) -> queue 12
16:54:51 INFO real_world.inference_controller: [infer] #126 | start 16:54:50.669 end 16:54:51.061 | took 391.8 ms | carried ids 365..378 | robot@id 370 (queued->373, lead 3) | srv 119 ms | L-gap 0.9cm R-gap 4.7cm
16:54:51 INFO real_world.grasp_recovery: [grasp-check] settle elapsed -> running detector on right wrist frame
[release-timing] 33 substeps/s | per-substep avg: total 20.7ms = recorder 0.0 + firmware 2.5 + dispatch 18.2 (STEP_TIME=8.3ms)
16:54:51 WARNING real_world.grasp_recovery: [recovery] missed grasp (P_empty=1.00 >= thr=0.50) -> streaming retreat
[pipeline] append: obs_row=373, clock=375, appended ids 378..378 (+0 catch-up) -> queue 13
[release-timing] 32 substeps/s | per-substep avg: total 16.0ms = recorder 0.0 + firmware 0.8 + dispatch 15.1 (STEP_TIME=8.3ms)
16:54:52 INFO real_world.grasp_recovery: [recovery] retreat complete -> policy re-approaches
16:54:52 INFO real_world.grasp_recovery: [grasp-check] right grip=109.1 (close>=50.0) closed=True armed=False checked=True
16:54:52 INFO real_world.grasp_recovery: [grasp-check] open->close (grip=109.1) -> attempt ARMED, detector check in 5.0s
16:54:52 INFO real_world.inference_controller: [infer] #127 | start 16:54:52.636 end 16:54:52.943 | took 307.3 ms | carried ids 385..398 | robot@id 384 (queued->387, lead 3) | srv 113 ms | anchor idx -1 OOR (buf (13, 20))
16:54:52 INFO real_world.grasp_recovery: [grasp-check] grasp armed, settling 0.1/5.0s before detector runs
[pipeline] append: obs_row=385, clock=387, appended ids 388..390 (+0 catch-up) -> queue 12
16:54:53 INFO real_world.inference_controller: [infer] #128 | start 16:54:52.946 end 16:54:53.354 | took 408.1 ms | carried ids 385..398 | robot@id 387 (queued->390, lead 3) | srv 119 ms | L-gap 2.1cm R-gap 2.9cm
[release-timing] 35 substeps/s | per-substep avg: total 17.1ms = recorder 0.0 + firmware 0.9 + dispatch 16.2 (STEP_TIME=8.3ms)
16:54:53 INFO real_world.inference_controller: [infer] #129 | start 16:54:53.357 end 16:54:53.740 | took 382.9 ms | carried ids 385..398 | robot@id 390 (queued->393, lead 3) | srv 106 ms | L-gap 0.7cm R-gap 4.0cm
16:54:54 INFO real_world.inference_controller: [infer] #130 | start 16:54:53.741 end 16:54:54.147 | took 406.2 ms | carried ids 388..401 | robot@id 393 (queued->396, lead 3) | srv 113 ms | L-gap 0.4cm R-gap 4.3cm
16:54:54 INFO real_world.grasp_recovery: [grasp-check] grasp armed, settling 1.3/5.0s before detector runs
[pipeline] append: obs_row=391, clock=396, appended ids 397..399 (+0 catch-up) -> queue 12
16:54:54 INFO real_world.inference_controller: [infer] #131 | start 16:54:54.150 end 16:54:54.537 | took 387.2 ms | carried ids 391..404 | robot@id 396 (queued->399, lead 3) | srv 99 ms | L-gap 0.1cm R-gap 4.3cm
[release-timing] 36 substeps/s | per-substep avg: total 18.8ms = recorder 0.0 + firmware 1.9 + dispatch 16.9 (STEP_TIME=8.3ms)
16:54:54 INFO real_world.inference_controller: [infer] #132 | start 16:54:54.538 end 16:54:54.909 | took 371.0 ms | carried ids 394..407 | robot@id 399 (queued->402, lead 3) | srv 115 ms | L-gap 0.1cm R-gap 5.1cm
16:54:55 INFO real_world.grasp_recovery: [grasp-check] right grip=115.4 (close>=50.0) closed=True armed=True checked=False
16:54:55 INFO real_world.inference_controller: [infer] #133 | start 16:54:54.910 end 16:54:55.343 | took 433.1 ms | carried ids 397..410 | robot@id 402 (queued->405, lead 3) | srv 135 ms | L-gap 0.5cm R-gap 5.1cm
16:54:55 INFO real_world.grasp_recovery: [grasp-check] grasp armed, settling 2.6/5.0s before detector runs
[release-timing] 32 substeps/s | per-substep avg: total 21.0ms = recorder 0.0 + firmware 2.4 + dispatch 18.6 (STEP_TIME=8.3ms)
[pipeline] append: obs_row=400, clock=405, appended ids 406..408 (+0 catch-up) -> queue 12
16:54:55 INFO real_world.inference_controller: [infer] #134 | start 16:54:55.347 end 16:54:55.779 | took 432.1 ms | carried ids 400..413 | robot@id 405 (queued->408, lead 3) | srv 139 ms | L-gap 0.5cm R-gap 4.9cm
16:54:56 INFO real_world.inference_controller: [infer] #135 | start 16:54:55.780 end 16:54:56.192 | took 412.2 ms | carried ids 403..416 | robot@id 408 (queued->411, lead 3) | srv 133 ms | L-gap 0.3cm R-gap 4.3cm
16:54:56 INFO real_world.inference_controller: [infer] #136 | start 16:54:56.196 end 16:54:56.622 | took 425.4 ms | carried ids 406..419 | robot@id 411 (queued->414, lead 3) | srv 114 ms | L-gap 0.7cm R-gap 3.9cm
[release-timing] 28 substeps/s | per-substep avg: total 19.1ms = recorder 0.0 + firmware 2.5 + dispatch 16.6 (STEP_TIME=8.3ms)
16:54:56 INFO real_world.grasp_recovery: [grasp-check] grasp armed, settling 3.8/5.0s before detector runs
[pipeline] append: obs_row=409, clock=414, appended ids 415..417 (+0 catch-up) -> queue 12
16:54:57 INFO real_world.inference_controller: [infer] #137 | start 16:54:56.622 end 16:54:57.070 | took 447.4 ms | carried ids 409..422 | robot@id 415 (queued->417, lead 2) | srv 116 ms | L-gap 0.3cm R-gap 3.9cm
16:54:57 INFO real_world.grasp_recovery: [grasp-check] right grip=118.3 (close>=50.0) closed=True armed=True checked=False
16:54:57 INFO real_world.inference_controller: [infer] #138 | start 16:54:57.073 end 16:54:57.471 | took 398.8 ms | carried ids 412..425 | robot@id 417 (queued->420, lead 3) | srv 114 ms | L-gap 0.9cm R-gap 4.2cm
[release-timing] 33 substeps/s | per-substep avg: total 19.6ms = recorder 0.0 + firmware 2.2 + dispatch 17.4 (STEP_TIME=8.3ms)
16:54:57 INFO real_world.inference_controller: [infer] #139 | start 16:54:57.472 end 16:54:57.898 | took 425.2 ms | carried ids 415..428 | robot@id 420 (queued->423, lead 3) | srv 157 ms | L-gap 0.1cm R-gap 5.0cm
16:54:57 INFO real_world.grasp_recovery: [grasp-check] settle elapsed -> running detector on right wrist frame
16:54:58 WARNING real_world.grasp_recovery: [recovery] missed grasp (P_empty=0.99 >= thr=0.50) -> streaming retreat
[pipeline] append: obs_row=423, clock=423, appended ids 423..426 (+0 catch-up) -> queue 20
[release-timing] 15 substeps/s | per-substep avg: total 26.7ms = recorder 0.0 + firmware 7.3 + dispatch 19.4 (STEP_TIME=8.3ms)
16:54:59 INFO real_world.grasp_recovery: [recovery] retreat complete -> policy re-approaches
16:54:59 INFO real_world.grasp_recovery: [grasp-check] right grip=50.8 (close>=50.0) closed=True armed=False checked=True
16:54:59 INFO real_world.grasp_recovery: [grasp-check] open->close (grip=50.8) -> attempt ARMED, detector check in 5.0s
[pipeline] append: obs_row=435, clock=434, appended ids 435..437 (+0 catch-up) -> queue 16
16:54:59 INFO real_world.inference_controller: [infer] #140 | start 16:54:59.542 end 16:54:59.898 | took 356.2 ms | carried ids 435..448 | robot@id 434 (queued->437, lead 3) | srv 114 ms | anchor idx -1 OOR (buf (13, 20))
16:54:59 INFO real_world.grasp_recovery: [grasp-check] grasp armed, settling 0.2/5.0s before detector runs
[release-timing] 52 substeps/s | per-substep avg: total 15.8ms = recorder 0.0 + firmware 0.9 + dispatch 14.9 (STEP_TIME=8.3ms)
16:55:00 INFO real_world.inference_controller: [infer] #141 | start 16:54:59.901 end 16:55:00.361 | took 459.5 ms | carried ids 435..448 | robot@id 437 (queued->440, lead 3) | srv 116 ms | L-gap 0.9cm R-gap 1.8cm
16:55:00 INFO real_world.inference_controller: [infer] #142 | start 16:55:00.364 end 16:55:00.791 | took 426.7 ms | carried ids 435..448 | robot@id 440 (queued->443, lead 3) | srv 113 ms | L-gap 0.4cm R-gap 3.4cm
[release-timing] 33 substeps/s | per-substep avg: total 19.4ms = recorder 0.0 + firmware 1.6 + dispatch 17.8 (STEP_TIME=8.3ms)
[pipeline] append: obs_row=438, clock=443, appended ids 444..446 (+0 catch-up) -> queue 12
16:55:01 INFO real_world.inference_controller: [infer] #143 | start 16:55:00.791 end 16:55:01.216 | took 424.5 ms | carried ids 438..451 | robot@id 443 (queued->446, lead 3) | srv 129 ms | L-gap 0.3cm R-gap 3.7cm
16:55:01 INFO real_world.grasp_recovery: [grasp-check] grasp armed, settling 1.5/5.0s before detector runs
16:55:01 INFO real_world.inference_controller: [infer] #144 | start 16:55:01.217 end 16:55:01.630 | took 413.2 ms | carried ids 441..454 | robot@id 446 (queued->449, lead 3) | srv 130 ms | L-gap 0.5cm R-gap 3.4cm
16:55:01 INFO real_world.grasp_recovery: [grasp-check] right grip=110.4 (close>=50.0) closed=True armed=True checked=False
16:55:02 INFO real_world.inference_controller: [infer] #145 | start 16:55:01.631 end 16:55:02.051 | took 419.9 ms | carried ids 444..457 | robot@id 449 (queued->452, lead 3) | srv 125 ms | L-gap 0.2cm R-gap 4.2cm
[release-timing] 31 substeps/s | per-substep avg: total 19.8ms = recorder 0.0 + firmware 2.8 + dispatch 16.9 (STEP_TIME=8.3ms)
[pipeline] append: obs_row=447, clock=452, appended ids 453..455 (+0 catch-up) -> queue 12
16:55:02 INFO real_world.inference_controller: [infer] #146 | start 16:55:02.054 end 16:55:02.479 | took 425.5 ms | carried ids 447..460 | robot@id 453 (queued->455, lead 2) | srv 118 ms | L-gap 0.4cm R-gap 4.3cm
16:55:02 INFO real_world.grasp_recovery: [grasp-check] grasp armed, settling 2.7/5.0s before detector runs
16:55:02 INFO real_world.inference_controller: [infer] #147 | start 16:55:02.480 end 16:55:02.938 | took 458.1 ms | carried ids 450..463 | robot@id 455 (queued->458, lead 3) | srv 117 ms | L-gap 0.7cm R-gap 4.3cm
[release-timing] 30 substeps/s | per-substep avg: total 21.6ms = recorder 0.0 + firmware 2.6 + dispatch 19.0 (STEP_TIME=8.3ms)
16:55:03 INFO real_world.inference_controller: [infer] #148 | start 16:55:02.939 end 16:55:03.380 | took 441.1 ms | carried ids 453..466 | robot@id 458 (queued->461, lead 3) | srv 119 ms | L-gap 0.2cm R-gap 4.1cm
[pipeline] append: obs_row=456, clock=461, appended ids 462..464 (+0 catch-up) -> queue 12
16:55:03 INFO real_world.inference_controller: [infer] #149 | start 16:55:03.380 end 16:55:03.788 | took 408.1 ms | carried ids 456..469 | robot@id 461 (queued->464, lead 3) | srv 118 ms | L-gap 0.5cm R-gap 3.8cm
16:55:03 INFO real_world.grasp_recovery: [grasp-check] grasp armed, settling 4.1/5.0s before detector runs
16:55:04 INFO real_world.grasp_recovery: [grasp-check] right grip=99.2 (close>=50.0) closed=True armed=True checked=False
16:55:04 INFO real_world.inference_controller: [infer] #150 | start 16:55:03.789 end 16:55:04.279 | took 490.3 ms | carried ids 459..472 | robot@id 464 (queued->467, lead 3) | srv 112 ms | L-gap 0.2cm R-gap 4.3cm
[release-timing] 30 substeps/s | per-substep avg: total 19.7ms = recorder 0.0 + firmware 1.6 + dispatch 18.1 (STEP_TIME=8.3ms)
16:55:04 INFO real_world.inference_controller: [infer] #151 | start 16:55:04.282 end 16:55:04.706 | took 423.4 ms | carried ids 462..475 | robot@id 467 (queued->470, lead 3) | srv 101 ms | L-gap 0.3cm R-gap 4.7cm
[pipeline] append: obs_row=465, clock=470, appended ids 471..473 (+0 catch-up) -> queue 12
16:55:05 INFO real_world.inference_controller: [infer] #152 | start 16:55:04.717 end 16:55:05.164 | took 446.8 ms | carried ids 465..478 | robot@id 471 (queued->473, lead 2) | srv 120 ms | L-gap 0.4cm R-gap 4.3cm
16:55:05 INFO real_world.grasp_recovery: [grasp-check] settle elapsed -> running detector on right wrist frame
[release-timing] 29 substeps/s | per-substep avg: total 21.4ms = recorder 0.0 + firmware 3.1 + dispatch 18.4 (STEP_TIME=8.3ms)
16:55:05 WARNING real_world.grasp_recovery: [recovery] missed grasp (P_empty=0.99 >= thr=0.50) -> streaming retreat
[pipeline] append: obs_row=473, clock=474, appended ids 477..477 (+0 catch-up) -> queue 12
[release-timing] 19 substeps/s | per-substep avg: total 19.4ms = recorder 0.0 + firmware 1.7 + dispatch 17.8 (STEP_TIME=8.3ms)
16:55:06 INFO real_world.grasp_recovery: [recovery] retreat complete -> policy re-approaches
16:55:07 INFO real_world.inference_controller: [auto] STOP requested — draining queue (152 inferences this run).
16:55:07 INFO real_world.grasp_recovery: [grasp-check] right grip=111.6 (close>=50.0) closed=True armed=False checked=True
16:55:07 INFO real_world.grasp_recovery: [grasp-check] open->close (grip=111.6) -> attempt ARMED, detector check in 5.0s
[pipeline] append: obs_row=485, clock=484, appended ids 485..487 (+0 catch-up) -> queue 16
16:55:07 INFO real_world.inference_controller: [infer] #153 | start 16:55:07.012 end 16:55:07.370 | took 358.2 ms | carried ids 485..498 | robot@id 484 (queued->487, lead 3) | srv 116 ms | anchor idx -1 OOR (buf (13, 20))
[release-timing] 40 substeps/s | per-substep avg: total 16.5ms = recorder 0.0 + firmware 1.1 + dispatch 15.4 (STEP_TIME=8.3ms)
^Z[6]   Killed                  python robot_control_gui.py

[7]+  Stopped                 python robot_control_gui.py
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
