# Humanoid


this project utilized the diffusion policy framework by rea-standford lab.


Teleoperation, data-collection, and policy-deployment stack for the AgiBot / 智元
**A2D** dual-arm humanoid, built on the vendored **a2d_sdk** robot SDK.

One codebase covers the full loop of imitation-learning robotics:

- **Teleoperate** the robot from a Pico VR headset and **record** demonstrations.
- **Build datasets** from those recordings into replay-buffer `.zarr` for policy training.
- **Deploy** a trained dual-arm policy: stream camera + proprioception observations to a
  policy server, turn the returned action chunks into safe, smooth arm motion, and complete
  the task with scripted place macros and YOLO perception gates.
- **Evaluate** on the robot or offline in a PyBullet sim, with a live success-rate dashboard.

Everything is driven from one Tkinter control console
([robot_control_gui.py](robot_control_gui.py)), which also runs **hardware-free** for demos,
screenshots, and UI work.

---

## How deployment works

The policy never talks to the robot directly. Observations flow out, action chunks flow
back, and the stack turns each chunk into velocity-bounded, temporally-consistent motion:

```
  ┌─────────────── HumanoidEnv (owns SDK + two threads) ───────────────┐
  │                                                                    │
  │  collect thread ─▶ cameras + dual-arm EE pose + grippers (10 Hz)   │
  │        │                                                           │
  │        ▼                                                           │
  │   get_obs() ──build_data──▶  ┌──────────────┐                      │
  │                              │ POLICY SERVER │  /predict            │
  │   submit_actions() ◀─────────└──────────────┘                      │
  │        │  action chunk (dual-arm EE + grip)                        │
  │        ▼                                                           │
  │   postprocess:  binarize grip ▸ temporal-ensemble merge ▸          │
  │                 dual-arm IK ▸ sim validate ▸ splice onto queue     │
  │        │                                                           │
  │        ▼                                                           │
  │  exec thread ─▶ drain queue, stream substeps to arms (120 Hz)      │
  └────────────────────────────────────────────────────────────────────┘
        ▲ inference loop (caller-owned):  obs → server → submit, repeat
```

the gripper was none binarized as the input to provie the model more information on the state of the gripper, while binarization in the data postprocessing, eliminates alot of fluctuations in the output command.


- **Observation** ([build_data.py](real_world/build_data.py)) — head + two wrist cameras
  (cropped/resized to 16:9, base64 JPEG) plus per-arm EE pose `[pos(3) + rot6d(6)]` and both
  gripper states, in the exact layout the policy was trained on.
- **Action post-processing** ([postprocess.py](real_world/postprocess.py)) — the output-side
  mirror of `build_data`: gripper binarize → cross-chunk temporal-ensemble merge (master-row-id
  aligned) → dual-arm IK → sim validation → splice onto the robot queue.
- **Inverse kinematics** ([ik.py](real_world/ik.py)) — a transparent URDF + Pinocchio solver
  that replaces the SDK's firmware IK blackbox; the same solver runs in the PyBullet check and
  on the robot. Calibrated by [config/fk_calibration*.json](real_world/config/).
- **Timing** ([timing.py](real_world/timing.py)) — single source of truth. `RECORD_HZ` (10 Hz)
  is the policy's action-row cadence; `CONTROL_HZ` (120 Hz) is the substep streaming rate. Every
  substep is bounded by `MAX_JOINT_VEL` / `MAX_JOINT_STEP`, with a `WATCHDOG_MAX_JOINT_JUMP`
  E-stop on any oversized single command.

## Manipulation pipeline

The policy only **grabs, lifts, and (sometimes) flips**. These modules run **inline in the
auto-inference loop** to finish the task — placing, releasing, recovering, and gating — each
self-contained with only small call-ins from the inference controller:

| Module | Role |
|--------|------|
| [flip_place.py](real_world/flip_place.py) | Scripted release **after** the policy's ~180° flip: warp a recorded path onto the live pose, move out, open, reverse back. |
| [no_flip_place.py](real_world/no_flip_place.py) | Same, for the **no-flip** case; the "place now" cue is a YOLO detection with a commit latch. |
| [grasp_recovery.py](real_world/grasp_recovery.py) | Wrist-camera YOLO (open / closed-gripped / **closed-empty**). On a failed grasp: clear queue, open, home both arms, let the policy re-plan. |
| [package_gate.py](real_world/package_gate.py) | Same model's `package` class — pause inference + park at home when there's nothing to work on. **Fail-open**: never pauses if it can't detect. |
| [retreat.py](real_world/retreat.py) | Torch-free, velocity-bounded "retreat to home" primitive shared by recovery and the unreachable-target handler. |

Detection runs **locally** on the client (ultralytics + `.pt` weights under
[real_world/assets/](real_world/assets/) and [data/](data/)), not on the policy server. The
**Detector tuning** GUI tab shows a live boxed head-cam view for tuning these gates.

## SDK-free split

The SDK (`a2d_sdk`) and ROS live **only on the robot machine**. The kinematics, sim, camera hub,
recorded-obs playback, and timing are written to import without them — via lazy package imports
and dependency injection — so the whole sim/eval/CI path (and `--demo` mode) runs on any laptop.
[requirements_sim.txt](requirements_sim.txt) is the SDK-free dependency set.

---

## Repository layout

| Path | What it is |
|------|------------|
| [robot_control_gui.py](robot_control_gui.py) | Main entry point — the Tkinter control console; assembles the GUI mixins and wires up robot / camera / env / inference. |
| [real_world/](real_world/) | Core library — the env, IK, inference controller, post-processing, sim backend, timing, data building, and the manipulation-pipeline modules above. |
| [gui/](gui/) | GUI feature mixins, one per domain, combined by multiple inheritance in `RobotControlGUI` (`Style`, `Camera`, `Inference`, `DetectorTuning`, `VR`, `DataCollection`, `Eval`) + the hardware-free [demo_backend.py](gui/demo_backend.py). |
| [pico_vr/](pico_vr/) | Pico VR teleop client/server and shared wire protocol. |
| [servers/](servers/) | Networking glue: robot-info HTTP server, inference-server ping, recording upload. |
| [MDM_data_collection/](MDM_data_collection/) | Data-collection GUI + dataset builder: recorded episodes → replay-buffer `.zarr` (recordings/buffers not versioned). |
| [scripts/](scripts/) | Eval, diagnostics, and path/waypoint builders (sim inference eval, FK-consistency check, retreat-waypoint estimation, …). |
| [tests/](tests/) | Pytest suite — safety invariants, append/splice, retreat waypoints, tagging. |
| [planning/](planning/) | Reachability, detection, and planning server subsystem. |
| [examples/](examples/) | Standalone reference scripts (e.g. wheel/base control). |
| [data/](data/), [real_world/assets/](real_world/assets/) | Local model weights (YOLO `.pt`) and recorded release paths / URDF assets. |
| [a2d_sdk/](a2d_sdk/) | Vendored A2D robot SDK (runtime dependency, imported as `a2d_sdk.robot`). |
| [docs/](docs/) | Vendor manuals and product references. |

> `G1_SDK_ENV/` is a legacy SDK snapshot kept on disk but excluded from version control.

---

## Getting started

Run everything from the repository root so the top-level packages (`real_world`, `gui`,
`servers`, `a2d_sdk`, …) resolve.

```bash
# On the robot machine — full stack (needs a2d_sdk + ROS + the robot):
pip install -r requirements_gui.txt          # numpy, opencv, ruckig, zxing-cpp, pyzmq, matplotlib, …
# ultralytics + the .pt weights are additionally needed for grasp recovery / package gating.
python robot_control_gui.py

# On any laptop — hardware-free DEMO mode (synthetic robot, live camera feeds,
# a filling evaluation dashboard; no SDK/ROS/robot):
python robot_control_gui.py --demo

# SDK-free sim / IK tools (Pinocchio + PyBullet):
python -m venv .venv && .venv/bin/pip install -r requirements_sim.txt
.venv/bin/python scripts/sim_infer_eval.py    # run the policy against a recording in sim
```

The console has four tabs:

- **Console** — camera views + policy inference (sim preview, validate, release, substep monitor).
- **Detector tuning** — live boxed head-cam view for tuning the YOLO place/package gates.
- **VR teleop** — Pico teleoperation + demonstration recording.
- **Evaluation** — live success-rate KPI dashboard fed by `infer_logs/eval/*.jsonl` (the logs
  [scripts/eval_trials.py](scripts/eval_trials.py) writes).

The UI theme and all ttk styles live in [gui/styles.py](gui/styles.py).

## Testing

```bash
pytest tests/          # safety invariants, append/splice, retreat waypoints, tagging
```

## Notes on runtime state

- [tuning_config.json](tuning_config.json) holds live tuning overrides (temporal-ensemble params,
  `speed_scale`, `append_ahead_rows`), loaded **once** at GUI startup; code constants are the
  defaults when a key is absent. Delete it to reset.
- Calibration lives in [real_world/config/](real_world/config/): `fk_calibration*.json` (the
  IK base offset, per arm), `nominal_arm_config.json`, `retreat_waypoints.json`. Both arms'
  calibrations must match the deployment torso pose.
- `live_joints.jsonl`, `released_substeps.jsonl`, recordings, `.zarr` buffers, and
  `infer_logs/eval/` are runtime scratch/data and are git-ignored.



Data training

you have to exagerate alot of actions for it to learn

for end to end model like this you have to be very explicit about what actions it should take

for example for grabbing packages, because packages are so low and close to the table. 

the head camera sees the package taks up x1 to x2 and y1 to y2 position of its camera, than you need to make sure your gripper always approaches x1 - n amount and x2 + n amount where this n can for example be 5cms. Once you are consistent on this, it will remeber to do so. This is the nature of a Motion diffusion model that learns from vision

Another important 


A drawback of diffusion policy model is that it at the current moment it requires a fixed head position.

In the original paper the implementation is all done at a fixed platform with a fixed camera on top



Ro alternative the author cheng-chi developed a hand held data collection tool that only uses the camera on the hand to complete tasks.

this frame is very efficient in training

the robot is able to relatively replicate the motion at 50 sets of data at around 50 to 100 epoches.

the largest model I have used was 200 and I was already expereiencing dimenishing returns above 100.

with around 100 to 200 set of data it is already capable to replicating small movements in my training data such as the tendency to release than use the lower half of the gripper to move the package into a straight position before pulling out the gripper fully.

to be explicit the trianed data we used are ened effector position and 6D orientation. the first because it is generally easier for the model to learn and 6D orientation because it has the adventage of being coutinuous compared to quatarian representation.


