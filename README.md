# Humanoid — Dual-Arm Diffusion Policy for Parcel Handling

Teleoperation, data-collection, and policy-deployment stack for the **AgiBot / 智元 Genie G1**
dual-arm humanoid, built on the vendored **a2d_sdk** robot SDK.

The robot picks a parcel off a table with two arms, lifts it, decides whether the shipping
label is visible, flips the parcel ~180° if it is not, and places it label-up for downstream
barcode scanning. The manipulation policy is a **diffusion policy** (Chi et al., Stanford
REALab), image-conditioned and trained by imitation from VR teleoperation demonstrations.

One codebase covers the full imitation-learning loop:

- **Teleoperate** from a Pico VR headset and **record** demonstrations.
- **Build datasets** from those recordings into replay-buffer `.zarr` for policy training.
- **Deploy** the trained policy: stream observations to a policy server, turn the returned
  action chunks into safe, smooth arm motion, and finish the task with scripted place macros.
- **Evaluate** on the robot or offline in a PyBullet sim, with a live success-rate dashboard.

Everything runs from one Tkinter console ([robot_control_gui.py](robot_control_gui.py)), which
also runs **hardware-free** (`--demo`) for demos, screenshots, and UI work.

> **📖 This README is the map, not the manual.** For the full rationale, debugging history,
> and tuning reference, read
> [KNOWHOW_Humanoid_Diffusion_Policy.md](documenation/Documentation_English/KNOWHOW_Humanoid_Diffusion_Policy.md)
> (中文: [KNOWHOW_..._CN.md](documenation/Documentation_中文/KNOWHOW_Humanoid_Diffusion_Policy_CN.md)).
> A PoC report and slide deck sit alongside them.

---

## What was used

| | Choice | Why / note |
|---|---|---|
| Robot | AgiBot / 智元 Genie **G1** dual-arm humanoid | SDK is `a2d_sdk`, URDF is `A2D.urdf` — name the robot and SDK separately, this has confused people |
| Policy | Diffusion policy — image-conditioned, dual-arm, **EE-space** | runs as a separate HTTP service (`POST /predict`); never drives the robot directly |
| Cameras | head + 2 wrist | own ResNet encoder each: head = coarse positioning, wrists = alignment + gripper |
| Kinematics | **Pinocchio + in-repo URDF**, replacing the SDK's black-box IK | same solver runs in sim and on the robot — the basis of every safety check |
| Sim | **PyBullet** (in-process) | chosen over Isaac Lab for simplicity + runs on a GPU-less laptop |
| Teleop | Pico VR headset over ZeroMQ | |
| Vision | **YOLO** (local, ultralytics) | gates placing / recovery / start-stop; picked after barcode, ArUco, and text-block detectors proved unstable |
| Timing | `RECORD_HZ` = 10 Hz, `CONTROL_HZ` = 120 Hz | policy row cadence vs. substep streaming rate |

---

## How deployment works

The policy never talks to the robot directly. Observations flow out, action chunks flow back,
and the stack turns each chunk into velocity-bounded, temporally-consistent motion:

```
  ┌─────────────── HumanoidEnv (owns SDK + threads) ───────────────────┐
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
        ▲ inference loop:  obs → server → submit, repeat
```

- **Observation** ([build_data.py](real_world/build_data.py)) — head + two wrist cameras
  (cropped/resized to 16:9, base64 JPEG) plus per-arm EE pose `[pos(3) + rot6d(6)]` and both
  gripper states, in the **exact layout the policy trained on**.
- **Post-processing** ([postprocess.py](real_world/postprocess.py)) — the output-side mirror of
  `build_data`: gripper binarize → cross-chunk temporal-ensemble merge → dual-arm IK → sim
  validation → splice onto the robot queue. Training imports the *same* build functions, so
  train/deploy parity holds by construction — most model-quality regressions traced back to a
  divergence between these two sides.
- **Inverse kinematics** ([ik.py](real_world/ik.py)) — a transparent URDF + Pinocchio
  damped-least-squares solver replacing the SDK's firmware IK black-box. Calibrated per arm
  ([config/fk_calibration*.json](real_world/config/)) to sub-millimetre FK agreement.
- **Timing** ([timing.py](real_world/timing.py)) — **single source of truth** for every rate
  and limit constant. Each policy row expands to a fixed number of substeps *uniform in time*,
  so wall-clock per row is independent of how far the joints move.

**Alignment keys on absolute master row IDs**, not a clock: every action row carries an ID from
the robot's own execution clock, so new inferences merge correctly against what is already
queued, surviving variable inference latency (the failure mode of the earlier clock-based sync).

### Safety is layered and independent of the policy

Invariants **C1–C7 / H1–H4**, pinned by `tests/test_safety_invariants.py` and run as a launch
pre-flight. The load-bearing ones: no sim running → nothing reaches the robot (C1); every
released trajectory was stepped and self-collision-checked in sim (C2); a latched E-stop (C3);
a per-command displacement watchdog that latches E-stop on any oversized jump (C6); and an
EE-outside-safe-box guard (C7). See §3 of the knowhow for the full table.

### Gripper: three mechanisms, easily confused

- **Obs side** — the model receives the **non-binarised commanded** value, so it sees true
  continuous state (firmware read-back lags by *seconds*, which made the policy re-issue grasps
  it had already completed).
- **Action side** — **binarised** in post-processing → clean open/close decision out, which
  eliminated a large amount of command fluctuation.
- **Dispatch side** — an anti-chatter **change latch** locks a gripper state for 20 row IDs so
  an oscillating policy cannot re-grab.

---

## Manipulation pipeline — hybrid policy + scripted macros

The policy only **grabs, lifts, and (sometimes) flips**. Everything geometrically constrained
and repeatable is a scripted macro running **inline in the auto-inference loop**. This reserves
the policy's capacity for the contact-rich part of the task and is far cheaper than the
demonstrations needed to learn placement — at the stated cost that new placement locations
require additional scripting.

| Module | Role |
|--------|------|
| [package_gate.py](real_world/package_gate.py) | Pause inference + park at home when there's no parcel. **Fail-open**: never pauses if it can't detect. |
| [no_flip_place.py](real_world/no_flip_place.py) | Place as-is: warp a recorded joint path onto the live pose, move out, open, reverse back. Cue = YOLO `barcode` with a commit latch. |
| [flip_place.py](real_world/flip_place.py) | Same, **after** the policy's ~180° flip. Cue = right wrist-roll ≥ 2.5 rad since the grab. |
| [grasp_recovery.py](real_world/grasp_recovery.py) | Wrist YOLO (`open`/`closed-gripped`/`closed-empty`). On a failed grasp: clear queue, open, retreat, let the policy re-plan. |
| [retreat.py](real_world/retreat.py) | Torch-free, velocity-bounded retreat primitive shared by recovery and the unreachable-target handler. |

The place macros work **in joint space throughout** (a fixed end joint config guarantees the
release point via FK with no IK, avoiding redundancy branch-flips); the **start adapts, the end
is fixed** via a decaying-offset warp; and **everything is cleared before and after** so nothing
snaps when auto resumes.

Detection runs **locally on the client**, not on the policy server. The **Detector tuning** GUI
tab shows a live boxed head-cam view, backed by the same `YoloGate` the robot uses, so a
confidence change applies to the running robot immediately.

---

## Key design decisions & considerations

- **Hybrid policy + scripted macros** — learn the hard part, script the repeatable part.
- **Transparent IK over the SDK black-box** — a black box permits no pre-execution trajectory
  validation, which makes every safety guarantee hollow. The same Pinocchio solver runs in the
  PyBullet check and on the robot.
- **EE-space, not joint-space** — on the diffusion-policy authors' recommendation; joint output
  was trained and tested and performed worse.
- **6D rotation output, quaternion input** — 6D rotation is continuous (quaternions
  double-cover and are discontinuous), which matters enormously for regression/training
  stability; quaternions are compact and fine as an *input*.
- **Absolute master-ID alignment + buffered Gaussian smoothing** — replaced explicit
  time/nearest-state trajectory merging, the consistent source of motion oscillation.
- **On-demand camera hub** — depth-camera bandwidth (~600 MB/s) saturated DDS and froze other
  streams; cameras now subscribe only while a consumer needs them.
- **Known limitation** — the diffusion policy currently requires a **fixed head-camera
  position**, as in the original fixed-platform paper.

---

## Data & training — the lessons that matter most

Model quality is dominated by **operator consistency during teleoperation**, none of which is
captured in the code:

- **Exaggerate movements.** Subtle motions do not survive training; moves that feel too large
  during teleop reproduce correctly on the robot.
- **Be explicit about intent.** An end-to-end vision model infers no implicit intent — it learns
  only patterns it sees repeated.
- **Keep approach margins consistent.** If the head cam sees the parcel spanning x1–x2, always
  approach at x1−n / x2+n for a fixed n (~5 cm). Inconsistency here was the **single largest
  source of grasp failures**, especially for parcels low and close to the table.

Practicalities: recordings at 30 Hz are built into a **10 Hz** dataset (every 3rd frame), which
fixed the robot stalling between inference updates. ~50 demos at 50–100 epochs already replicate
relative motion; ~100–200 demos capture fine idiosyncrasies; **diminishing returns set in above
~100 epochs** (a 200-epoch run showed no visible gain). The current dataset is 200 episodes.

---

## SDK-free split

The SDK (`a2d_sdk`) and ROS live **only on the robot machine**. Kinematics, sim, camera hub,
recorded-obs playback, and timing are written to import without them — via lazy package imports
and dependency injection — so the whole sim/eval/CI path (and `--demo` mode) runs on any laptop,
roughly halving iteration time. [requirements_sim.txt](requirements_sim.txt) is that SDK-free
dependency set. **Preserve this separation.**

---

## Repository layout

| Path | What it is |
|------|------------|
| [robot_control_gui.py](robot_control_gui.py) | Main entry point — the Tkinter console; wires up robot / camera / env / inference. |
| [real_world/](real_world/) | Core library — env, IK, inference controller, post-processing, sim backend, timing, data building, and the manipulation-pipeline modules. |
| [gui/](gui/) | GUI feature mixins (Camera, Inference, DetectorTuning, VR, DataCollection, Eval) + the hardware-free [demo_backend.py](gui/demo_backend.py). |
| [pico_vr/](pico_vr/) | Pico VR teleop client/server and shared wire protocol. |
| [servers/](servers/) | Networking glue: robot-info HTTP server, inference-server ping, recording upload. |
| [MDM_data_collection/](MDM_data_collection/) | Data-collection GUI + dataset builder: recorded episodes → replay-buffer `.zarr`. |
| [scripts/](scripts/) | Eval, diagnostics, and path/waypoint builders (sim inference eval, FK-consistency check, …). |
| [tests/](tests/) | Pytest suite — safety invariants, append/splice, retreat waypoints, tagging. |
| [planning/](planning/) | Reachability, detection, and planning server subsystem. |
| [examples/](examples/) | Standalone reference scripts (e.g. wheel/base control). |
| [data/](data/), [real_world/assets/](real_world/assets/) | Local YOLO `.pt` weights, recorded release paths, vendored URDF. |
| [a2d_sdk/](a2d_sdk/) | Vendored A2D robot SDK (runtime dependency, imported as `a2d_sdk.robot`). |
| [docs/](docs/) | Vendor manuals and product references. |
| [documenation/](documenation/) | **Knowhow, PoC report, and slide deck** (EN + 中文). Start here for depth. |

> `G1_SDK_ENV/` is a legacy SDK snapshot kept on disk but excluded from version control.

---

## Getting started

Run everything from the repository root so the top-level packages (`real_world`, `gui`,
`servers`, `a2d_sdk`, …) resolve.

```bash
# On the robot machine — full stack (needs a2d_sdk + ROS + the robot):
pip install -r requirements_gui.txt      # + `pip install ultralytics` for the YOLO gates
python robot_control_gui.py

# On any laptop — hardware-free DEMO mode (synthetic robot, live camera, no SDK/ROS/robot):
python robot_control_gui.py --demo

# SDK-free sim / IK tools (Pinocchio + PyBullet):
python -m venv .venv && .venv/bin/pip install -r requirements_sim.txt
.venv/bin/python scripts/sim_infer_eval.py    # run the policy against a recording in sim
```

The console has four tabs — **Console** (camera views + inference: sim preview → auto-run;
auto-run refuses to start without the sim preview, for safety), **Detector tuning** (live boxed
head-cam for tuning the YOLO gates), **VR teleop** (Pico teleop + recording), and **Evaluation**
(live success-rate dashboard fed by `infer_logs/eval/*.jsonl`).

### Testing

```bash
pytest tests/    # safety invariants, append/splice continuity, retreat waypoints, tagging
```

The test suite also runs automatically as a **launch pre-flight**; a safety-invariant regression
blocks the GUI from starting (`HUMANOID_SKIP_SAFETY_PREFLIGHT=1` bypasses it, for exceptional
cases only).

### Runtime state

- [tuning_config.json](tuning_config.json) holds live operator tuning (temporal-ensemble params,
  `speed_scale`, `append_ahead_rows`), loaded once at GUI startup; code constants are the
  defaults. Delete it to reset.
- Calibration lives in [real_world/config/](real_world/config/) — `fk_calibration*.json`,
  `nominal_arm_config.json`, `retreat_waypoints.json`; must match the deployment torso pose.
- `live_joints.jsonl`, `released_substeps.jsonl`, recordings, `.zarr` buffers, and
  `infer_logs/eval/` are runtime scratch/data and are git-ignored.
