# Humanoid

Teleoperation, data collection, and policy-inference control stack for the
AgiBot / 智元 **A2D** dual-arm humanoid, built on the vendored **a2d_sdk** robot SDK.

## Layout

| Path | What it is |
|------|------------|
| [robot_control_gui.py](robot_control_gui.py) | Main entry point — Tkinter GUI assembling camera view, teleop, data collection, and inference. |
| [real_world/](real_world/) | Core library: robot env, IK, inference controller, timing, sim backend, data building. |
| [gui/](gui/) | GUI mixins (`StyleMixin`, `CameraMixin`, `InferenceMixin`, `VRMixin`, `DataCollectionMixin`). |
| [pico_vr/](pico_vr/) | Pico VR teleop client/server and shared wire protocol. |
| [servers/](servers/) | Networking glue: robot-info HTTP server, inference-server ping, recording upload. |
| [examples/](examples/) | Standalone reference scripts (e.g. wheel/base control). |
| [scripts/](scripts/) | Eval, diagnostics, and safety-invariant test scripts. |
| [MDM_data_collection/](MDM_data_collection/) | Data-collection GUI and dataset builder (recorded data is not versioned). |
| [planning/](planning/) | Reachability, detection, and planning server subsystem. |
| [a2d_sdk/](a2d_sdk/) | Vendored A2D robot SDK (runtime dependency, imported as `a2d_sdk.robot`). |
| [docs/](docs/) | Vendor manuals and product references. |

> `G1_SDK_ENV/` is a legacy SDK snapshot kept on disk but excluded from version
> control (see `.gitignore`).

## Running

Run from the repository root so top-level packages (`real_world`, `gui`,
`servers`, `examples`, `a2d_sdk`, …) resolve:

```bash
# GUI dependencies
pip install -r requirements_gui.txt   # add requirements_sim.txt for the sim backend

# Launch the control GUI (requires the robot + a2d SDK + ROS)
python robot_control_gui.py

# Hardware-free DEMO mode — synthetic robot, live camera feeds, and a filling
# evaluation dashboard. Runs on any laptop with Tk; no SDK/ROS/robot needed.
# Use it for recorded demos, screenshots, and UI work.
python robot_control_gui.py --demo
```

The console is organized as three tabs — **Console** (camera views + policy
inference: sim preview, validate, release, substep monitor), **VR teleop**
(Pico teleoperation + data collection), and **Evaluation** (a live success-rate
KPI dashboard fed by `infer_logs/eval/*.jsonl`, the same logs
[scripts/eval_trials.py](scripts/eval_trials.py) writes). The UI theme and all
ttk styles live in [gui/styles.py](gui/styles.py); the demo stand-ins live in
[gui/demo_backend.py](gui/demo_backend.py).

## Notes

- Timing is single-sourced in [real_world/timing.py](real_world/timing.py) — import
  `RECORD_HZ` / `CONTROL_HZ` from there rather than redefining them.
- `live_joints.jsonl` is a runtime scratch log (git-ignored).
