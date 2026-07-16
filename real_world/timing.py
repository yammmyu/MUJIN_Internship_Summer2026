"""Single source of timing truth for the inference -> robot execution path (SDK-free).

Everything derives from two knobs:

  * RECORD_HZ  — the rate the policy was trained / the data was recorded at. This is the
                 cadence of the policy's action *rows*: consecutive rows are meant to be
                 ROW_DT = 1/RECORD_HZ apart in wall-clock, and that spacing IS the intended
                 motion speed, so it must be preserved on execution.
  * CONTROL_HZ — the rate we stream *substep* waypoints to the arm controller. Each policy
                 row gap is filled with SUBSTEPS_PER_ROW = CONTROL_HZ/RECORD_HZ time-uniform
                 substeps, drained one per STEP_TIME.

Why this module exists — the constants used to be defined in three places (humanoid_env,
inference_controller, robot_data_collect) and had drifted out of sync, and execution timing
was decoupled from the recorded speed: each row gap was subdivided in JOINT space
(ceil(|dq|/MAX_JOINT_STEP) substeps) but drained in TIME (one substep per STEP_TIME), so the
wall-clock to execute a row depended on how far the joints moved (big moves played too slow,
small moves too fast). That broke the auto-splice merges, whose "now" index
f = round(elapsed/STEP_TIME) assumes substep i lands at obs_ts + i*STEP_TIME.

The fix (encoded by these constants): each row expands to exactly SUBSTEPS_PER_ROW substeps
*uniform in time*, so:
  * inter-row wall-clock is ALWAYS ROW_DT, independent of motion magnitude (faithful speed), and
  * smoothness is governed purely by CONTROL_HZ (more substeps per row -> finer motion).

MAX_JOINT_STEP is then a SAFETY ceiling only (not a smoothness/timing knob): a row whose
per-substep delta exceeds it — equivalently, whose joint velocity exceeds MAX_JOINT_VEL — is
rejected by validation (the previous trajectory keeps draining).
"""

# --- the knobs ---------------------------------------------------------------------------
# Fraction of recorded speed to execute at. 1.0 = demo speed; <1 = slower (each row spans more
# substeps, so a chunk lasts longer); >1 = faster. Lower it so one chunk lasts ~one inference
# period — otherwise the arm drains a chunk faster than the next inference arrives and stutters.
# Alignment stays correct at any value because it is keyed on the master row-ID (the robot's own
# execution clock), not wall-clock.
SPEED_SCALE = 0.2       # [persisted: tuning_config.json] DEFAULT SEED ONLY. The GUI restores the saved
                        # value at startup and tunes it live (HumanoidEnv.set_speed_scale); the live
                        # runtime value is env.speed_scale (-> env.substeps_per_row / ramp_joint_step).
                        # Editing this literal only changes the fallback when there is no saved JSON
                        # entry (e.g. headless). TUNE (0,1+]: fraction of demo speed. LOWER -> more
                        # substeps/row (smoother + slower, chunk lasts longer so the append buffer
                        # starves less); >1 = faster. Alignment stays correct at any value (row-id keyed).
RECORD_HZ = 10          # policy action-row cadence (training / recording rate), Hz. NOT a free knob:
                        # fixed by how the policy was trained; changing it desyncs row timing.
CONTROL_HZ = 120        # TUNE: substep streaming rate to the arm controller, Hz. Primary motion-
                        # RESOLUTION knob: higher -> more substeps/row -> finer motion. Raise toward
                        # the SDK's sustainable ABS_JOINT waypoint rate (verify on hardware). MUST be
                        # an integer multiple of RECORD_HZ.

assert CONTROL_HZ % RECORD_HZ == 0, (
    f"CONTROL_HZ ({CONTROL_HZ}) must be an integer multiple of RECORD_HZ ({RECORD_HZ}) so each "
    f"policy row maps to a whole number of substeps")

# --- derived: timing -------------------------------------------------------------------------
ROW_DT = 1.0 / RECORD_HZ                          # wall-clock between policy rows (s).
STEP_TIME = 1.0 / CONTROL_HZ                       # wall-clock per substep (s); release-loop tick.
SUBSTEPS_PER_ROW = round((CONTROL_HZ / RECORD_HZ)/SPEED_SCALE)   # time-uniform substeps per row gap (= K).
# ^ DERIVED (do not set directly): motion granularity per row = K. Tune it via CONTROL_HZ (up) or
#   SPEED_SCALE (down). Higher K -> finer/smoother per-row motion.

# --- derived: safety velocity ceiling --------------------------------------------------------
# Genuine per-joint velocity ceiling for the LEFT arm. Measured across the recordings: real demo
# motion peaks ~5 rad/s (p99.9 ~2.7 rad/s); the rare ~17 rad/s spikes are sensor glitches. 6 rad/s
# clears all real motion with margin while rejecting the glitches. A policy row faster than this
# is refused by validation. This is a SAFETY cap only — it does not set motion smoothness.
MAX_JOINT_VEL = 4.0                                # rad/s (~344 deg/s).
MAX_JOINT_STEP = min(MAX_JOINT_VEL / CONTROL_HZ, 0.05)       # max per-substep joint delta (rad); the C5 cap.

# --- derived: large-rotation watchdog (C6) ---------------------------------------------------
# A HARD cutoff on a SINGLE commanded joint jump (rad) at dispatch. MAX_JOINT_STEP above bounds
# per-substep VELOCITY — a big jump becomes a slow ramp, so a bad IK branch-flip / jumpy policy row
# / queue seam is still executed, just slowly (the "arm swings through a huge rotation" symptom).
# This bounds per-command DISPLACEMENT instead: a waypoint more than WATCHDOG_MAX_JOINT_JUMP from
# the last commanded config is a discontinuity, not motion, so the release path REFUSES it (latches
# the E-stop) rather than ramping it through. Set FAR above MAX_JOINT_STEP (~0.033) so legitimate
# velocity-bounded motion and pre-subdivided ramps never trip it, and below a dangerous single-joint
# rotation. TUNE on hardware: lower = stricter cutoff (more sensitive), 0 disables the watchdog.
WATCHDOG_MAX_JOINT_JUMP = 0.5                       # rad (~29 deg); per-command joint-jump cutoff.

# --- derived: ramp cruise speed --------------------------------------------------------------
# Ramps/bridges (the initial C5 ramp-in AND the per-chunk seam bridge in append_actions) connect
# the arm's current pose to the next action row. They are NOT trajectory-following motion, so the
# SPEED_SCALE that is baked into SUBSTEPS_PER_ROW never reaches them: subdividing a ramp by
# MAX_JOINT_STEP moves it at the full SAFETY-cap velocity — visibly ~1/SPEED_SCALE faster than
# normal rows. That is the "arm darts fast for one action between chunks, then slows" spike. Scaling
# the ramp's per-substep delta by SPEED_SCALE makes a bridge cruise at the same speed as ordinary
# motion. It is <= MAX_JOINT_STEP, so the C5 safety bound is untouched (ramps only ever get SLOWER).
RAMP_JOINT_STEP = min(MAX_JOINT_STEP, MAX_JOINT_STEP * SPEED_SCALE)   # per-substep joint delta for
# ramps (rad). Clamped to MAX_JOINT_STEP so SPEED_SCALE > 1 (faster than demo) can never push a ramp
# ABOVE the C5 safety cap — auto-path ramps go straight to the queue without a _subdivide_points re-clamp.
# ^ DERIVED: sets how fast the seam bridge cruises (smoothness of the chunk-to-chunk seam). Lower
#   multiplier -> slower, smoother ramp; always <= MAX_JOINT_STEP so the C5 safety bound is untouched.

# --- trace output location -------------------------------------------------------------------
# One folder that ALL inference/release JSONL traces are written to, so a run's logs stay together
# instead of scattering into the process CWD: buffer/requests/chunks (inference controller) and
# released_substeps/live_joints (env release loop). Relative to CWD by default; override with the
# HUMANOID_TRACE_DIR env var. Callers mkdir(parents=True, exist_ok=True) once before opening files.
import os as _os
import pathlib as _pathlib
TRACE_DIR = _pathlib.Path(_os.environ.get("HUMANOID_TRACE_DIR", "infer_logs"))
