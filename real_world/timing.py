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
SPEED_SCALE = 0.2
RECORD_HZ = 30          # policy action-row cadence (training / recording rate), Hz.
CONTROL_HZ = 120        # substep streaming rate to the arm controller, Hz. Smoothness knob:
                        # raise toward the SDK's sustainable ABS_JOINT waypoint rate (verify on
                        # hardware) for finer motion. MUST be an integer multiple of RECORD_HZ.

assert CONTROL_HZ % RECORD_HZ == 0, (
    f"CONTROL_HZ ({CONTROL_HZ}) must be an integer multiple of RECORD_HZ ({RECORD_HZ}) so each "
    f"policy row maps to a whole number of substeps")

# --- derived: timing -------------------------------------------------------------------------
ROW_DT = 1.0 / RECORD_HZ                          # wall-clock between policy rows (s).
STEP_TIME = 1.0 / CONTROL_HZ                       # wall-clock per substep (s); release-loop tick.
SUBSTEPS_PER_ROW = round((CONTROL_HZ / RECORD_HZ)/SPEED_SCALE)   # time-uniform substeps per row gap (= K).

# --- derived: safety velocity ceiling --------------------------------------------------------
# Genuine per-joint velocity ceiling for the LEFT arm. Measured across the recordings: real demo
# motion peaks ~5 rad/s (p99.9 ~2.7 rad/s); the rare ~17 rad/s spikes are sensor glitches. 6 rad/s
# clears all real motion with margin while rejecting the glitches. A policy row faster than this
# is refused by validation. This is a SAFETY cap only — it does not set motion smoothness.
MAX_JOINT_VEL = 4.0                                # rad/s (~344 deg/s).
MAX_JOINT_STEP = min(MAX_JOINT_VEL / CONTROL_HZ, 0.05)       # max per-substep joint delta (rad); the C5 cap.
