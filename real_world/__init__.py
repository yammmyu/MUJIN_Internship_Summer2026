"""real_world package.

`ik` is SDK-free and imported eagerly so the offline FK probe and the PyBullet sim
runner can use it without `a2d_sdk` present. The SDK-dependent names
(HumanoidEnv / InferenceController / rot6d_to_quat / ...) are imported lazily via
module __getattr__, so `from real_world.ik import ...` (or importing the package on a
machine without the SDK) does not pull in `a2d_sdk`.
"""

from real_world.ik import (
    IKQuery, IKSolver, PinocchioArmModel, PinocchioDLSIKSolver,
    FKCalibration, load_calibration, build_solver,
)

__all__ = [
    # SDK-free (eager)
    "IKQuery", "IKSolver", "PinocchioArmModel", "PinocchioDLSIKSolver",
    "FKCalibration", "load_calibration", "build_solver",
    # SDK-dependent (lazy)
    "HumanoidEnv", "rot6d_to_quat", "INFERENCE_CAMERAS",
    "InferenceController", "encode_image", "post_predict",
]

_LAZY = {
    "HumanoidEnv": "humanoid_env",
    "rot6d_to_quat": "humanoid_env",
    "INFERENCE_CAMERAS": "humanoid_env",
    "InferenceController": "inference_controller",
    "encode_image": "inference_controller",
    "post_predict": "inference_controller",
}


def __getattr__(name):
    mod = _LAZY.get(name)
    if mod is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    m = importlib.import_module(f"real_world.{mod}")
    return getattr(m, name)
