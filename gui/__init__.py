"""GUI 功能 Mixin 集合。

RobotControlGUI 通过多继承组合下列 Mixin，每个 Mixin 聚焦一个功能域，
仅依赖主类在 __init__ 中初始化的共享状态（self.robot / self.env /
self.inference / self.camera_images / 各 Tk 变量等）。相机统一经 self.env 按需订阅。
"""

from gui.styles import StyleMixin
from gui.camera_panel import CameraMixin
from gui.inference_panel import InferenceMixin
from gui.detector_tuning import DetectorTuningMixin
from gui.vr_control import VRMixin
from gui.data_collection import DataCollectionMixin
from gui.eval_panel import EvalMixin

__all__ = [
    "StyleMixin",
    "CameraMixin",
    "InferenceMixin",
    "DetectorTuningMixin",
    "VRMixin",
    "DataCollectionMixin",
    "EvalMixin",
]
