"""GUI 功能 Mixin 集合。

RobotControlGUI 通过多继承组合下列 Mixin，每个 Mixin 聚焦一个功能域，
仅依赖主类在 __init__ 中初始化的共享状态（self.robot / self.camera /
self.camera_images / 各 Tk 变量等）。
"""

from gui.styles import StyleMixin
from gui.camera_panel import CameraMixin
from gui.status_panel import StatusMixin
from gui.coordinates import CoordinateMixin
from gui.pick_place import PickPlaceMixin
from gui.inference import InferenceMixin
from gui.manual_control import ManualControlMixin
from gui.vr_control import VRMixin

__all__ = [
    "StyleMixin",
    "CameraMixin",
    "StatusMixin",
    "CoordinateMixin",
    "PickPlaceMixin",
    "InferenceMixin",
    "ManualControlMixin",
    "VRMixin",
]
