import dataclasses
import math
import threading
import time
import typing

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from pico_vr.pico_vr_server.server import _TRIGGER_LABELS
from pico_vr.pico_vr_common.protocol import JointState


# ---------------------------------------------------------------------- #
# 抖动抑制工具：One-Euro 滤波器 + 软死区 + 幅值钳位 + 四元数旋转差
# ---------------------------------------------------------------------- #

# One-Euro filter（Casiez 等人，2012）
# 当输入变化慢时强力低通滤波 → 抑制手部细微抖动
# 当输入变化快时自动放宽截止频率 → 保留快速动作的响应性
class OneEuroFilter:
    """Single-channel One-Euro filter. Maintains its own clock."""

    def __init__(self, mincutoff: float = 1.0, beta: float = 5.0,
                 dcutoff: float = 1.0):
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self._x_prev: typing.Optional[float] = None
        self._dx_prev: float = 0.0
        self._t_prev: typing.Optional[float] = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self) -> None:
        self._x_prev = None
        self._dx_prev = 0.0
        self._t_prev = None

    def __call__(self, x: float, t: float) -> float:
        if self._t_prev is None:
            self._t_prev = t
            self._x_prev = x
            return x
        dt = max(t - self._t_prev, 1e-4)
        # 平滑的导数估计
        dx = (x - self._x_prev) / dt
        a_d = self._alpha(self.dcutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev
        # 动态截止频率：动作越快，截止越高 → 跟得上
        cutoff = self.mincutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self._x_prev
        self._x_prev = x_hat
        self._dx_prev = dx_hat
        self._t_prev = t
        return x_hat


def _soft_deadzone_vec(v: np.ndarray, dead: float) -> np.ndarray:
    """对向量按"幅值"做线性软死区：|v|<=dead → 0；否则 (|v|-dead) 沿方向。

    避免硬阈值在边界附近产生的来回抖动，且不改变运动方向。
    """
    m = float(np.linalg.norm(v))
    if m <= dead or m == 0.0:
        return np.zeros_like(v)
    return v * ((m - dead) / m)


def _clamp_magnitude(v: np.ndarray, max_m: float) -> np.ndarray:
    """按整体幅值钳位（保持方向），而不是各轴独立钳位（会扭曲方向）。"""
    m = float(np.linalg.norm(v))
    if m > max_m and m > 0.0:
        return v * (max_m / m)
    return v


def _stick_expo(value: float, deadzone: float, expo: float) -> float:
    """把原始摇杆轴 [-1,1] 整形为指令系数 [-1,1]（速率控制用）。

    先去中心死区(滤掉松手回中漂移)，把剩余行程重标定到 0..1，再做 expo 混合
    （线性↔三次方）：中心附近更细腻、推到边缘更快——这正是「精细滚转」的关键。
    """
    a = abs(float(value))
    if a <= deadzone:
        return 0.0
    a = (a - deadzone) / (1.0 - deadzone)        # 剩余行程重标定到 0..1
    a = (1.0 - expo) * a + expo * a ** 3          # expo 混合
    return math.copysign(a, value)


# 腕部滚转写入的旋转分量下标（6 维 delta 的旋转槽，0 基于整条向量）。
# get_action_delta 把旋转排成 [index3=-rv[1]=roll, index4=rv[0]=pitch, index5=rv[2]=yaw]，
# 故 roll = index 3。若上机后发现摇杆实际驱动的不是滚转，改成 4 或 5 即可；
# 方向不对则把 VR_PARAMS.wrist_roll_rate 取负（或反转摇杆轴）。
_WRIST_ROLL_SLOT = 3

# 腕部滚转所对应的手臂关节（J7 法兰滚转）在 arm_joint_states()(14,) 中的下标：
# 左臂 0–6、右臂 7–13，故 J7 = 左 6 / 右 13。硬限位即钳制这个关节的绝对角。
_WRIST_ROLL_JOINT_IDX = {"left": 6, "right": 13}
# 正向滚转「指令」是否使该关节角增大。上机若发现限位方向相反，改成 -1.0 即可。
_WRIST_ROLL_JOINT_SIGN = 1.0


def _quat_mult_xyzw(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product, scipy convention (x, y, z, w)."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ])


def _rotvec_delta(q_prev_xyzw, q_curr_xyzw) -> np.ndarray:
    """从 q_prev 到 q_curr 的小角度旋转，以 rotvec 表示。

    用四元数差代替欧拉角差，避免奇点与累计误差。
    """
    qx, qy, qz, qw = q_prev_xyzw
    q_prev_conj = np.array([-qx, -qy, -qz, qw])
    q_d = _quat_mult_xyzw(np.asarray(q_curr_xyzw, dtype=float), q_prev_conj)
    # 保证走"短路径"
    if q_d[3] < 0.0:
        q_d = -q_d
    return R.from_quat(q_d).as_rotvec()


# ---------------------------------------------------------------------- #
# 遥操作灵敏度 / 平滑 / 限幅参数
# ---------------------------------------------------------------------- #
@dataclasses.dataclass(frozen=True)
class VRParameters:
    """VR 遥操作的全部可调参数，集中于此便于统一调参。

    通用约定
    --------
    * 单位：位置 = 米 (m)，姿态/旋转 = 弧度 (rad)，时间 = 秒 (s)。
    * 凡是 ``*_max_*`` / ``*_clamp`` 量，都是「**每个执行周期** (= execution_period)
      内允许下发的最大增量」，相当于**速度上限**。换算速度 = 该值 / execution_period。
      值越大 → 越跟手、越激进；越小 → 越稳、越钝。
    * One-Euro 滤波三参数 (mincutoff / beta / dcutoff) 决定「抖动抑制 ↔ 跟手性」折中。
    * ``*_deadzone`` 软死区：单拍增量模长小于此值视为手部抖动 → 直接清零；
      作用于模长而非各轴，所以不改变运动方向。
    """

    # ===================== One-Euro 滤波：位置通道 (px, py, pz) =====================
    # mincutoff: 手部静止时的低通截止频率 (Hz)。越低 → 静止越稳，但慢动作越钝/越滞后。
    #            1.0Hz 足以压住 4-12Hz 的生理性手抖。
    pos_mincutoff: float = 0.3
    # beta: 速度自适应增益。越大 → 动作越快时截止频率放得越开 → 越跟手，但也放进更多抖动。
    pos_beta: float = 20.0
    # dcutoff: 对「速度估计值」本身的低通截止 (Hz)，通常固定 1.0，无需调。
    pos_dcutoff: float = 1.0

    # ===================== One-Euro 滤波：四元数通道 (qx, qy, qz, qw) =====================
    # 姿态比位置更易抖，静止时用更激进的 0.6Hz 压住手腕微颤；转得快时靠大 beta 立即放开。
    quat_mincutoff: float = 0.6
    quat_beta: float = 20.0
    quat_dcutoff: float = 1.0

    # ===================== 单拍增量计算 (Position.get_action_delta) =====================
    # pose_mult: 平移整体增益（线性缩放手→臂的位移）。1.0 = 1:1 跟手；<1 更迟钝、更稳。
    pose_mult: float = 1.0
    # qut_mult: 旋转整体增益（线性缩放手→臂的转动）。1.0 = 1:1 跟手；0.5 = 手腕转动量的一半映射到机械臂。
    #           想整体提高/降低旋转灵敏度优先调这个。
    qut_mult: float = 1.0
    # pos_deadzone: 位置软死区 (m)。单拍位移模长 < 1mm 视为抖动 → 清零。
    pos_deadzone: float = 0.001
    # pos_max_delta: 单拍最大位移 (m)，超出按模长钳回(保方向)。0.04m@50Hz ≈ 2.0 m/s 速度上限。
    pos_max_delta: float = 0.04
    # quat_deadzone: 姿态软死区 (rad)。单拍旋转模长 < 0.005rad(≈0.3°) 视为抖动 → 清零。
    quat_deadzone: float = 0.005
    # quat_max_delta: 单拍最大旋转 (rad)，超出按模长钳回。0.04rad@50Hz ≈ 114°/s 速度上限。
    quat_max_delta: float = 0.04
    # pitch_gain: 手腕「俯仰 (pitch)」轴的额外放大倍数（在通用 qut_mult 之上单独加成）。
    #             1.0 = 不额外放大。注意：此增益施加在「单拍计算」阶段，若 > 1，多数情况下
    #             仍会被下游 merge_rot_clamp 的合并模长钳位抵消 → 想真正提升 pitch 灵敏度，
    #             需同步抬高 merge_rot_clamp，或改为在合并钳位「之后」单独放大 pitch 轴。
    pitch_gain: float = 1.0

    # ===================== 队列合并后再钳位 (_rebuild_arm_action) =====================
    # 一个执行周期内可能累加了多拍回调动作，合并后再按模长钳一次，防止积压越界。
    # 这是作用于机械臂的**最终速度上限**，通常应 >= 上面的单拍 *_max_delta。
    merge_pos_clamp: float = 0.06   # 合并后位移模长上限 (m)
    merge_rot_clamp: float = 0.04   # 合并后旋转模长上限 (rad)

    # ===================== 头部绝对目标限幅 =====================
    # 头部按「绝对目标角」下发（非增量）；以下为偏航/俯仰的行程上下限，保护舵机。
    head_yaw_limit: float = 1.0     # 偏航 ±1.0 rad (≈±57°)
    head_pitch_limit: float = 0.5   # 俯仰 ±0.5 rad (≈±29°)

    # ===================== 执行线程节拍 / 队列 =====================
    # execution_period: 执行周期 (s)，与 SDK time_step 一致 = 50Hz。
    #                   它是上面所有「单拍」量换算成速度的时间基准，改它会同时改变所有速度上限。
    execution_period: float = 0.02
    # queue_max: 动作队列上限。积压超过此值丢弃最旧动作，保证「最新意图优先」、避免长按滞后。
    queue_max: int = 30

    # ===================== 摇杆腕部滚转（速率控制）=====================
    # 左/右摇杆 X 轴 → 对应手臂腕部「滚转速率」。与手部姿态、与 L_Y/R_B 移动闸门均解耦：
    # 不必按住闸门，摇杆推到底即以 wrist_roll_rate(rad/执行周期) 持续滚转，回中即停。
    # 适合连续/精细的腕部滚转——手腕物理活动范围有限，靠姿态映射很难做到。
    # 在 _vr_auto_execution_thread 的 50Hz 节拍里积分（手不动也会持续转）。
    # wrist_roll_rate: 满量程每拍滚转增量(rad)。×50Hz ⇒ 角速度上限，0.02≈1.0rad/s≈57°/s。
    wrist_roll_rate: float = 0.02
    # wrist_roll_deadzone: 摇杆中心死区(0..1)，滤除松手时的回中漂移。
    wrist_roll_deadzone: float = 0.15
    # wrist_roll_expo: 0=线性，1=纯三次方；越大中心越细腻、边缘越快。
    wrist_roll_expo: float = 0.6
    # 腕部滚转「硬限位」：针对滚转关节 J7 的绝对角(弧度)。摇杆滚转每拍被钳到使 J7
    # 落在 [lo, hi] 内——到限即停、绝不越界（非渐进）。需按实际关节零位标定。
    wrist_roll_limit_lo: float = -1.57
    wrist_roll_limit_hi: float = 1.57


# 全局默认参数实例。需要现场调参时修改此处或替换该实例即可，所有引用点统一生效。
# 运行时也可由 GUI 的 VR 参数面板通过 `dataclasses.replace` 原子替换该全局
# 实例(见 VRMixin._vrp_apply)，所有「调用时读 VR_PARAMS.x」的引用点立即生效。
VR_PARAMS = VRParameters()

# 出厂默认的不可变快照，供面板「重置默认」使用(VR_PARAMS 会被实时替换)。
VR_PARAMS_DEFAULTS = VRParameters()


# ---------------------------------------------------------------------- #
# GUI 可调参数清单：(attr, 中文标签, 下限, 上限, 步进, 是否整数, 分组)
# 下限/上限即「安全上下限」，面板滑块与输入框都会强制钳到此区间。
# pose_mult / qut_mult 下限 > 0：get_action_delta 里有 assert 必须为正。
# ---------------------------------------------------------------------- #
VR_PARAM_SPECS = [
    # 平移 / 旋转整体增益
    ("pose_mult",       "平移增益",       0.10, 3.00, 0.05, False, "灵敏度增益"),
    ("qut_mult",        "旋转增益",       0.10, 3.00, 0.05, False, "灵敏度增益"),
    ("pitch_gain",      "俯仰额外增益",    0.50, 5.00, 0.10, False, "灵敏度增益"),
    # 单拍增量上限(= 速度上限)
    ("pos_max_delta",   "单拍位移上限(m)",  0.002, 0.100, 0.001, False, "速度上限(单拍)"),
    ("quat_max_delta",  "单拍旋转上限(rad)", 0.002, 0.200, 0.001, False, "速度上限(单拍)"),
    # 合并后再钳位(最终速度上限)
    ("merge_pos_clamp", "合并位移上限(m)",  0.002, 0.100, 0.001, False, "速度上限(合并)"),
    ("merge_rot_clamp", "合并旋转上限(rad)", 0.002, 0.200, 0.001, False, "速度上限(合并)"),
    # 软死区(抖动抑制)
    ("pos_deadzone",    "位移死区(m)",     0.0000, 0.0200, 0.0005, False, "死区"),
    ("quat_deadzone",   "旋转死区(rad)",    0.0000, 0.0500, 0.0005, False, "死区"),
    # One-Euro 滤波(跟手性 ↔ 抖动抑制)
    ("pos_mincutoff",   "位置静止截止(Hz)", 0.10, 10.00, 0.10, False, "One-Euro 滤波"),
    ("pos_beta",        "位置速度增益β",    0.00, 50.00, 0.50, False, "One-Euro 滤波"),
    ("quat_mincutoff",  "姿态静止截止(Hz)", 0.10, 10.00, 0.10, False, "One-Euro 滤波"),
    ("quat_beta",       "姿态速度增益β",    0.00, 60.00, 0.50, False, "One-Euro 滤波"),
    # 头部行程限位(保护舵机)
    ("head_yaw_limit",  "头部偏航限位(rad)", 0.00, 1.57, 0.01, False, "头部限位"),
    ("head_pitch_limit", "头部俯仰限位(rad)", 0.00, 1.00, 0.01, False, "头部限位"),
    # 摇杆腕部滚转(速率控制)
    ("wrist_roll_rate",     "滚转速率(rad/拍)", 0.000, 0.060, 0.002, False, "腕部滚转(摇杆)"),
    ("wrist_roll_deadzone", "摇杆死区",        0.000, 0.500, 0.010, False, "腕部滚转(摇杆)"),
    ("wrist_roll_expo",     "expo 曲线",       0.000, 1.000, 0.050, False, "腕部滚转(摇杆)"),
    ("wrist_roll_limit_lo", "滚转下限(rad)",   -3.140, 3.140, 0.010, False, "腕部滚转(摇杆)"),
    ("wrist_roll_limit_hi", "滚转上限(rad)",   -3.140, 3.140, 0.010, False, "腕部滚转(摇杆)"),
    # 动作队列
    ("queue_max",       "动作队列上限",     1, 100, 1, True, "队列"),
]

# 改这些会更新 One-Euro 滤波器；需同步推到已创建的滤波器实例上才能立即生效。
_VRP_FILTER_ATTRS = frozenset({
    "pos_mincutoff", "pos_beta", "pos_dcutoff",
    "quat_mincutoff", "quat_beta", "quat_dcutoff",
})

# VR 心跳超过此值(秒)未刷新 → 面板视为「未连接」。与 DataCollectionMixin
# 的 DC_VR_STALE_TIMEOUT_S 取同一量级:手柄一旦掉线很快就反映为未连接。
VRC_CONN_TIMEOUT_S = 1.0

# 预设移动的执行参数:get_smooth_paths 的单关节单步上限(rad) + run_trajectory
# 每航点执行时间(s)。越小越慢越稳。
VRC_PRESET_SMOOTH_STEP = 0.005
VRC_PRESET_DT = 0.01

# ---------------------------------------------------------------------- #
# 单臂「一键回到」预设位姿。**每条 = 面板上一个按钮**;后续要加预设,直接往
# 这个列表追加字典即可,面板会自动按 side 分组生成按钮,无需改 UI 代码。
#   label   : 按钮文字
#   side    : "left" / "right"
#   joints  : 7 个绝对轴值 (rad) —— 目标关节角
#   gripper : "open" / "close" / None —— 到位后顺带开/合该侧夹爪;None=不动
# ---------------------------------------------------------------------- #
VR_ARM_PRESETS = [
    {
        "label": "↩ 左手初始位",
        "side": "left",
        "joints": [0.9738845229148865, 0.5554335117340088, -0.6520541310310364,
                   -1.832319736480713, 0.8602326512336731, 0.7369483709335327,
                   1.0203189849853516],
        "gripper": "open",
    },
    {
        "label": "↩ 左手预备位",
        "side": "left",
        "joints": [0.7920729517936707, -0.7327778339385986, -0.7821090221405029,
                   -0.7740122079849243, 0.12358089536428452, 1.321889877319336,
                   0.3455798029899597],
        "gripper": "open",
    },
    {
        "label": "↩ 右手初始位",
        "side": "right",
        "joints": [-0.8415611982345581, -0.6472030878067017, 0.4974645972251892,
                   1.8316916227340698, -0.825088381767273, -0.700041651725769,
                   -1.101060152053833],
        "gripper": "open",
    },
    {
        "label": "↩ 右手预备位",
        "side": "right",
        "joints": [-0.9775315523147583, 0.5038163661956787, 0.9730992317199707,
                   0.6804278492927551, -0.08786074817180634, -1.4976462125778198,
                   0.9541834592819214],
        "gripper": "open",
    },
    # 在此追加更多预设(示例):
    # {"label": "举起左手", "side": "left",
    #  "joints": [..7 个轴值..], "gripper": None},
]

# 头部 / 腰部预设的安全上下限(到这里钳一刀,保护舵机)。
# 头部俯仰放到 ±35° 以容纳 30° 的初始位预设(手动控制里 UI 限幅是 ±0.5rad)。
VRC_HEAD_YAW_LIMIT = math.radians(60)      # 偏航 ±60°
VRC_HEAD_PITCH_LIMIT = math.radians(35)    # 俯仰 ±35°
VRC_WAIST_PITCH_LIMIT = math.radians(45)   # 腰部俯仰 ±45°
VRC_WAIST_HEIGHT_MIN = 0.0                 # 腰部升降下限 (cm)
VRC_WAIST_HEIGHT_MAX = 100.0               # 腰部升降上限 (cm)

# 头部「一键回到」预设。**每条 = 一个按钮**,后续直接追加字典即可。
#   label / yaw_deg(偏航°) / pitch_deg(俯仰°,+ 为下俯)
# 头部按绝对目标角下发: robot.move_head([yaw_rad, pitch_rad])。
VR_HEAD_PRESETS = [
    {"label": "↩ 头部初始位", "yaw_deg": 0.0, "pitch_deg": 30.0},
]

# 腰部「一键回到」预设。**每条 = 一个按钮**,后续直接追加字典即可。
#   label / pitch_deg(俯仰°,+ 为前倾) / height_cm(升降高度 cm)
# 腰部按绝对目标下发: robot.move_waist([pitch_rad, height_cm])。
VR_WAIST_PRESETS = [
    {"label": "↩ 腰部初始位", "pitch_deg": 40.0, "height_cm": 2.0},
]


class VRMixin:
    """VR 遥操：手柄 JointState 回调驱动机器人，以及相机画面 VR 串流。"""

    def _handle_vr_joints(self, state: JointState) -> None:
        @dataclasses.dataclass(frozen=True)
        class Position:
            # 机器人坐标系下的位置
            px: float
            py: float
            pz: float
            # 机器人坐标系下的四元数（scipy 约定 x, y, z, w）— 用于旋转差
            qx: float
            qy: float
            qz: float
            qw: float
            # 仅用于打印 / 头部偏航俯仰使用，已不参与姿态 delta
            yaw: float
            pitch: float
            roll: float

            def format(self) -> str:
                return (
                    f"pos=({self.px:+.3f},{self.py:+.3f},{self.pz:+.3f}) "
                    f"ypr=({math.degrees(self.yaw):+6.1f}°,{math.degrees(self.pitch):+6.1f}°,{math.degrees(self.roll):+6.1f}°)"
                )

            def get_action_delta(
                self, previous_position,
            ) -> typing.Optional[list[float]]:
                """Compute (pos3, rotvec3) delta with magnitude-based clamp +
                soft deadzone. Returns None if the resulting delta is too small
                to be worth sending.

                Position uses vector magnitude (preserves direction).
                Orientation uses quaternion difference → rotvec (correct under
                gimbal lock, no Euler wraparound).

                所有灵敏度/限幅参数取自模块级 VR_PARAMS（见 VRParameters）。
                """
                p = VR_PARAMS
                assert p.pose_mult > 0
                assert p.qut_mult > 0

                # ---- position ----
                dp = np.array([
                    self.px - previous_position.px,
                    self.py - previous_position.py,
                    self.pz - previous_position.pz,
                ]) * p.pose_mult
                dp = _soft_deadzone_vec(dp, p.pos_deadzone)
                dp = _clamp_magnitude(dp, p.pos_max_delta)

                # ---- orientation: quaternion → rotvec（小角等价旋转）----
                q_curr = np.array([self.qx, self.qy, self.qz, self.qw])
                q_prev = np.array([previous_position.qx, previous_position.qy,
                                   previous_position.qz, previous_position.qw])
                rv = _rotvec_delta(q_prev, q_curr) * p.qut_mult
                rv = _soft_deadzone_vec(rv, p.quat_deadzone)
                rv = _clamp_magnitude(rv, p.quat_max_delta)

                # Empirical: the SDK's first two orientation slots are
                # actually [-dpitch_axis, droll_axis, dyaw_axis] — i.e. slot 0
                # commands rotation around robot Y (pitch) and slot 1 around
                # robot X (roll). This is opposite to the order in the
                # `move_arm_relative` docstring (`[droll, dpitch, dyaw]`) but
                # matches observed behaviour: without the swap, rolling the
                # wrist made the arm pitch and vice-versa. Yaw (slot 2 / Z)
                # was unaffected.
                # slot4 = pitch（rv[0]），额外乘 pitch_gain 单独提升俯仰灵敏度
                action_delta = (
                    [float(v) for v in dp]
                    + [-float(rv[1]), float(rv[0] * p.pitch_gain), float(rv[2])]
                )
                if any(abs(v) > 1e-5 for v in action_delta):
                    return action_delta
                return None

        def _format_pose(slab: list[float]) -> Position:

            def transform_vr_to_robot_full(px, py, pz, qx, qy, qz, qw):
                """
                完整转换VR手柄到机械臂坐标系（位置+姿态）

                坐标系映射:
                    X_robot = -Z_vr
                    Y_robot = -X_vr
                    Z_robot = Y_vr
                """
                # ========== 位置转换 ==========
                pos_robot = np.array([-pz, -px, py])

                # ========== 姿态转换 ==========
                # 定义坐标系转换矩阵
                # VR坐标系到机械臂坐标系的旋转矩阵
                R_vr_to_robot = np.array([
                    [0, 0, -1],  # 机械臂X轴 = -VR的Z轴
                    [-1, 0, 0],  # 机械臂Y轴 = -VR的X轴
                    [0, 1, 0]  # 机械臂Z轴 = VR的Y轴
                ])

                # VR手柄的四元数 → 旋转矩阵
                q_vr = [qx, qy, qz, qw]
                R_vr_hand = R.from_quat(q_vr).as_matrix()

                # 转换到机械臂坐标系下的旋转矩阵
                # R_robot_hand = R_vr_to_robot * R_vr_hand * R_vr_to_robot^T
                R_robot_hand = R_vr_to_robot @ R_vr_hand @ R_vr_to_robot.T

                # 旋转矩阵 → 四元数
                q_robot = R.from_matrix(R_robot_hand).as_quat()

                return pos_robot, q_robot

            pos, quat = transform_vr_to_robot_full(*slab)
            # py, pz, -px, qx, qy, qz, qw = slab
            px, py, pz = pos
            qx, qy, qz, qw = quat
            sin_p = 2.0 * (qw * qx - qz * qy)
            if abs(sin_p) >= 1.0:
                pitch = math.copysign(math.pi / 2.0, sin_p)
            else:
                pitch = math.asin(sin_p)
            yaw = math.atan2(
                2.0 * (qw * qy + qx * qz),
                1.0 - 2.0 * (qx * qx + qy * qy),
            )
            roll = math.atan2(
                2.0 * (qw * qz + qx * qy),
                1.0 - 2.0 * (qz * qz + qx * qx),
            )
            return Position(
                float(px), float(py), float(pz),
                float(qx), float(qy), float(qz), float(qw),
                yaw, pitch, roll,
            )

        def _fmt_buttons(triggers: list[bool]) -> list[str]:
            """Format face-button + thumbstick-click bits (triggers[2:8]). Empty if none set."""
            pressed = [
                _TRIGGER_LABELS[i]
                for i in range(2, min(len(triggers), len(_TRIGGER_LABELS)))
                if triggers[i]
            ]
            return pressed

        def _fmt_stick(axes: list[float], offset: int) -> tuple[float, float]:
            """Format one (x, y) thumbstick reading from axes[offset:offset+2]."""
            return axes[offset], axes[offset + 1]

        def _vr_auto_execution_thread() -> None:
            """固定 50Hz 节拍：每拍把队列里所有 delta 合并成一条命令下发。

            合并的好处：(a) SDK 单次命令的 time_step 与节拍一致，底层有
            充裕的插补时间；(b) 不再以 1ms 的步长喷射上百条命令；(c) 合并
            后的总位移与各 sample 的位移之和一致 → 物理意义保持。
            """
            print("Starting vr execution thread...")
            next_tick = time.monotonic()
            while self.is_vr_control:
                now = time.time()
                if now - self.last_joint_update_timestamp > 0.2:
                    print("Stopping vr execution thread, due to VR connection time out: 200ms")
                    self.vr_actions = []
                    self.is_vr_control = False
                    break

                # ---- 把本拍内所有 pending action 合并 ----
                head_target = None  # 头部用绝对目标，取最新
                left_sum = np.zeros(6, dtype=float)
                right_sum = np.zeros(6, dtype=float)
                left_grip_changed = False
                right_grip_changed = False
                while self.vr_actions:
                    head, left_hand, right_hand, lg, rg = self.vr_actions.pop(0)
                    if head is not None:
                        head_target = head
                    if left_hand is not None:
                        left_sum += np.asarray(left_hand, dtype=float)
                    if right_hand is not None:
                        right_sum += np.asarray(right_hand, dtype=float)
                    if lg is not None:
                        left_grip_changed = True
                    if rg is not None:
                        right_grip_changed = True

                # ---- 合并后再做一次幅值钳位（防止累加越界）----
                def _rebuild_arm_action(s6: np.ndarray):
                    pos = _clamp_magnitude(s6[:3], VR_PARAMS.merge_pos_clamp)
                    rot = _clamp_magnitude(s6[3:], VR_PARAMS.merge_rot_clamp)
                    out = np.concatenate([pos, rot])
                    if float(np.linalg.norm(out)) < 1e-5:
                        return None
                    return [float(v) for v in out]

                left_action = _rebuild_arm_action(left_sum)
                right_action = _rebuild_arm_action(right_sum)

                # ---- 摇杆腕部滚转（速率，独立于 L_Y/R_B 闸门；J7 硬限位）----
                # 在 merge 限幅「之后」叠加：滚转量自身已被 wrist_roll_rate 限到每拍上限，
                # 不与手部旋转共用 merge_rot_clamp，因而可在位置冻结（未按闸门）时单独滚转。
                # 手不动且队列为空时 left/right_action 为 None，这里按需新建一个纯滚转动作。
                # 每拍读实测关节，把滚转钳到使 J7 绝对角落在 [lo, hi] 内——到限即停，绝不越界。
                axes_now = getattr(self, "vr_axes", None) or []
                p = VR_PARAMS
                try:
                    roll_joints, _ = self.robot.arm_joint_states()
                except Exception as e:
                    print(f"vr wrist-roll joint read error: {e}")
                    roll_joints = None

                def _roll_delta(axis_idx: int) -> float:
                    if len(axes_now) <= axis_idx:
                        return 0.0
                    shaped = _stick_expo(axes_now[axis_idx],
                                         p.wrist_roll_deadzone, p.wrist_roll_expo)
                    return shaped * p.wrist_roll_rate

                def _limit_roll(roll: float, arm: str) -> float:
                    """钳制滚转，使 J7 绝对角不越过 [lo, hi]（硬限位）。读不到关节则不滚转
                    —— 宁可不动也不盲目越界。J7 ≈ 滚转命令 1:1（带 _SIGN）。"""
                    if abs(roll) < 1e-6:
                        return 0.0
                    idx = _WRIST_ROLL_JOINT_IDX[arm]
                    if roll_joints is None or len(roll_joints) <= idx:
                        return 0.0
                    j = float(roll_joints[idx])
                    s = _WRIST_ROLL_JOINT_SIGN
                    # 命令域边界：使 j 恰好到达 lo / hi。j_next = j + s*roll ∈ [lo, hi]。
                    a = (p.wrist_roll_limit_lo - j) / s
                    b = (p.wrist_roll_limit_hi - j) / s
                    return max(min(roll, max(a, b)), min(a, b))

                def _apply_roll(action, roll: float):
                    if abs(roll) < 1e-6:
                        return action
                    if action is None:
                        action = [0.0] * 6
                    action[_WRIST_ROLL_SLOT] += roll
                    return action

                left_action = _apply_roll(left_action, _limit_roll(_roll_delta(0), "left"))    # 左摇杆 X
                right_action = _apply_roll(right_action, _limit_roll(_roll_delta(2), "right"))  # 右摇杆 X

                # ---- 头部（绝对目标）----
                if head_target is not None:
                    head_yaw_pos = float(np.clip(math.radians(head_target.yaw),
                                                 -VR_PARAMS.head_yaw_limit, VR_PARAMS.head_yaw_limit))
                    head_pitch_pos = float(np.clip(math.radians(head_target.pitch),
                                                   -VR_PARAMS.head_pitch_limit, VR_PARAMS.head_pitch_limit))
                    try:
                        self.robot.move_head([head_yaw_pos, head_pitch_pos])
                    except Exception as e:
                        print(f"vr move_head error: {e}")

                # ---- 夹爪（触发即下发当前缓存值，无需每拍重发）----
                if left_grip_changed or right_grip_changed:
                    try:
                        self.robot.move_gripper([self.left_gripper_pos, self.right_gripper_pos])
                    except Exception as e:
                        print(f"vr move_gripper error: {e}")

                # ---- 臂部（增量目标）----
                robot_actions = {}
                if left_action is not None:
                    robot_actions["left_arm"] = {
                        "action_data": left_action,
                        "control_type": "DELTA_POSE",
                    }
                if right_action is not None:
                    robot_actions["right_arm"] = {
                        "action_data": right_action,
                        "control_type": "DELTA_POSE",
                    }
                if robot_actions:
                    try:
                        arm_states, _ = self.robot.arm_joint_states()
                        head_states, _ = self.robot.head_joint_states()
                        waist_states, _ = self.robot.waist_joint_states()
                        assert arm_states
                        assert head_states
                        assert waist_states
                        robot_states = {
                            "head": head_states,
                            "waist": waist_states,
                            "arm": arm_states,
                        }
                        # time_step 与节拍对齐 → 给底层 20ms 平滑插补
                        self.robot_controller.trajectory_tracking_control(
                            int(time.time() * 1e9),
                            robot_states,
                            [robot_actions],
                            "base_link",
                            VR_PARAMS.execution_period,
                        )
                    except Exception as e:
                        print(f"vr trajectory_tracking_control error: {e}")

                # ---- 定步长 sleep；落后则只追上不补帧 ----
                next_tick += VR_PARAMS.execution_period
                sleep_for = next_tick - time.monotonic()
                if sleep_for > 0:
                    time.sleep(sleep_for)
                else:
                    next_tick = time.monotonic()
            print("Stopping vr execution thread...")

        if self.is_vr_control:
            if self.vr_execution_thread is None:
                self.vr_actions = []
                self.vr_execution_thread = threading.Thread(target=_vr_auto_execution_thread, daemon=True)
                self.vr_execution_thread.start()
        elif self.vr_execution_thread is not None:
            self.vr_actions = []
            self.vr_execution_thread.join()
            self.vr_execution_thread = None

        now = time.time()

        if self.is_vr_control and now - self.last_joint_update_timestamp > 0.2:
            print("Resetting vr execution thread, due to VR connection time out: 200ms")
            self.is_vr_control = False
            self.vr_actions = []
            self.previous_vr_positions = []

        self.last_joint_update_timestamp = now

        positions = state.positions
        triggers = state.triggers
        axes = state.axes
        assert len(triggers) >= 2
        left_grab = bool(triggers[0])
        right_grab = bool(triggers[1])
        buttons = _fmt_buttons(triggers)
        # Cache the currently-pressed button set for any sibling consumer
        # (e.g. DataCollectionMixin uses {"L_Y", "R_B"} as the recording
        # gate). Snapshotted as a new set every callback so consumers can
        # do membership tests without locking. last_joint_update_timestamp
        # is the staleness signal — if it ages out, treat buttons as released.
        self.vr_buttons_pressed = set(buttons)
        assert len(axes) == 4
        # Cache the raw thumbstick axes for the 50Hz exec loop's stick-driven wrist
        # roll (axes[0]=left stick x, axes[2]=right stick x). Replaced as a new list
        # each callback so the exec thread reads a consistent snapshot without locking;
        # last_joint_update_timestamp is the staleness signal (exec loop times out on it).
        self.vr_axes = list(axes)
        left_stick = _fmt_stick(axes, 0)
        right_stick = _fmt_stick(axes, 2)
        assert len(positions) == 21
        # 在原始 7 维姿态（px,py,pz,qx,qy,qz,qw）上做 One-Euro 滤波
        # 滤波器状态挂在 self 上，连续跨回调；按手柄分组，独立维护时间。
        filter_t = time.monotonic()
        head_raw = self._vr_filter_apply("head", positions[0:7], filter_t)
        left_raw = self._vr_filter_apply("left_hand", positions[7:14], filter_t)
        right_raw = self._vr_filter_apply("right_hand", positions[14:21], filter_t)
        head_position = _format_pose(head_raw)
        left_hand_position = _format_pose(left_raw)
        right_hand_position = _format_pose(right_raw)

        # head, left, right, left gripper, right gripper
        action = [None, None, None, None, None]
        if self.previous_vr_positions and self.is_vr_control:
            if "L_Y" in buttons:
                action[1] = left_hand_position.get_action_delta(self.previous_vr_positions[1])
            if "R_B" in buttons:
                action[2] = right_hand_position.get_action_delta(self.previous_vr_positions[2])
            # TODO: gripper
            if left_grab and left_grab != self.previous_vr_positions[3]:
                self.left_gripper_pos = 0.0 if self.left_gripper_pos > 0.5 else 1.0
                action[3] = self.left_gripper_pos > 0.5
            if right_grab and right_grab != self.previous_vr_positions[4]:
                self.right_gripper_pos = 0.0 if self.right_gripper_pos > 0.5 else 1.0
                action[4] = self.right_gripper_pos > 0.5
        if any(act is not None for act in action):
            # 防止用户长按导致积压：超出上限则丢最早的一条（保持最新意图优先）
            while len(self.vr_actions) >= VR_PARAMS.queue_max:
                self.vr_actions.pop(0)
            self.vr_actions.append(action)
        elif any(any(value is not None for value in act[:3]) for act in self.vr_actions):
            # clear all pending move action when buttons released
            self.vr_actions = []

        self.previous_vr_positions = [head_position, left_hand_position, right_hand_position, left_grab, right_grab]

        if not hasattr(self, 'vr_info'):
            self.vr_info = [now, 0]
        else:
            self.vr_info[-1] += 1
            if now - self.vr_info[0] > 1:
                print("now: ", now, "fps: ", self.vr_info[-1] / (now - self.vr_info[0]))
                self.vr_info = [now, 0]
            else:
                return

        print("=" * 10)
        print("head_position: ", head_position.format())
        print("left_hand_position: ", left_hand_position.format())
        print("right_hand_position: ", right_hand_position.format())
        print("buttons: ", buttons)
        print("left_grab: ", left_grab)
        print("right_grab: ", right_grab)
        print("left_stick: ", left_stick)
        print("right_stick: ", right_stick)
        print("=" * 10)
        # if len(positions) == 21:
        #     head = positions[0:7]
        #     left = positions[7:14]
        #     right = positions[14:21]
        #     logger.info(
        #         f"xr pose t={state.timestamp:.3f}  "
        #         f"{_format_pose('H', head)}  |  "
        #         f"{_format_pose('L', left)} grab={_fmt_trigger(left_grab)} stk={left_stick}  |  "
        #         f"{_format_pose('R', right)} grab={_fmt_trigger(right_grab)} stk={right_stick}  "
        #         f"btn={buttons}"
        #     )
        # elif len(positions) == 14:
        #     left = positions[0:7]
        #     right = positions[7:14]
        #     logger.info(
        #         f"controller pose t={state.timestamp:.3f}  "
        #         f"{_format_pose('L', left)} grab={_fmt_trigger(left_grab)} stk={left_stick}  |  "
        #         f"{_format_pose('R', right)} grab={_fmt_trigger(right_grab)} stk={right_stick}  "
        #         f"btn={buttons}"
        #     )
        # else:
        #     preview = ", ".join(f"{p:+.3f}" for p in positions[:6])
        #     logger.info(
        #         f"upstream joints t={state.timestamp:.3f} n={len(positions)} "
        #         f"positions=[{preview}] triggers={triggers} axes={axes}"
        #     )

    def _vr_filter_apply(self, key: str, vec7, t: float) -> list:
        """对 7 维原始姿态 (px,py,pz,qx,qy,qz,qw) 做 One-Euro 滤波。

        位置 3 维与四元数 4 维各自独立通道；滤波完后对四元数重新归一化，
        避免漂出单位球。滤波器按 key 持久化于 self._vr_pose_filters。
        """
        if not hasattr(self, "_vr_pose_filters"):
            self._vr_pose_filters = {}
        filters = self._vr_pose_filters.get(key)
        if filters is None:
            p = VR_PARAMS
            filters = [
                # 位置（米）：手部静止时强力抑制 4-12Hz 生理性抖动
                OneEuroFilter(p.pos_mincutoff, p.pos_beta, p.pos_dcutoff),
                OneEuroFilter(p.pos_mincutoff, p.pos_beta, p.pos_dcutoff),
                OneEuroFilter(p.pos_mincutoff, p.pos_beta, p.pos_dcutoff),
                # 四元数 4 维：静止时更激进，转动快时立即放开
                OneEuroFilter(p.quat_mincutoff, p.quat_beta, p.quat_dcutoff),
                OneEuroFilter(p.quat_mincutoff, p.quat_beta, p.quat_dcutoff),
                OneEuroFilter(p.quat_mincutoff, p.quat_beta, p.quat_dcutoff),
                OneEuroFilter(p.quat_mincutoff, p.quat_beta, p.quat_dcutoff),
            ]
            self._vr_pose_filters[key] = filters

        out = [filters[i](float(vec7[i]), t) for i in range(7)]
        # 四元数重新归一化
        qx, qy, qz, qw = out[3], out[4], out[5], out[6]
        n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
        if n > 1e-6:
            inv = 1.0 / n
            out[3] = qx * inv
            out[4] = qy * inv
            out[5] = qz * inv
            out[6] = qw * inv
        return out

    # ====================== VR 灵敏度参数面板 ====================== #
    # 把 VRParameters 里的灵敏度/平滑/限幅参数暴露成滑块 + 输入框,带安全
    # 上下限。用户拖动/输入即时生效:
    #   - 绝大多数参数:调用处都「按名读 VR_PARAMS.x」,所以用
    #     dataclasses.replace 原子替换模块全局 VR_PARAMS 即可立即生效。
    #   - One-Euro 滤波三参数:已创建的滤波器实例缓存了取值,额外把新值推到
    #     self._vr_pose_filters 里现存实例上(_vrp_sync_filters)才会立即生效。
    # 替换/赋值都是原子操作,VR 回调线程与执行线程并发读取无需加锁(frozen
    # dataclass 不会被读到半成品)。

    def _vrp_ensure_state(self):
        if getattr(self, "_vrp_inited", False):
            return
        self._vrp_inited = True
        self._vrp_spec = {s[0]: s for s in VR_PARAM_SPECS}
        self._vrp_scales = {}
        self._vrp_entries = {}
        # 程序性回写 UI 时置位,避免 Scale/Entry 回调互相触发成环。
        self._vrp_updating = False

    @staticmethod
    def _vrp_decimals(res: float) -> int:
        if res >= 1:
            return 0
        return max(0, int(math.ceil(-math.log10(res))))

    def _vrp_fmt(self, attr: str, value) -> str:
        spec = self._vrp_spec[attr]
        if spec[5]:  # is_int
            return str(int(round(value)))
        return f"{float(value):.{self._vrp_decimals(spec[4])}f}"

    def setup_vr_params_panel(self, parent):
        """在 Notebook `parent` 上新增「VR 灵敏度」标签页。"""
        import tkinter as tk
        from tkinter import ttk

        self._vrp_ensure_state()

        tab = ttk.Frame(parent)
        parent.add(tab, text="🎚  VR 灵敏度")

        # 顶部说明 + 重置按钮
        top = ttk.Frame(tab)
        top.pack(fill=tk.X, padx=12, pady=(12, 4))
        ttk.Label(top, text="拖动滑块或输入数值即时生效(已限制安全范围)",
                  style="Subtitle.TLabel").pack(side=tk.LEFT)
        ttk.Button(top, text="↺  重置默认", style="Muted.TButton",
                   command=self._vrp_reset).pack(side=tk.RIGHT)

        # 可滚动容器(参数较多)
        canvas = tk.Canvas(tab, highlightthickness=0,
                           bg=(ttk.Style().lookup("TFrame", "background") or "#f5f6f8"))
        sb = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        body = ttk.Frame(canvas)
        body.bind("<Configure>",
                  lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=body, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side="left", fill="both", expand=True, padx=(12, 0), pady=4)
        sb.pack(side="right", fill="y")

        # 按分组渲染
        seen_groups = []
        for spec in VR_PARAM_SPECS:
            group = spec[6]
            if group not in seen_groups:
                seen_groups.append(group)
                gframe = ttk.LabelFrame(body, text=group)
                gframe.pack(fill=tk.X, expand=True, padx=4, pady=(6, 2))
                self._vrp_groups = getattr(self, "_vrp_groups", {})
                self._vrp_groups[group] = gframe
            self._vrp_add_row(self._vrp_groups[group], spec)

        return tab

    def _vrp_add_row(self, parent, spec):
        import tkinter as tk
        from tkinter import ttk

        attr, label, lo, hi, res, is_int, _group = spec
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, padx=6, pady=3)

        ttk.Label(row, text=label, width=18, anchor="w").pack(side=tk.LEFT)

        cur = getattr(VR_PARAMS, attr)
        scale = tk.Scale(
            row, from_=lo, to=hi, resolution=res,
            orient=tk.HORIZONTAL, showvalue=False, length=180,
            command=lambda v, a=attr: self._vrp_on_scale(a, v),
        )
        scale.set(cur)
        scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 6))
        self._vrp_scales[attr] = scale

        entry = ttk.Entry(row, width=8, justify="right")
        entry.insert(0, self._vrp_fmt(attr, cur))
        entry.bind("<Return>", lambda e, a=attr: self._vrp_on_entry(a))
        entry.bind("<FocusOut>", lambda e, a=attr: self._vrp_on_entry(a))
        entry.pack(side=tk.RIGHT)
        self._vrp_entries[attr] = entry

        # 范围提示
        ttk.Label(row, text=f"[{self._vrp_fmt(attr, lo)},{self._vrp_fmt(attr, hi)}]",
                  style="Subtitle.TLabel").pack(side=tk.RIGHT, padx=(6, 4))

    # ---- 回调 ----

    def _vrp_on_scale(self, attr, strval):
        if self._vrp_updating:
            return
        try:
            v = float(strval)
        except (TypeError, ValueError):
            return
        v = self._vrp_apply(attr, v)
        self._vrp_set_entry(attr, v)

    def _vrp_on_entry(self, attr):
        if self._vrp_updating:
            return
        entry = self._vrp_entries[attr]
        try:
            raw = float(entry.get())
        except (TypeError, ValueError):
            # 非法输入 → 回退到当前值
            self._vrp_set_ui(attr, getattr(VR_PARAMS, attr))
            return
        # 先让 Scale 钳到量程并按 resolution 吸附,再以它为准应用
        self._vrp_updating = True
        try:
            self._vrp_scales[attr].set(raw)
        finally:
            self._vrp_updating = False
        v = self._vrp_apply(attr, float(self._vrp_scales[attr].get()))
        self._vrp_set_entry(attr, v)

    def _vrp_set_entry(self, attr, value):
        self._vrp_updating = True
        try:
            entry = self._vrp_entries[attr]
            entry.delete(0, "end")
            entry.insert(0, self._vrp_fmt(attr, value))
        finally:
            self._vrp_updating = False

    def _vrp_set_ui(self, attr, value):
        """程序性把滑块 + 输入框设到 value(不重复 apply)。"""
        self._vrp_updating = True
        try:
            self._vrp_scales[attr].set(value)
            entry = self._vrp_entries[attr]
            entry.delete(0, "end")
            entry.insert(0, self._vrp_fmt(attr, value))
        finally:
            self._vrp_updating = False

    # ---- 应用 / 重置 ----

    def _vrp_apply(self, attr, value):
        """钳到安全区间后,原子替换全局 VR_PARAMS;若是滤波参数再推到现存
        滤波器实例。返回最终生效值。"""
        spec = self._vrp_spec[attr]
        lo, hi, is_int = spec[2], spec[3], spec[5]
        value = int(round(value)) if is_int else float(value)
        value = max(lo, min(hi, value))

        global VR_PARAMS
        VR_PARAMS = dataclasses.replace(VR_PARAMS, **{attr: value})

        if attr in _VRP_FILTER_ATTRS:
            self._vrp_sync_filters()
        return value

    def _vrp_sync_filters(self):
        """把当前 VR_PARAMS 的 One-Euro 三参数推到已创建的滤波器实例上。

        滤波器在每次 __call__ 时读 self.mincutoff/beta/dcutoff,故就地改属性
        即可立即生效,且保留各通道的历史状态(不重置,避免突跳)。"""
        filters_map = getattr(self, "_vr_pose_filters", None)
        if not filters_map:
            return
        p = VR_PARAMS
        for filters in filters_map.values():
            for i in range(0, 3):       # 位置三通道
                filters[i].mincutoff = p.pos_mincutoff
                filters[i].beta = p.pos_beta
                filters[i].dcutoff = p.pos_dcutoff
            for i in range(3, 7):       # 四元数四通道
                filters[i].mincutoff = p.quat_mincutoff
                filters[i].beta = p.quat_beta
                filters[i].dcutoff = p.quat_dcutoff

    def _vrp_reset(self):
        """一键恢复出厂默认(VR_PARAMS_DEFAULTS),刷新所有控件。"""
        global VR_PARAMS
        VR_PARAMS = VRParameters()
        self._vrp_sync_filters()
        for attr in self._vrp_scales:
            self._vrp_set_ui(attr, getattr(VR_PARAMS, attr))
        if getattr(self, "status_text", None) is not None:
            try:
                self.status_text.set("VR 灵敏度参数已重置为默认")
            except Exception:
                pass

    def start_vr_stream_thread(self):
        """启动 VR 串流线程：将 self.camera_images 合成为一张 image，推给 DummyServer。"""
        target_period = 1.0 / 30.0

        def _to_rgb_uint8(img):
            """Normalize an entry from self.camera_images to an HxWx3 RGB uint8 ndarray."""
            if img is None:
                return None
            if isinstance(img, (list, tuple)):
                if not img:
                    return None
                img = img[-1]
            if not isinstance(img, np.ndarray):
                return None
            if img.ndim == 2:
                # 深度或灰度：归一化到 8-bit，再展成三通道
                arr = img.astype(np.float32)
                if np.isfinite(arr).any():
                    valid = arr[np.isfinite(arr)]
                    vmin = float(valid.min()) if valid.size else 0.0
                    vmax = float(valid.max()) if valid.size else 1.0
                else:
                    vmin, vmax = 0.0, 1.0
                span = max(vmax - vmin, 1e-6)
                arr = np.clip((arr - vmin) / span, 0.0, 1.0)
                arr = (arr * 255.0).astype(np.uint8)
                return np.stack([arr, arr, arr], axis=-1)
            if img.ndim == 3 and img.shape[2] >= 3:
                arr = img[:, :, :3]
                if arr.dtype != np.uint8:
                    arr = np.clip(arr, 0, 255).astype(np.uint8)
                return arr
            return None

        def _compose(frames):
            """水平拼接所有相机的最新帧，统一到相同高度。"""
            if not frames:
                return None
            target_h = max(f.shape[0] for f in frames)
            resized = []
            for f in frames:
                h, w = f.shape[:2]
                if h != target_h:
                    new_w = max(1, int(round(w * target_h / h)))
                    f = cv2.resize(f, (new_w, target_h), interpolation=cv2.INTER_AREA)
                resized.append(f)
            return np.concatenate(resized, axis=1)

        def stream_loop():
            next_tick = time.monotonic()
            while True:
                try:
                    frames = []
                    for name in sorted(self.camera_images.keys()):
                        rgb = _to_rgb_uint8(self.camera_images.get(name))
                        if rgb is not None:
                            frames.append(rgb)
                    composite = _compose(frames)
                    if composite is not None:
                        self.dummy_server.set_image(composite, color_order="RGB")
                except Exception as e:
                    print(f"VR 串流合成错误: {e}")
                next_tick += target_period
                sleep_for = next_tick - time.monotonic()
                if sleep_for > 0:
                    time.sleep(sleep_for)
                else:
                    next_tick = time.monotonic()

        stream_thread = threading.Thread(target=stream_loop, daemon=True)
        stream_thread.start()
