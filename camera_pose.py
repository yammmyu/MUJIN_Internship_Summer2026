import numpy as np
import math


def euler_to_rotation_matrix(roll, pitch, yaw, order='ZYX'):
    """
    将欧拉角转换为旋转矩阵

    参数:
        roll, pitch, yaw: 欧拉角 (弧度)
        order: 旋转顺序，默认为 'ZYX' (yaw -> pitch -> roll)

    返回:
        3x3 旋转矩阵 (从相机坐标系到世界坐标系)
    """
    # 计算三角函数
    cr = math.cos(roll)
    sr = math.sin(roll)
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cy = math.cos(yaw)
    sy = math.sin(yaw)

    if order == 'ZYX':  # 先 yaw (Z), 再 pitch (Y), 再 roll (X)
        R_z = np.array([[cy, -sy, 0],
                        [sy, cy, 0],
                        [0, 0, 1]])

        R_y = np.array([[cp, 0, sp],
                        [0, 1, 0],
                        [-sp, 0, cp]])

        R_x = np.array([[1, 0, 0],
                        [0, cr, -sr],
                        [0, sr, cr]])

        # 注意乘法顺序：R = R_x @ R_y @ R_z
        R = R_x @ R_y @ R_z

    elif order == 'XYZ':  # 先 roll (X), 再 pitch (Y), 再 yaw (Z)
        R_x = np.array([[1, 0, 0],
                        [0, cr, -sr],
                        [0, sr, cr]])

        R_y = np.array([[cp, 0, sp],
                        [0, 1, 0],
                        [-sp, 0, cp]])

        R_z = np.array([[cy, -sy, 0],
                        [sy, cy, 0],
                        [0, 0, 1]])

        R = R_z @ R_y @ R_x

    else:
        raise ValueError(f"不支持的旋转顺序: {order}")

    return R


def compute_point_B_world(A_c, B_c, A_w, roll, pitch, yaw, order='ZYX'):
    """
    计算 B 点的世界坐标

    参数:
        A_c: A 点在相机坐标系中的坐标 [x, y, z]
        B_c: B 点在相机坐标系中的坐标 [x, y, z]
        A_w: A 点在世界坐标系中的坐标 [x, y, z]
        roll, pitch, yaw: 相机姿态欧拉角 (弧度)
        order: 欧拉角旋转顺序，默认为 'ZYX'

    返回:
        B_w: B 点在世界坐标系中的坐标 [x, y, z]
    """
    # 转换为 numpy 数组
    A_c = np.array(A_c)
    B_c = np.array(B_c)
    A_w = np.array(A_w)

    # 计算旋转矩阵
    R = euler_to_rotation_matrix(roll, pitch, yaw, order)

    # 计算相对向量并旋转到世界坐标系
    delta_w = R @ (B_c - A_c)

    # 计算 B 的世界坐标
    B_w = A_w + delta_w

    return B_w


def compute_T_and_B(A_c, B_c, A_w, roll, pitch, yaw, order='ZYX'):
    """
    另一种计算方法：先求平移向量 T，再求 B

    返回:
        T: 平移向量 (相机原点在世界坐标系中的位置)
        B_w: B 点世界坐标
    """
    R = euler_to_rotation_matrix(roll, pitch, yaw, order)
    A_c = np.array(A_c)
    B_c = np.array(B_c)
    A_w = np.array(A_w)

    # 计算平移向量
    T = A_w - R @ A_c

    # 计算 B 的世界坐标
    B_w = R @ B_c + T

    return T, B_w


# ============ 示例使用 ============

if __name__ == "__main__":
    # # 示例 1: 简单情况
    # print("=" * 60)
    # print("示例 1: 相机水平放置，无旋转")
    # print("=" * 60)
    #
    # # 相机坐标系中的点
    # A_c = [1.0, 0.0, 2.0]  # A 点: x=1, y=0, z=2 (前方2米)
    # B_c = [1.0, 1.0, 2.5]  # B 点: 在 A 点右方1米，前方0.5米
    #
    # # A 点世界坐标
    # A_w = [10.0, 10.0, 0.0]  # 地面上的点 (假设 Z 向上)
    #
    # # 相机姿态: 无旋转 (相机坐标系与世界坐标系对齐)
    # roll = 0.0
    # pitch = 0.0
    # yaw = 0.0
    #
    # B_w = compute_point_B_world(A_c, B_c, A_w, roll, pitch, yaw)
    # print(f"A_c = {A_c}")
    # print(f"B_c = {B_c}")
    # print(f"A_w = {A_w}")
    # print(f"相机姿态: roll={roll:.1f}°, pitch={pitch:.1f}°, yaw={yaw:.1f}°")
    # print(f"B_w = {B_w}")
    # print()
    #
    # # 示例 2: 相机旋转 90 度（绕 Z 轴）
    # print("=" * 60)
    # print("示例 2: 相机绕 Z 轴旋转 90° (向右看)")
    # print("=" * 60)
    #
    # yaw_deg = 90.0
    # yaw_rad = math.radians(yaw_deg)
    #
    # B_w = compute_point_B_world(A_c, B_c, A_w, roll, pitch, yaw_rad)
    # print(f"A_c = {A_c}")
    # print(f"B_c = {B_c}")
    # print(f"A_w = {A_w}")
    # print(f"相机姿态: roll={roll:.1f}°, pitch={pitch:.1f}°, yaw={yaw_deg}°")
    # print(f"B_w = {B_w}")
    # print()
    #
    # # 示例 3: 有俯仰和滚转的情况
    # print("=" * 60)
    # print("示例 3: 复杂姿态 (俯仰30°, 滚转15°, 偏航45°)")
    # print("=" * 60)
    #
    # roll_deg = 15.0
    # pitch_deg = 30.0
    # yaw_deg = 45.0
    #
    # roll_rad = math.radians(roll_deg)
    # pitch_rad = math.radians(pitch_deg)
    # yaw_rad = math.radians(yaw_deg)
    #
    # B_w = compute_point_B_world(A_c, B_c, A_w, roll_rad, pitch_rad, yaw_rad)
    # print(f"A_c = {A_c}")
    # print(f"B_c = {B_c}")
    # print(f"A_w = {A_w}")
    # print(f"相机姿态: roll={roll_deg}°, pitch={pitch_deg}°, yaw={yaw_deg}°")
    # print(f"B_w = {B_w}")
    # print()
    #
    # # 验证两种方法结果一致
    # print("=" * 60)
    # print("验证: 两种计算方法结果对比")
    # print("=" * 60)
    #
    # T, B_w2 = compute_T_and_B(A_c, B_c, A_w, roll_rad, pitch_rad, yaw_rad)
    # print(f"方法1 (直接公式): B_w = {B_w}")
    # print(f"方法2 (先求 T):    B_w = {B_w2}")
    # print(f"平移向量 T (相机原点在世界中的位置): {T}")
    # print()
    #
    # # 验证旋转矩阵的性质
    # print("=" * 60)
    # print("验证: 旋转矩阵的正交性")
    # print("=" * 60)
    #
    # R = euler_to_rotation_matrix(roll_rad, pitch_rad, yaw_rad)
    # print(f"旋转矩阵 R:\n{R}")
    # print(f"R @ R^T (应该是单位矩阵):\n{R @ R.T}")
    # print(f"行列式 det(R) = {np.linalg.det(R):.6f} (应为1或-1)")


    print("=" * 60)

    yaw_deg = -28.64788975654116
    # yaw_deg = 0
    yaw_rad = math.radians(yaw_deg)
    print(yaw_rad)
    roll = 0
    # pitch = math.radians(0.2199999237060547)
    pitch = 0

    # 相机坐标系中的点
    # A_c = [0.22, -0.378, 1.207]  # A 点
    # B_c = [0.404, -0.434, 1.207]  # B 点

    A_c = [-0.199, 0.076, 0.466]  # A 点
    B_c = [0.019, 0.045, 0.466]  # B 点

    # A 点世界坐标
    A_w = [0.5214255306006392, 0.4455203887213439, 0.8947249170524721]  # 地面上的点 (假设 Z 向上)

    B_w = compute_point_B_world(A_c, B_c, A_w, roll, pitch, yaw_rad)
    print(f"A_c = {A_c}")
    print(f"B_c = {B_c}")
    print(f"A_w = {A_w}")
    print(f"相机姿态: roll={roll:.1f}°, pitch={pitch:.1f}°, yaw={yaw_deg}°")
    print(f"B_w = {B_w}")
    print()

    print(f"Diff={np.array(B_w) - A_w}")