import numpy as np
import time
import json
from a2d_sdk.robot import RobotDds as Robot, CosineCamera as Camera, RobotController, Slam
from scipy.spatial.transform import Rotation as R


class RobotCoordinateTransformer:
    def __init__(self, urdf_params=None):
        # 从URDF中提取的固定偏移量 (单位: 米)
        self.offsets = {
            'lift_base': [0, 0, 0.6485],              # base_link -> link-up-down_body
            'body_pitch_origin': [0.131, 0, 0],        # link-up-down_body -> link-pitch_body
            'head_yaw_origin': [0.441, 0, 0],          # link-pitch_body -> link-yaw_head
            'head_pitch_origin': [-0.050238, 0, 0.060065], # link-yaw_head -> link-pitch_head
            'camera_offset': [0, 0, 0]                 # link-pitch_head -> 相机光学中心
        }

        # 机械臂段长度 (用于IK求解)
        self.arm_lengths = [0.188, 0.305, 0.1975, 0.181, 0.23]
        self.left_arm_home = [-1.0743615627288818, 0.6110466122627258, 0.2795490026473999, -1.2839884757995605, 0.7303872108459473, 1.4954301118850708, -0.18760496377944946]
        self.right_arm_home = [1.0743964910507202, -0.6110990047454834, -0.27946174144744873, 1.2839535474777222, -0.7304046154022217, -1.4952905178070068, 0.18762239813804626]

        self.left_arm_pose = [ 0.89756692, -0.59635312, 0.66221681]
        self.right_arm_pose = [0.89754079, 0.59639798, 0.73783894]

    def get_transformation_matrix(self, translation, rpy):
        """创建齐次变换矩阵"""
        mat = np.eye(4)
        mat[:3, :3] = R.from_euler('xyz', rpy).as_matrix()
        mat[:3, 3] = translation
        return mat

    def compute_camera_to_base_matrix(self, joint_angles):
        """
        计算相机坐标系到机器人基座的变换矩阵
        joint_angles: [waist_lift, waist_pitch, head_yaw, head_pitch]
        """
        w_lift, w_pitch, h_yaw, h_pitch = joint_angles

        # 1. Base -> Lift Body (Prismatic)
        T_lift = np.eye(4)
        T_lift[2, 3] = self.offsets['lift_base'][2] + w_lift

        # 2. Lift Body -> Pitch Body (Revolute)
        T_body_pitch = self.get_transformation_matrix(self.offsets['body_pitch_origin'], [1.5708, -1.5708, 0])
        R_body_pitch = R.from_rotvec([0, 0, -w_pitch]).as_matrix()
        T_body_pitch[:3, :3] = T_body_pitch[:3, :3] @ R_body_pitch

        # 3. Pitch Body -> Yaw Head (Fixed/Revolute)
        T_head_yaw = self.get_transformation_matrix(self.offsets['head_yaw_origin'], [1.5708, 0, 1.5708])
        R_head_yaw = R.from_rotvec([0, 0, h_yaw]).as_matrix()
        T_head_yaw[:3, :3] = T_head_yaw[:3, :3] @ R_head_yaw

        # 4. Yaw Head -> Pitch Head (Fixed/Revolute)
        T_head_pitch = self.get_transformation_matrix(self.offsets['head_pitch_origin'], [1.5708, 0, 0])
        R_head_pitch = R.from_rotvec([0, 0, h_pitch]).as_matrix()
        T_head_pitch[:3, :3] = T_head_pitch[:3, :3] @ R_head_pitch

        # 组合所有变换
        T_base_camera = T_lift @ T_body_pitch @ T_head_yaw @ T_head_pitch
        return T_base_camera

    def pixel_to_world(self, pixel_x, pixel_y, depth_value, intrinsics, joint_angles):
        """将像素点转换为世界坐标系下的位置"""
        fx, fy = intrinsics['fx'], intrinsics['fy']
        cx, cy = intrinsics['cx'], intrinsics['cy']
        z = depth_value * 0.001 # mm -> m

        x_cam = (pixel_x - cx) * z / fx
        y_cam = (pixel_y - cy) * z / fy
        p_cam = np.array([x_cam, y_cam, z, 1.0])

        T_base_camera = self.compute_camera_to_base_matrix(joint_angles)
        p_base = T_base_camera @ p_cam
        return p_base[:3]

    def compute_arm_fk(self, side, joint_angles, rot=[0, 0, 0]):
        """
        计算手臂末端在基座坐标系下的位置 (简易版)
        joint_angles: [q1, q2, q3, q4, q5, q6, q7]
        """
        # 初始位置 (手臂挂载点)
        # p = np.array([0.3, 0.025 if side == 'left' else -0.025, 0.7])
        p = np.array([0.3, 0.25 if side == 'left' else -0.25, 0.7])
        # p = np.array([0, 0, 0.15 if side == 'left' else -0.15])

        # 简化正运动学：将各段长度沿当前旋转方向累加
        # 注意：这里是简化模型，实际应使用 URDF 的完整 DH 参数或变换矩阵链
        current_rot = R.from_euler('xyz', rot)

        for i in range(len(self.arm_lengths)):
            # 更新当前旋转 (简化：假设每个关节绕特定轴旋转)
            if i < len(joint_angles):
                # 简化轴向映射
                axis = [0,0,1] if i%2==0 else [0,1,0]
                current_rot = current_rot * R.from_rotvec(np.array(axis) * joint_angles[i])

            # 沿当前方向移动一段距离
            direction = current_rot.apply([1, 0, 0]) # 假设臂沿X轴延伸
            p = p + direction * self.arm_lengths[i]

        return p

    def solve_ik(self, side, target_pos, current_joints):
        """
        使用数值优化方法求解逆运动学 (IK)
        target_pos: [x, y, z] 目标位置
        current_joints: 当前关节角度 (用于作为优化起点，减少突跳)
        """
        from scipy.optimize import minimize

        def objective(q):
            # 计算当前关节角下的末端位置与目标位置的欧氏距离
            current_pos = self.compute_arm_fk(side, q)
            return np.linalg.norm(current_pos - target_pos)

        # 关节限位 (从 URDF 提取)
        bounds = [(-3.14, 3.14)] * 7 # 简化处理，实际应根据 URDF limit 设置

        res = minimize(objective, current_joints, bounds=bounds, method='L-BFGS-B')

        if res.success:
            return res.x
        else:
            return None

class TrajectoryPlanner:
    """基于RRT-Connect的轨迹规划器"""
    def __init__(self, transformer, collision_checker):
        self.transformer = transformer
        self.collision_checker = collision_checker
        self.max_iter = 5000
        self.step_size = 0.1 # 关节空间步长(rad)

    def plan_path(self, side, start_joints, target_pos, other_hand_start_joints):
        """
        在 A 点和 B 点之间寻找一条不碰撞的路径
        start_joints: [q1...q7] 起始关节角
        target_pos: [x, y, z] 目标末端位置
        """
        # 1. 首先尝试用 IK 求解目标点的关节角
        goal_joints = self.transformer.solve_ik(side, target_pos, start_joints)
        print("goal_joints = ", goal_joints)
        if goal_joints is None:
            print("IK 求解失败，无法到达目标点")
            return None

        # 2. RRT 采样搜索
        # 状态定义: q = [q1, q2, q3, q4, q5, q6, q7]
        tree = {tuple(start_joints): None} # {node: parent}
        nodes = [np.array(start_joints)]

        for i in range(self.max_iter):
            # 随机采样目标点
            if np.random.rand() < 0.1: # 10% 概率直接向目标点生长
                sample = np.array(goal_joints)
            else:
                # 在关节限位内随机采样
                # sample = np.random.uniform(-3.14, 3.14, 7)
                sample = [np.random.uniform(value - 0.3, value + 0.3) for value in start_joints]

            # 找到树中最近的节点
            dists = [np.linalg.norm(node - sample) for node in nodes]
            nearest_node = nodes[np.argmin(dists)]

            # 向采样点方向延伸一步
            diff = sample - nearest_node
            dist = np.linalg.norm(diff)
            if dist > 0:
                step = (diff / dist) * self.step_size
                new_node = nearest_node + step

                # 碰撞检测: 检查新节点
                if side == 'left':
                    collided, msg = self.collision_checker.check_collision(new_node, other_hand_start_joints)
                else:
                    collided, msg = self.collision_checker.check_collision(other_hand_start_joints, new_node)

                if not collided:
                    tree[tuple(new_node)] = tuple(nearest_node)
                    nodes.append(new_node)

                    # 检查是否到达目标点
                    if np.linalg.norm(new_node - goal_joints) < self.step_size:
                        # 找到路径，回溯
                        path = []
                        curr = tuple(new_node)
                        while curr is not None:
                            path.append(list(curr))
                            curr = tree[curr]
                        return path[::-1]

        print("RRT 规划在最大迭代次数内未找到路径")
        return None

    def smooth_path(self, path):
        """简单的路径平滑: 尝试删除冗余中间点"""
        if len(path) <= 2: return path

        smoothed = [path[0]]
        curr_idx = 0
        while curr_idx < len(path) - 1:
            # 尝试从当前点直接跳到后面尽可能远且不碰撞的点
            for next_idx in range(len(path)-1, curr_idx, -1):
                # 这里简化处理，假设直接跳跃，实际应进行采样检测
                smoothed.append(path[next_idx])
                curr_idx = next_idx
                break
        return smoothed

class CollisionChecker:

    """基于简化几何体的机器人自碰撞检测类"""
    def __init__(self):
        # 定义安全半径 (米)
        self.safety_radius = 0.04

        # 定义机器人身体的简化几何体 (相对中心点, 半径, 高度)
        # 简化为几个核心圆柱体
        self.body_cylinders = [
            {'name': 'base', 'pos': [0, 0, 0.3], 'radius': 0.3, 'height': 0.6},
            {'name': 'torso', 'pos': [0.1, 0, 0.8], 'radius': 0.2, 'height': 0.4}
        ]

    def _dist_line_to_line(self, p1, p2, p3, p4):
        """计算两条线段 (p1,p2) 和 (p3,p4) 之间的最短距离"""
        u = p2 - p1
        v = p4 - p3
        w = p1 - p3
        a = np.dot(u, u)
        b = np.dot(u, v)
        c = np.dot(v, v)
        d = np.dot(u, w)
        e = np.dot(v, w)
        D = a * c - b * b

        sc, tc = 0, 0

        if D < 1e-8: # 线段平行
            sc = 0.0
            tc = d / b if b > 0 else 0.0
        else:
            sc = (b * e - c * d) / D
            tc = (a * e - b * d) / D

        sc = np.clip(sc, 0, 1)
        tc = np.clip(tc, 0, 1)

        closest_p1 = p1 + sc * u
        closest_p2 = p3 + tc * v
        return np.linalg.norm(closest_p1 - closest_p2)

    def get_arm_segments(self, side, joint_angles):
        """
        根据关节角计算手臂各段的端点坐标
        joint_angles: 7个手臂关节角度 (rad)
        """
        # 臂段长度 (从Transformer同步)
        lengths = [0.188, 0.305, 0.1975, 0.181, 0.23]

        # 起始点 (手臂挂载点)
        # current_pos = np.array([0.3, 0.025 if side == 'left' else -0.025, 0.7])
        current_pos = np.array([0.3, 0.25 if side == 'left' else -0.25, 0.7])
        segments = []

        # 建立简单的变换链
        # 简化模型：每个关节绕一个轴旋转，并沿着X轴延伸
        current_rot = R.from_euler('xyz', [0, 0, 0])

        # 我们需要处理7个关节，但只有5个长度段
        # 假设关节与段的对应关系如下 (简化处理):
        # q0, q1 -> 段0; q2, q3 -> 段1...
        # 实际上最简单且相对准确的方法是为每个关节定义一个旋转轴
        axes = [
            [0, 0, 1], [0, 1, 0], [0, 0, 1], [0, 1, 0],
            [0, 0, 1], [0, 1, 0], [0, 0, 1]
        ]

        # 处理关节旋转和段延伸
        for i in range(len(joint_angles)):
            # 1. 更新旋转
            current_rot = current_rot * R.from_rotvec(np.array(axes[i]) * joint_angles[i])

            # 2. 如果这个关节对应一个物理长度段，则延伸
            if i < len(lengths):
                direction = current_rot.apply([1, 0, 0])
                next_pos = current_pos + direction * lengths[i]
                segments.append((current_pos.copy(), next_pos.copy()))
                current_pos = next_pos

        return segments

    def check_collision(self, left_joints, right_joints):
        """检查机器人是否发生碰撞"""
        # 1. 计算左右臂的线段模型
        left_segs = self.get_arm_segments('left', left_joints)
        right_segs = self.get_arm_segments('right', right_joints)

        # 2. 检查左臂与右臂是否碰撞
        for ls in left_segs:
            for rs in right_segs:
                if self._dist_line_to_line(ls[0], ls[1], rs[0], rs[1]) < (self.safety_radius * 2):
                    return True, "Left arm vs Right arm"

        # 3. 检查手臂与身体碰撞
        # 身体简化为一个大圆柱体 (P1, P2 线段)
        body_line = (np.array([0, 0, 0]), np.array([0, 0, 1.0]))
        for side, segs in [('left', left_segs), ('right', right_segs)]:
            for s in segs:
                if self._dist_line_to_line(s[0], s[1], body_line[0], body_line[1]) < self.safety_radius:
                    return True, f"{side} arm vs Body"

        return False, ""

def move_arm_relative(robot, robot_controller, arm_side, delta_position, delta_orientation=None):
    """相对移动机械臂末端 - 使用DELTA_POSE控制模式
    Args:
        arm_side: 'left' 或 'right'
        delta_position: [dx, dy, dz] 相对位置变化（米）
        delta_orientation: [droll, dpitch, dyaw] 相对姿态变化（弧度，可选）
    """
    if delta_orientation is None:
        delta_orientation = [0.0, 0.0, 0.0]

    # 将欧拉角转换为四元数
    roll, pitch, yaw = delta_orientation
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    # 构建相对位姿字典（使用很小的数值作为相对变化）
    pose_dict = {
        'x': delta_position[0],
        'y': delta_position[1],
        'z': delta_position[2],
        'qx': qx,
        'qy': qy,
        'qz': qz,
        'qw': qw
    }

    # 设置控制组
    control_group = [f"{arm_side}_arm"]

    # 使用DELTA_POSE模式进行相对控制
    # 注意：DELTA_POSE模式下，实际实现可能需要使用轨迹跟踪控制
    print(f"{arm_side}臂相对移动: 位置变化={delta_position}m, 姿态变化={delta_orientation}rad")

    numDelta = 10

    for _ in range(numDelta):
        # 由于DELTA_POSE是控制模式，我们需要通过轨迹跟踪控制来实现
        arm_states, _ = robot.arm_joint_states()
        head_states, _ = robot.head_joint_states()
        waist_states, _ = robot.waist_joint_states()
        assert arm_states and head_states and waist_states

        # 构建完整的机器人状态（参考SDK文档的格式）
        robot_states = {
            "head": head_states,
            "waist": waist_states,
            "arm": arm_states  # 14个关节
        }

        # 构建6维动作数据（位置3 + 姿态3）
        action_data = delta_position + delta_orientation
        action_data = [float(v) / numDelta for v in action_data]
        robot_actions = [{
            f"{arm_side}_arm": {
                "action_data": action_data,
                "control_type": "DELTA_POSE"
            }
        }]

        # 使用轨迹跟踪控制
        robot_controller.trajectory_tracking_control(
            int(time.time() * 1e9),
            robot_states,
            robot_actions,
            "base_link",
            # 1.0  # 较短的执行时间
            0.02  # 较短的执行时间
        )
        time.sleep(0.02)

def run_trajectory(robot, robot_controller, path_nodes, side, delta_time):
    try:
        for node in path_nodes:
            # 获取当前实时状态
            arm_states, _ = robot.arm_joint_states()
            head_states, _ = robot.head_joint_states()
            waist_states, _ = robot.waist_joint_states()
            assert arm_states
            assert head_states
            assert waist_states

            robot_states = {
                "head": head_states,
                "waist": waist_states,
                "arm": arm_states,
            }

            # 构建关节空间控制动作
            # 假设 SDK 的 trajectory_tracking_control 在 JOINT 模式下
            # action_data 接受关节目标值
            arm_key = f"{side}_arm"
            robot_actions = [{
                arm_key: {
                    "action_data": node,
                    "control_type": "ABS_JOINT"
                }
            }]

            print(robot_states)
            print(robot_actions)
            # 发送控制命令
            robot_controller.trajectory_tracking_control(
                int(time.time() * 1e9),
                robot_states,
                robot_actions,
                "base_link",
                delta_time  # 每个点执行时间 s
            )
            time.sleep(delta_time)
        print(f"{side}臂轨迹执行完毕", "success")
    except Exception as traj_e:
        print(f"轨迹执行错误: {traj_e}")


def main():
    # 初始化机器人和相机
    robot = Robot()
    camera = Camera(["hand_left", "hand_right", "head", "head_depth", "head_center_fisheye"])
    robot_controller = RobotController()
    slam = Slam()

    # 坐标转换处理器
    transformer = RobotCoordinateTransformer()
    # 碰撞检测处理器
    collision_checker = CollisionChecker()
    # 轨迹规划器
    planner = TrajectoryPlanner(transformer, collision_checker)
    while True:
        left_j0 = robot.arm_joint_states()[0][:7]
        right_j0 = robot.arm_joint_states()[0][7:14]
        if left_j0 and right_j0:
            break
    left_p0 = transformer.compute_arm_fk('left', left_j0)
    right_p0 = transformer.compute_arm_fk('right', right_j0)
    print(left_j0, right_j0)
    print(left_p0, right_p0)
    # planner.plan_path('left', left_j0, [0.7, -0.565, 0.72], right_j0)

    def Left_Move(direction):
        right_j0 = robot.arm_joint_states()[0][7:14]
        assert right_j0
        left_j0 = robot.arm_joint_states()[0][:7]
        assert left_j0
        left_p0 = transformer.compute_arm_fk('left', left_j0)
        assert len(left_p0) == 3
        print(left_j0, left_p0)
        if direction in ["PX", "NX"]:
            new_left_p0 = left_p0
            new_left_p0[0] += 0.05 if direction == "PX" else -0.05
        elif direction in ["PY", "NY"]:
            new_left_p0 = left_p0
            new_left_p0[1] += 0.05 if direction == "PY" else -0.05
        elif direction in ["PZ", "NZ"]:
            new_left_p0 = left_p0
            new_left_p0[2] += 0.05 if direction == "PZ" else -0.05
        else:
            raise RuntimeError(f"Unknown direction: {direction}")
        paths = planner.plan_path('left', left_j0, left_p0, right_j0)
        assert paths
        print("Paths: ", paths)
        r = input("Press 'yes' to continue")
        if r != "yes":
            return
        run_trajectory(robot, robot_controller, paths, "left")

    def record(timeout=5, save=None):
        t0 = time.time()
        data = []
        while True:
            left_j0 = robot.arm_joint_states()[0][:7]
            data.append(left_j0)
            if time.time() - t0 > timeout:
                break
            time.sleep(0.02)
        if save is not None:
            with open(save, "w+") as f:
                json.dump({"data": data}, f)
        return data

    from IPython import embed
    embed()

main()
