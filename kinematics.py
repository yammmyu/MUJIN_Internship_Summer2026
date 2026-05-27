import numpy as np
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

    def compute_arm_fk(self, side, joint_angles):
        """
        计算手臂末端在基座坐标系下的位置 (简易版)
        joint_angles: [q1, q2, q3, q4, q5, q6, q7]
        """
        # 初始位置 (手臂挂载点)
        p = np.array([0.3, 0.025 if side == 'left' else -0.025, 0.7])

        # 简化正运动学：将各段长度沿当前旋转方向累加
        # 注意：这里是简化模型，实际应使用 URDF 的完整 DH 参数或变换矩阵链
        current_rot = R.from_euler('xyz', [0, 0, 0])

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
