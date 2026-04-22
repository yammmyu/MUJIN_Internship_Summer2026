import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, Point, Quaternion
from genie_msgs.msg import EndEffectorPoseControl
from std_msgs.msg import Header

class ArmController(Node):
    def __init__(self):
        super().__init__('arm_controller_example')
        self.publisher_ = self.create_publisher(
            EndEffectorPoseControl,
            '/wbc/end_effector_pose_control',
            10)
        self.timer = self.create_timer(1.0, self.publish_pose)
        self.get_logger().info('Arm Controller Example node started. Publishing a target pose for the left arm.')

    def publish_pose(self):
        msg = EndEffectorPoseControl()

        # 1. 设置 Header
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'

        # 2. 设置控制参数
        msg.lifetime = 10.0  # 指令生命周期
        msg.control_group = 4  # 4 代表控制左臂

        # 3. 设置左臂的目标位姿
        target_pose = Pose()
        # 设置目标位置 (x, y, z) - 您可以修改这些值
        target_pose.position = Point(x=0.5, y=0.3, z=0.6)
        # 设置目标姿态 (四元数 x, y, z, w) - 您可以修改这些值
        target_pose.orientation = Quaternion(x=0.0, y=0.707, z=0.0, w=0.707)
        msg.left_end_effector_pose = target_pose

        # 4. 发布消息
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing to /wbc/end_effector_pose_control: control_group={msg.control_group}')
        
        # 销毁定时器，这样指令只发布一次
        self.timer.cancel()
        self.get_logger().info('Target pose published once. Shutting down in 2 seconds.')
        # 等待一段时间后关闭节点
        self.create_timer(2.0, self.shutdown)

    def shutdown(self):
        self.destroy_node()
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    arm_controller = ArmController()
    rclpy.spin(arm_controller)
    # 在节点销毁后，程序会自动退出

if __name__ == '__main__':
    main()