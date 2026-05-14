import copy
import threading
import time
import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from a2d_sdk.robot import Slam
from std_msgs.msg import Header

class WheelController(Node):
    def __init__(self, args=None):
        super().__init__('wheel_controller_example')
        self.publisher_ = self.create_publisher(
            TwistStamped,
            '/mbc/wheel_command',
            10)
        self.get_logger().info('Arm Controller Example node started. Publishing a target pose for the left arm.')
        self.shutdown = False
        self.commands = []
        self._agv_angle = 0.0
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    @property
    def agv_angle(self) -> float:
        return self._agv_angle

    def _update_agv_angle(self):
        self._agv_angle = math.degrees(self.slam.get_chassis_position()[0]['agv_angle'])

    def _run(self):
        self.slam = Slam()
        time.sleep(2)
        self.slam.switch_nav_mode(1)

        msg = TwistStamped()
        twist = msg.twist

        # 1. 设置 Header
        msg.header = Header()
        # msg.header.stamp = self.get_clock().now().to_msg()
        # msg.header.frame_id = 'base_link'

        def _reset():
            # 3. 设置目标位姿
            twist.linear.x = 0.0
            twist.linear.y = 0.0
            twist.linear.z = 0.0
            twist.angular.x = 0.0
            twist.angular.y = 0.0
            twist.angular.z = 0.0
            self.publisher_.publish(msg)

        _reset()

        # 4. 发布消息
        while not self.shutdown:
            self._update_agv_angle()
            if not self.commands:
                time.sleep(0.1)
                continue
            target_angle_deg = self.commands.pop(0) % 360  # target angle
            initial_angle_deg = copy.deepcopy(self._agv_angle)
            sign = 1 if target_angle_deg >= initial_angle_deg else -1
            while not self.shutdown:
                self._update_agv_angle()
                current_angle_deg = self._agv_angle
                new_sign = 1 if target_angle_deg >= current_angle_deg else -1
                diff = abs(current_angle_deg % 360 - target_angle_deg)
                if diff <= 5:
                    twist.angular.z = 0.02 * sign
                elif diff <= 10:
                    twist.angular.z = 0.1 * sign
                elif diff <= 20:
                    twist.angular.z = 0.2 * sign
                else:
                    twist.angular.z = 0.4 * sign
                if diff < 1 or new_sign != sign:
                    break
                # msg.header.stamp = self.get_clock().now().to_msg()
                self.publisher_.publish(msg)
                self.get_logger().info(f'[{current_angle_deg}->{target_angle_deg}] Publishing to /mbc/wheel_command: linear={twist.linear}, angular={twist.angular}')
                time.sleep(0.1)
            _reset()
        
        # 销毁定时器，这样指令只发布一次
        # self.timer.cancel()
        # self.get_logger().info('Target pose published once. Shutting down in 2 seconds.')
        # 等待一段时间后关闭节点
    #     self.create_timer(2.0, self.shutdown)
    #
    # def shutdown(self):
    #     self.destroy_node()
    #     rclpy.shutdown()
    #     self.thread.join()
    #     self.thread = None

def main(args=None):
    rclpy.init(args=args)
    wheel_controller = WheelController(args=args)
    thread = threading.Thread(target=rclpy.spin, args=(wheel_controller,), daemon=True)
    thread.start()
    try:
        from IPython import embed

        embed()
    finally:
        wheel_controller.shutdown = True

if __name__ == '__main__':
    main()