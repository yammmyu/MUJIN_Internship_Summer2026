import time
import sys
from a2d_sdk.robot import RobotDds as Robot
# 初始化机器人
robot = Robot()
time.sleep(0.5) #等待资源初始化，收到消息
# 获取当前状态
arm_pos, timestamp = robot.arm_joint_states()
head_pos, timestamp = robot.head_joint_states()
# 基础运动控制
robot.move_head([0.5, 0.5]) # 控制头部运动
robot.move_gripper([1, 1]) # 控制夹持器
# 重置到初始位置
robot.reset()
# 关闭连接
robot.shutdown()
sys.exit(0)