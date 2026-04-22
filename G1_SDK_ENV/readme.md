# 智元G1 机器人GDK 安装手顺

## 
```bash
sudo apt install iproute2

mkdir G1_SDK_ENV
cd G1_SDK_ENV
```
## 编辑 requirements_gui.txt 文件
```bash
numpy
protobuf==3.12.4
ruckig==0.14.0
opencv-python==4.10.0.84
scipyzmq==0.0.0
pyzmq==26.2.0
matplotlib
```

```bash

# conda create -n GDK python=3.10.12
# conda create -n GDK python=3.11
conda create -n GDK python=3.10
conda activate GDK
pip install -r ./requirements_gui.txt


curl -sSL http://10.13.125.76:8849/install.sh | bash

## 下面环境仅为使用 pink IK，如非必须可以跳过
pip install pinocchio
pip install "numpy<2"
pip install --force-reinstall scipy
## 上面环境仅为使用 pink IK，如非必须可以跳过

cd a2d_sdk
pip install -r requirements_gdk.txt
source env.sh
# 切换模式
robot-service -s -c ./conf/compressed_image.pbtxt
robot-service -s -c ./conf/copilot.pbtxt

robot-service -s -c ./conf/hybrid_deploy_develop.pbtxt

robot-service -s -c ./conf/idle.pbtxt
robot-service -s -c ./conf/vr.pbtxt
## /hand_left_color /hand_right_color /head_color /head_depth

# 图像查看
rviz2 -d rviz/hybrid_deploy.rviz
# 查看启动模式

ros2 topic echo --once /launcher/scene_mode
```

# 机器人快速工具
```bash
robot-controller
## 头部
he 0,0
## 腰部
wa 25,30
## 夹爪
gr 0,0
## 复位
re



# ros2 topic 控制接口
## 头部yaw控制
ros2 topic pub /wbc/joint_position_control genie_msgs/msg/JointPositionControl "{header: {stamp: {sec: 0, nanosec: 0}, frame_id: ''}, lifetime: 5.0, control_group: 1, head_yaw_joint_position: 0.5}"

## 腰部升降控制
ros2 topic pub /wbc/joint_position_control genie_msgs/msg/JointPositionControl "{header: {stamp: {sec: 0, nanosec: 0}, frame_id: ''}, lifetime: 5.0, control_group: 16, waist_lift_joint_position: 0.3}"

## 腰部俯仰控制
ros2 topic pub /wbc/joint_position_control genie_msgs/msg/JointPositionControl "{header: {stamp: {sec: 0, nanosec: 0}, frame_id: ''}, lifetime: 5.0, control_group: 32, waist_pitch_joint_position: 0.5}"

# 控制右夹爪
ros2 topic pub /wbc/joint_position_control genie_msgs/msg/JointPositionControl "{header: {stamp: {sec: 0, nanosec: 0}, frame_id: ''}, lifetime: 5, control_group: 256, right_tool_joint_positions: [0.5]}"

# 使用末端位置控制左右手臂
ros2 topic pub -r 50 /wbc/end_effector_pose_control genie_msgs/msg/EndEffectorPoseControl "{
  header: {
    stamp: {sec: 0, nanosec: 0},
    frame_id: 'base_link'
  },
  lifetime: 10,
  control_group: 12,
  left_end_effector_pose: {
    position: {x: 0.6706307451432025, y: 0.29775424009454377, z: 0.7109336265657352},
    orientation: {x: -0.20724560103774245, y: 0.7973315022771579, z: -0.5668400344607115, w: -0.002027722746358685}
  },
  right_end_effector_pose: {
    position: {x: 0.6298673206706626, y: -0.34436789386677213, z: 0.5942527678756625},
    orientation: {x: -0.7895529235635838, y: 0.15635523877588278, z: -0.1802825440499219, w: 0.5653825470514844}
  }
}"
```
# SLAM、地图管理与导航接口
```bash
ros2 service call /hal/nav/move_relative genie_msgs/srv/NavigatePose "{
  header: {
    stamp: {sec: 0, nanosec: 0},
    frame_id: 'base_link'
  },
  x: 0.0,
  y: 0.01,
  theta: 0.0
}"

#原地逆时针旋转 90 度 (约 1.57 弧度):
ros2 service call /hal/nav/move_relative genie_msgs/srv/NavigatePose "{header: {stamp: {sec: 0, nanosec: 0}, frame_id: 'base_link'}, x: 0.0, y: 0.0, theta: 1.57}"

ros2 service call /hal/switch_nav_mode genie_msgs/srv/AGVModeControl "{header: {stamp: {sec: 0, nanosec: 0}, frame_id: ''}, mode: 1}"

ls -l /dev/shm/
rm /dev/shm/fastrtps*
```

# 使用python脚本(ros2 topic 控制)控制机械臂
```bash
cd /opt/workspace/G1_SDK_ENV/
conda activate GDK
source /opt/ros/humble/setup.bash
source ./a2d_sdk/env.sh
python /opt/workspace/G1_SDK_ENV/control_arm_example.py
```

# 使用python脚本控制/监视机器人 SDK方式。
```bash
conda activate GDK
source /opt/ros/humble/setup.bash
source ./a2d_sdk/env.sh
python /opt/workspace/G1_SDK_ENV/robot_control_gui.py

```

## 常用End Effactor Pose
```bash
#初始位置1
Left:  (0.4,0.4,0.6) (0.0,0.0,0.0)
Right: (0.4,-0.4,0.6) (0.0,0.0,0.0)
#初始位置2
Left:  (0.4,0.4,0.6) (0.0.0,0.0,-1.0)
Right: (0.4,-0.4,0.6) (-1.5,0.0,-1.0)

Right: (1.0,-0.3,0.9) (-1.5,0.0,-1.0)

```