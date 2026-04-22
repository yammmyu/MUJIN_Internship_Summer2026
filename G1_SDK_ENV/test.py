#!/usr/bin/env python3
import time
from a2d_sdk.robot import RobotController, RobotDds
def poll_state(getter, length, name, timeout=2.0, interval=0.1):
    deadline = time.time() + timeout
    last_vals = None
    while time.time() < deadline:
        vals, _ = getter()
        last_vals = vals
        # 获取长度
        try:
            actual_len = len(vals)
        except Exception:
            actual_len = None

        # 判断每个元素是否可转 float
        all_numeric = (actual_len == length) and all(_is_number(v) for v in vals)
        if all_numeric:
            print(f"✅ {name} 就绪")
            return list(vals)
        time.sleep(interval)
    raise RuntimeError(f"{name} 在 {timeout}s 内未就绪，最后 vals={last_vals!r}")
def _is_number(x):
    try:
        float(x)
        return True
    except Exception:
        return False
def execute(rc: RobotController, action: dict):
    """
    调用 SDK 下发相对位姿轨迹控制命令。
    """
    # 定义机器人状态字典
    robot_states = {
        "head": action["head_joint_states"],
        "waist": action["waist_joint_states"],
        "arm": action["arm_joint_states"],
    }
    # 定义机器人动作列表
    robot_actions = [
        {
            "left_arm": {"action_data": delta[:6], "control_type": "DELTA_POSE"},
            "right_arm": {"action_data": delta[6:12], "control_type": "DELTA_POSE"},
        }
        for delta in action["arm_cmd"]
    ]
    # 调用 SDK 下发相对位姿轨迹控制命令
    rc.trajectory_tracking_control(
        infer_timestamp = action["observation_timestamp"],#必须有的时间
        robot_states = robot_states,# 必须的参考状态，有了更好
        robot_actions = robot_actions,
        robot_link = "base_link",#基础坐标系
        trajectory_reference_time = 1.0,#运行的参考时间，越小运行的越快
    )
def check_motion_status(rc: RobotController):
    """检查运动控制状态"""
    try:
        motion_status = rc.get_motion_status()
        if motion_status:
            mode = motion_status.get('mode', {})
            error = motion_status.get('error', {})
            
            print(f"运动控制状态:")
            print(f"  模式: {mode.get('name', 'Unknown')} (值: {mode.get('value', -1)})")
            print(f"  错误: {error.get('message', '无')} (代码: {error.get('code', 0)})")
            
            if error.get('has_error', False):
                print(f"⚠️  运动控制存在错误: {error.get('message', 'Unknown error')}")
                return False
                
            # 检查是否在正确的模式下
            if mode.get('value') != 1:  # 1 应该是 SERVO 模式
                print(f"⚠️  机器人不在SERVO模式，当前模式: {mode.get('name', 'Unknown')}")
                return False
                
            return True
        else:
            print("⚠️  无法获取运动控制状态")
            return False
    except Exception as e:
        print(f"⚠️  检查运动状态失败: {e}")
        return False

def main():
    rc = RobotController()
    rd = RobotDds()
    time.sleep(2.0)
    
    print(">>> 检查机器人运动状态...")
    if not check_motion_status(rc):
        print("⚠️  机器人状态异常，尝试重置...")
        rd.reset()
        time.sleep(3.0)
        if not check_motion_status(rc):
            print("❌ 机器人状态仍然异常，退出程序")
            # rd.shutdown()
            # return
    
    head_states = poll_state(rd.head_joint_states, length=2, name="head_joint_states")
    waist_states = poll_state(rd.waist_joint_states, length=2, name="waist_joint_states")
    arm_states = poll_state(rd.arm_joint_states, length=14, name="arm_joint_states")
    print(">>> 所有传感器数据就绪，开始下发相对位姿控制")
    
    # 打印初始状态
    print(f">>> 初始手臂状态: {arm_states[:7]} (左臂前7关节), {arm_states[7:14] if len(arm_states) >= 14 else arm_states[:7]} (右臂)")
    
    try:
        step = 0
        while True:
            step += 1
            ts_ns = int(time.time() * 1e9)
            
            # 增大控制幅度，并添加周期性变化
            amplitude = 0.3  # 增大到0.3米
            if step % 4 == 1:
                left_delta = [amplitude, 0, 0, 0, 0, 0]  # 左臂向前
                right_delta = [amplitude, 0, 0, 0, 0, 0]  # 右臂向前
            elif step % 4 == 2:
                left_delta = [-amplitude, 0, 0, 0, 0, 0]  # 左臂向后
                right_delta = [-amplitude, 0, 0, 0, 0, 0]  # 右臂向后
            elif step % 4 == 3:
                left_delta = [0, amplitude, 0, 0, 0, 0]  # 左臂向左
                right_delta = [0, -amplitude, 0, 0, 0, 0]  # 右臂向右
            else:
                left_delta = [0, -amplitude, 0, 0, 0, 0]  # 左臂向右
                right_delta = [0, amplitude, 0, 0, 0, 0]  # 右臂向左
            
            action = {
                "observation_timestamp": ts_ns,
                "head_joint_states": head_states,
                "waist_joint_states": waist_states,
                "arm_joint_states": arm_states,
                "arm_cmd": [
                    left_delta + right_delta  # 合并左右臂控制
                ],
            }
            
            print(f">>> [{time.strftime('%H:%M:%S')}] 第{step}步: 左臂{left_delta[:3]}, 右臂{right_delta[:3]}")
            execute(rc, action)
            
            # 检查运动状态
            if step % 5 == 0:  # 每5步检查一次状态
                check_motion_status(rc)
            
            time.sleep(2.0)  # 增加等待时间，让机器人有足够时间运动
            
            # 更新状态用于下一步控制
            try:
                new_arm_states, _ = rd.arm_joint_states()
                if new_arm_states and len(new_arm_states) >= 14:
                    arm_states = new_arm_states
                    print(f">>> 当前手臂状态: 左臂{arm_states[:3]}, 右臂{arm_states[7:10]}")
            except Exception as e:
                print(f"⚠️  更新手臂状态失败: {e}")
                
    except KeyboardInterrupt:
        print("\n>>> 收到中断，退出。")
    finally:
        print(">>> 关闭机器人连接...")
        rd.shutdown()
if __name__ == "__main__":
    main()