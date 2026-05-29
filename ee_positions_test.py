from a2d_sdk.robot import RobotDds as Robot, RobotController


def get_hand_statuses(robot_controller):
    status = robot_controller.get_motion_status()
    frames = status['frames']

    def _extract(link):
        f    = frames[link]
        pos  = f['position']
        quat = f['orientation']['quaternion']
        return (
            pos['x'],  pos['y'],  pos['z'],
            quat['x'], quat['y'], quat['z'], quat['w'],
        )

    left  = _extract('arm_left_link7')
    right = _extract('arm_right_link7')
    return left, right


if __name__ == '__main__':
    robot = Robot()
    robot.connect()
    controller = robot.robot_controller  # adjust to actual API

    while True:
        left, right = get_hand_statuses(controller)
        print(f"left ee position: {left} | right ee position: {right}")
