import rospy
from kuavo_sdk.msg import robotHandPosition


def publish_controlEndHand(hand_traj):
    """
    控制机器人的手部动作。

    参数:
    hand_traj (list): 包含左手和右手位置的列表，前6个元素为左手，后6个元素为右手。

    返回:
    bool: 服务调用结果，成功返回True，失败返回False。
    """
    try:
        # 创建Publisher对象
        pub = rospy.Publisher('control_robot_hand_position', robotHandPosition, queue_size=10)
        rospy.sleep(0.5)  # 确保Publisher注册
        # 创建消息对象
        msg = robotHandPosition()
        # 设置左手和右手的位置
        msg.left_hand_position = hand_traj[0:6]
        msg.right_hand_position = hand_traj[6:]

        # 发布消息
        pub.publish(msg)

    except rospy.ServiceException as e:
        # 记录错误日志
        rospy.logerr(f"controlEndHand 服务调用失败: {e}")
        return False

def main():
    # 初始化ROS节点
    rospy.init_node('robot_hand_controller')

    # 示例手部轨迹数据，假设每只手有6个位置参数
    hand_traj = [0, 0, 0, 0, 0, 0,  # 左手位置
                 20, 20, 20, 20, 20, 20]  # 右手位置

    publish_controlEndHand(hand_traj)


if __name__ == "__main__":
    main()