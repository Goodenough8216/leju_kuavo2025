#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist


def publish_cmd_vel(cmd_vel_pub):
    """
    发布 /cmd_vel 话题的控制指令。
    """
    # 设置发布频率
    rate = rospy.Rate(10)  # 10 Hz

    # 创建Twist消息对象
    cmd_vel_msg = Twist()

    # 设置速度指令
    cmd_vel_msg.linear.x = 0.2  # x方向速度 (m/s)
    cmd_vel_msg.linear.y = 0.0  # y方向速度 (m/s)
    cmd_vel_msg.linear.z = 0.0  # 增量高度 (m)
    cmd_vel_msg.angular.z = 0.0  # yaw方向速度 (radian/s)

    # 未使用的字段设置为0
    cmd_vel_msg.angular.x = 0.0
    cmd_vel_msg.angular.y = 0.0

    rospy.loginfo("Publishing /cmd_vel message...")

    while not rospy.is_shutdown():
        # 发布消息
        cmd_vel_pub.publish(cmd_vel_msg)
        # 打印当前发布的速度指令
        rospy.loginfo(f"Published cmd_vel: linear.x={cmd_vel_msg.linear.x}, linear.y={cmd_vel_msg.linear.y}, linear.z={cmd_vel_msg.linear.z}, angular.z={cmd_vel_msg.angular.z}")
        # 等待下一个发布周期
        rate.sleep()

def main():
    # 初始化ROS节点
    rospy.init_node('cmd_vel_publisher', anonymous=True)

    # 创建一个发布者，发布到 /cmd_vel 话题
    cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

    # 调用发布函数
    publish_cmd_vel(cmd_vel_pub)


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass