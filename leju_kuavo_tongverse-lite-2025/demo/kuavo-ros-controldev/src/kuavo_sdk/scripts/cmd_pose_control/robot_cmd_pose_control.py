#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from geometry_msgs.msg import Twist
import argparse

def publish_cmd_pose():

    # 创建一个发布者，发布到 /cmd_pose 话题
    cmd_pose_pub = rospy.Publisher('/cmd_pose', Twist, queue_size=10)

    pose_optiops = {
    1:[1.0, 0.0, 0.0, 0.0],   #前三个单位是米，第四个单位是弧度
    2:[-1.0, 0.0, 0.0, 0.0],
    3:[0.0, 1.0, 0.0, 0.0],
    4:[0.0, -1.0, 0.0, 0.0],
    5:[0.0, 0.0, 0.0, 1.57],
    6:[0.0, 0.0, 0.0, -1.57]
    }

    # 解析命令行参数  
    parser = argparse.ArgumentParser(description="选择不同的 pose")
    parser.add_argument("--pose_id", type=int, choices=[1, 2, 3, 4, 5, 6], required=True, help="选择 pose 的 ID (1-6)")
    args = parser.parse_args()

    # 根据传入的 pose_id 选择对应的 pose
    pose = pose_optiops[args.pose_id]


    # 创建Twist消息对象
    cmd_pose_msg = Twist()

    # 设置位置指令
    cmd_pose_msg.linear.x = pose[0]  # 基于当前位置的 x 方向值 (m)
    cmd_pose_msg.linear.y = pose[1]  # 基于当前位置的 y 方向值 (m)
    cmd_pose_msg.linear.z = pose[2]  # 增量高度 (m)
    cmd_pose_msg.angular.z = pose[3]  # yaw方向速度 (radian/s)

    # 未使用的字段设置为0
    cmd_pose_msg.angular.x = 0.0
    cmd_pose_msg.angular.y = 0.0

    # 确保发布者连接成功
    rospy.sleep(1)

    # 发布消息
    cmd_pose_pub.publish(cmd_pose_msg)
    rospy.loginfo(f"Published cmd_pose: linear.x={cmd_pose_msg.linear.x}, linear.y={cmd_pose_msg.linear.y}, linear.z={cmd_pose_msg.linear.z}, angular.z={cmd_pose_msg.angular.z}")


if __name__ == '__main__':
    try:
        # 初始化ROS节点
        rospy.init_node('cmd_pose_publisher', anonymous=True)
        publish_cmd_pose()
    except rospy.ROSInterruptException:
        pass