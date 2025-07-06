#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
from std_msgs.msg import Float64MultiArray

def send_effort_command():
    rospy.init_node('effort_controller_test', anonymous=True)
    
    # 发布到 '/limbs_effort_controller/command' 话题，消息类型为 Float64MultiArray
    pub = rospy.Publisher('/limbs_effort_controller/command', Float64MultiArray, queue_size=10)
    rospy.sleep(1)  # 等待一些时间，确保publisher已初始化
    
    # 创建 Float64MultiArray 消息
    effort_msg = Float64MultiArray()
    
    # 创建一个 effort list，设置所有关节的努力值为 1.0
    effort_values = [100.0] * 26  # 假设有 28 个关节
    effort_msg.data = effort_values

    # 发布 effort 消息
    pub.publish(effort_msg)

if __name__ == '__main__':
    try:
        send_effort_command()
    except rospy.ROSInterruptException:
        pass
