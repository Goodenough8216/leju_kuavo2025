#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    测试右手固定抓取轨迹
"""

import rospy
from humanoid_arm_control.srv import armControl, armControlRequest

TAG_ID = 0
if __name__ == '__main__':
    try:
        rospy.init_node("client_arm_service_demo_node", anonymous=True)
        # 等待 /arm_control 服务启动
        rospy.loginfo("Waiting for service...")
        rospy.wait_for_service('/arm_control')
        arm_control_service = rospy.ServiceProxy('/arm_control', armControl)

        req = armControlRequest()
        req.req = 2
        req.tagid = TAG_ID
        response = arm_control_service(req)  # 向ROS服务发送请求
        rospy.loginfo(f"Response: {response.success}, {response.time}")

    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")
        exit(0)