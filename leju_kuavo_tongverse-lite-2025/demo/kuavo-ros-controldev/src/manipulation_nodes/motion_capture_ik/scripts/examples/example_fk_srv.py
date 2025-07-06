#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from motion_capture_ik.srv import fkSrv

import numpy as np

def fk_srv_client(joint_angles):
    rospy.wait_for_service('/ik/fk_srv')
    try:
        fk_srv = rospy.ServiceProxy('/ik/fk_srv', fkSrv)
        fk_result = fk_srv(joint_angles)
        print("FK result:", fk_result.success)
        return fk_result.hand_poses
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)


if __name__ == "__main__":
    rospy.init_node("example_fk_srv_node", anonymous=True)
    # 单位：弧度
    joint_angles = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.38, -1.39, -0.29, -0.43, 0.0, -0.17, 0.0]

    # 调用 FK 正解服务
    hand_poses = fk_srv_client(joint_angles)
    if hand_poses is not None:
        print("left eef position:", hand_poses.left_pose.pos_xyz)
        print("\nright eef position: ", hand_poses.right_pose.pos_xyz)
    else:
        print("No hand poses returned")
        
   
