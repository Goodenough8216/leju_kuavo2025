#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from std_msgs.msg import Float32MultiArray
from general_ik.srv import fkSrv

import numpy as np

def fk_srv_client(joint_angles):
    rospy.wait_for_service('/ik/fk_srv')
    try:
        fk_srv = rospy.ServiceProxy('/ik/fk_srv', fkSrv)
        fk_result = fk_srv(joint_angles)
        return fk_result.hand_poses
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)


if __name__ == "__main__":
    rospy.init_node("test_fk_srv", anonymous=True)
    num_dof = 14
    joint_angles = np.zeros(num_dof)

    hand_poses = fk_srv_client(joint_angles)
    if hand_poses is not None:
        print(hand_poses)
    else:
        print("No hand poses returned")
        
   
