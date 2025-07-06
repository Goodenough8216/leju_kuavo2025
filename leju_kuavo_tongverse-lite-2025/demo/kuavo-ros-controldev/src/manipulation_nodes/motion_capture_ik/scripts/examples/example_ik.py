#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import time
from motion_capture_ik.msg import twoArmHandPoseCmd, ikSolveParam
from motion_capture_ik.msg import twoArmHandPose

# decide use custom ik param or not
use_custom_ik_param = True
# joint angles as initial guess for ik
joint_angles_as_q0 = False # True for custom initial guess, False for default initial guess
# ik solver param
ik_solve_param = ikSolveParam()
# snopt params
ik_solve_param.major_optimality_tol = 1e-3
ik_solve_param.major_feasibility_tol = 1e-3
ik_solve_param.minor_feasibility_tol = 1e-3
ik_solve_param.major_iterations_limit = 100
# constraint and cost params
ik_solve_param.oritation_constraint_tol= 1e-3
ik_solve_param.pos_constraint_tol = 1e-3 # 0.001m, work when pos_cost_weight==0.0
ik_solve_param.pos_cost_weight = 0.0 # If U need high accuracy, set this to 0.0 !!!


def kuavo_ik_result_callback(msg):
    print("************ik result***********")
    print(f"left_joint_angles: {msg.left_pose.joint_angles}")
    print(f"right_joint_angles: {msg.right_pose.joint_angles}")

if __name__ == "__main__":
    rospy.init_node("example_ik_node", anonymous=True)
    pub = rospy.Publisher('/ik/two_arm_hand_pose_cmd', twoArmHandPoseCmd, queue_size=10)
    ik_result_sub = rospy.Subscriber("/ik/result", twoArmHandPose, kuavo_ik_result_callback)
    time.sleep(1.0)

    eef_pose_msg = twoArmHandPoseCmd()
    eef_pose_msg.ik_param = ik_solve_param
    eef_pose_msg.use_custom_ik_param = use_custom_ik_param        # # True for custom ik param, False for default ik param  
    eef_pose_msg.joint_angles_as_q0 = joint_angles_as_q0

    # joint_angles_as_q0 为 False 时，这两个参数不会被使用
    eef_pose_msg.hand_poses.left_pose.joint_angles = np.zeros(7)    # rads
    eef_pose_msg.hand_poses.right_pose.joint_angles = np.zeros(7)   # rads

    # 设置左手末端执行器的位置和姿态
    eef_pose_msg.hand_poses.left_pose.pos_xyz =  np.array([0.45,0.25,0.11988012])
    eef_pose_msg.hand_poses.left_pose.quat_xyzw = [0.0,-0.70682518,0.0,0.70738827] # 四元数
    eef_pose_msg.hand_poses.left_pose.elbow_pos_xyz = np.zeros(3) # 设置成 0.0 时,不会被使用

    # 设置右手末端执行器的位置和姿态
    eef_pose_msg.hand_poses.right_pose.pos_xyz =  np.array([0.45,-0.25,0.11988012])
    eef_pose_msg.hand_poses.right_pose.quat_xyzw = [0.0,-0.70682518,0.0,0.70738827] # 四元数
    eef_pose_msg.hand_poses.right_pose.elbow_pos_xyz = np.zeros(3)  # 设置成 0.0 时,不会被使用

    print("publish ik cmd")
    pub.publish(eef_pose_msg)

    rospy.spin()