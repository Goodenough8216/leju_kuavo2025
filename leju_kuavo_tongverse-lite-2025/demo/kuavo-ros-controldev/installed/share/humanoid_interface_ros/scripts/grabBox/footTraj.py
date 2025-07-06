#!/usr/bin/env python3

import rospy
from kuavo_msgs.msg import footPose, footPoseTargetTrajectories  # 导入自定义消息类型
import numpy as np


def get_foot_pose_traj_msg(time_traj, foot_idx_traj, foot_traj, torso_traj):
    num = len(time_traj)

    # 创建消息实例
    msg = footPoseTargetTrajectories()
    msg.timeTrajectory = time_traj  # 设置时间轨迹
    msg.footIndexTrajectory = foot_idx_traj         # 设置脚索引
    msg.footPoseTrajectory = []  # 初始化脚姿态轨迹

    for i in range(num):
        # 创建脚姿态信息
        foot_pose_msg = footPose()
        foot_pose_msg.footPose = foot_traj[i]      # 设置脚姿态
        foot_pose_msg.torsoPose = torso_traj[i]    # 设置躯干姿态

        # 将脚姿态添加到消息中
        msg.footPoseTrajectory.append(foot_pose_msg)

    # 发布消息
    return msg

def generate_steps(torso_pos, torso_yaw, foot_bias):
    l_foot_bias = np.array([0, foot_bias, -torso_pos[2]])
    r_foot_bias = np.array([0, -foot_bias, -torso_pos[2]])
    R_z = np.array([
        [np.cos(torso_yaw), -np.sin(torso_yaw), 0],
        [np.sin(torso_yaw), np.cos(torso_yaw), 0],
        [0, 0, 1]
    ])
    l_foot = torso_pos + R_z.dot(l_foot_bias)
    r_foot = torso_pos + R_z.dot(r_foot_bias)
    return l_foot, r_foot   

