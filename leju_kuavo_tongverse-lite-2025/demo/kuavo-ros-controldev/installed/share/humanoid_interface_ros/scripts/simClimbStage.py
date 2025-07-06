#!/usr/bin/env python3

import rospy
from kuavo_msgs.msg import footPose, footPoseTargetTrajectories  # 导入自定义消息类型
import numpy as np

def publish_foot_pose(foot_idx, foot_pose, torso_pose=[0.0]*4):
    # 初始化 ROS 节点
    rospy.init_node('foot_pose_publisher', anonymous=True)
    # 创建发布者，话题为 /humanoid_mpc_foot_pose_target_trajectories
    pub = rospy.Publisher('/humanoid_mpc_foot_pose_target_trajectories', footPoseTargetTrajectories, queue_size=10)

    # 等待一定时间以确保订阅者已经准备好
    rospy.sleep(1)

    # 创建消息实例
    msg = footPoseTargetTrajectories()
    msg.timeTrajectory = [0.1]  # 设置时间轨迹
    msg.footIndexTrajectory = [foot_idx]         # 设置脚索引
    msg.footPoseTrajectory = []  # 初始化脚姿态轨迹

    # 创建脚姿态信息
    foot_pose_msg = footPose()
    foot_pose_msg.footPose = foot_pose      # 设置脚姿态
    foot_pose_msg.torsoPose = torso_pose     # 设置躯干姿态

    # 将脚姿态添加到消息中
    msg.footPoseTrajectory.append(foot_pose_msg)

    # 发布消息
    print(f"{'Left' if foot_idx == 0 else 'Right'} foot move to: {foot_pose}")
    pub.publish(msg)

    # 停止发布一段时间（可根据需要调整）
    rospy.sleep(1.5)

def publish_foot_pose_traj(time_traj, foot_idx_traj, foot_traj, torso_traj):
    num = len(time_traj)
    # 初始化 ROS 节点
    rospy.init_node('foot_pose_publisher', anonymous=True)
    # 创建发布者，话题为 /humanoid_mpc_foot_pose_target_trajectories
    pub = rospy.Publisher('/humanoid_mpc_foot_pose_target_trajectories', footPoseTargetTrajectories, queue_size=10)

    # 等待一定时间以确保订阅者已经准备好
    rospy.sleep(1)

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
    # print(f"{'Left' if foot_idx == 0 else 'Right'} foot move to: {foot_pose}")
    pub.publish(msg)

    # 停止发布一段时间（可根据需要调整）
    rospy.sleep(1.5)

def generate_steps(torso_pos, torso_yaw, foot_bias):
    l_foot_bias = np.array([0, foot_bias, 0])
    r_foot_bias = np.array([0, -foot_bias, 0])
    R_z = np.array([
        [np.cos(torso_yaw), -np.sin(torso_yaw), 0],
        [np.sin(torso_yaw), np.cos(torso_yaw), 0],
        [0, 0, 1]
    ])
    l_foot = torso_pos + R_z.dot(l_foot_bias)
    r_foot = torso_pos + R_z.dot(r_foot_bias)
    return l_foot, r_foot   


def publish_multiple_steps():
    height = 0.0
    body_poses = [
        [0.4, 0.0, 0.1, 0],
        [0.8, 0.0, 0.2, 0],
    ]

    num_steps = 2*len(body_poses)
    time_traj = []
    foot_idx_traj = []
    foot_traj = []
    torso_traj = []
    dt = 0.8
    for i in range(num_steps):
        time_traj.append(dt * (i+1))
        body_pose = body_poses[i//2]
        torso_pos = np.asarray(body_pose[:3])
        torso_yaw = np.radians(body_pose[3])
        l_foot, r_foot = generate_steps(torso_pos, torso_yaw, 0.1)
        l_foot = [*l_foot[:3], torso_yaw]
        r_foot = [*r_foot[:3], torso_yaw]
        if(i%2 == 0):
            foot_idx_traj.append(0)
            foot_traj.append(l_foot)
        else:
            foot_idx_traj.append(1)
            foot_traj.append(r_foot)
        torso_traj.append([*body_pose[:3], torso_yaw])
    print("time_traj:", time_traj)
    print("foot_idx_traj:", foot_idx_traj)
    print("foot_traj:", foot_traj)
    print("torso_traj:", torso_traj)
    publish_foot_pose_traj(time_traj, foot_idx_traj, foot_traj, torso_traj)


if __name__ == '__main__':
    publish_multiple_steps()
