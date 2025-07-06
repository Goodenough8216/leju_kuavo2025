#!/usr/bin/env python3

import rospy
from kuavo_sdk.msg import footPose, footPoseTargetTrajectories  # 导入自定义消息类型
import numpy as np
from utils.sat import RotatingRectangle  # 导入用于碰撞检测的工具类


def get_foot_pose_traj_msg(time_traj, foot_idx_traj, foot_traj, torso_traj):
    """
    创建并返回一个 footPoseTargetTrajectories 消息对象。
    
    参数：
    - time_traj: 时间轨迹列表
    - foot_idx_traj: 脚索引轨迹列表
    - foot_traj: 脚姿态轨迹列表
    - torso_traj: 躯干姿态轨迹列表
    
    返回：
    - footPoseTargetTrajectories 消息对象
    """
    num = len(time_traj)
    msg = footPoseTargetTrajectories()  # 创建消息实例
    msg.timeTrajectory = time_traj  # 设置时间轨迹
    msg.footIndexTrajectory = foot_idx_traj  # 设置脚索引
    msg.footPoseTrajectory = []  # 初始化脚姿态轨迹

    for i in range(num):
        foot_pose_msg = footPose()  # 创建脚姿态信息
        foot_pose_msg.footPose = foot_traj[i]  # 设置脚姿态
        foot_pose_msg.torsoPose = torso_traj[i]  # 设置躯干姿态
        msg.footPoseTrajectory.append(foot_pose_msg)  # 将脚姿态添加到消息中

    return msg

def generate_steps(torso_pos, torso_yaw, foot_bias):
    """
    根据躯干位置和偏航角生成左右脚的位置。
    
    参数：
    - torso_pos: 躯干位置
    - torso_yaw: 躯干偏航角（弧度）
    - foot_bias: 脚偏移量
    
    返回：
    - 左脚和右脚的位置
    """
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

def get_multiple_steps_msg(body_poses, dt, is_left_first=True, collision_check=True):
    """
    生成多步步态消息。
    
    参数：
    - body_poses: 身体姿态列表
    - dt: 时间间隔
    - is_left_first: 是否左脚先行
    - collision_check: 是否进行碰撞检测
    
    返回：
    - footPoseTargetTrajectories 消息对象
    """
    num_steps = 2 * len(body_poses)  # 每个身体姿态对应两步
    time_traj = []  # 时间轨迹
    foot_idx_traj = []  # 脚索引轨迹
    foot_traj = []  # 脚姿态轨迹
    torso_traj = []  # 躯干姿态轨迹

    # 初始化左右脚的旋转矩形用于碰撞检测
    l_foot_rect_last = RotatingRectangle(center=(0, 0.1), width=0.24, height=0.1, angle=0)
    r_foot_rect_last = RotatingRectangle(center=(0, -0.1), width=0.24, height=0.1, angle=0)

    for i in range(num_steps):
        time_traj.append(dt * (i + 1))  # 添加时间点
        body_pose = body_poses[i // 2]  # 获取当前身体姿态
        torso_pos = np.asarray(body_pose[:3])  # 躯干位置
        torso_yaw = np.radians(body_pose[3])  # 躯干偏航角（转换为弧度）

        l_foot, r_foot = generate_steps(torso_pos, torso_yaw, 0.1)  # 生成左右脚位置
        l_foot = [*l_foot[:3], torso_yaw]  # 左脚姿态
        r_foot = [*r_foot[:3], torso_yaw]  # 右脚姿态

        if collision_check and i % 2 == 0:  # 每两步进行一次碰撞检测
            l_foot_rect_next = RotatingRectangle(center=(l_foot[0], l_foot[1]), width=0.24, height=0.1, angle=torso_yaw)
            r_foot_rect_next = RotatingRectangle(center=(r_foot[0], r_foot[1]), width=0.24, height=0.1, angle=torso_yaw)
            l_collision = l_foot_rect_next.is_collision(r_foot_rect_last)
            r_collision = r_foot_rect_next.is_collision(l_foot_rect_last)

            if l_collision and r_collision:
                print("\033[91m[Error] Detect collision, Please adjust your body_poses input!!!\033[0m")
                break
            elif l_collision:
                print("\033[92m[Info] Left foot is in collision, switch to right foot\033[0m")
                is_left_first = False
            else:
                is_left_first = True

            l_foot_rect_last = l_foot_rect_next
            r_foot_rect_last = r_foot_rect_next

        if i % 2 == 0:  # 偶数步
            if is_left_first:
                foot_idx_traj.append(0)  # 左脚
                foot_traj.append(l_foot)
            else:
                foot_idx_traj.append(1)  # 右脚
                foot_traj.append(r_foot)
        else:  # 奇数步
            if is_left_first:
                foot_idx_traj.append(1)  # 右脚
                foot_traj.append(r_foot)
            else:
                foot_idx_traj.append(0)  # 左脚
                foot_traj.append(l_foot)

        torso_traj.append([*body_pose[:3], torso_yaw])  # 添加躯干姿态

    print("time_traj:", time_traj)
    print("foot_idx_traj:", foot_idx_traj)
    print("foot_traj:", foot_traj)
    print("torso_traj:", torso_traj)

    return get_foot_pose_traj_msg(time_traj, foot_idx_traj, foot_traj, torso_traj)


if __name__ == '__main__':
    # 初始化 ROS 节点
    rospy.init_node('foot_pose_publisher', anonymous=True)

    # 创建发布者，话题为 /humanoid_mpc_foot_pose_target_trajectories
    pub = rospy.Publisher('/humanoid_mpc_foot_pose_target_trajectories', footPoseTargetTrajectories, queue_size=10)

    # 等待一定时间以确保订阅者已经准备好
    rospy.sleep(1)

    is_left_first_default = True  # 缺省左脚先行
    collision_check = True  # 缺省开启碰撞检测

    # body_poses基于局部坐标系给定，每一个身体姿态对应两步到达
    dt = 0.4  # 迈一步的时间间隔，腾空相和支撑相时间占比各dt/2

    # 一次完整的步态Mode序列为:[SS FS SS SF SS]或者[SS SF SS FS SS]
    # body_pose： [x(m), y(m), z(m), yaw(deg)]
    body_poses = [
        [0.1, -0.0, 0, -30],
        [0.1, -0.1, 0, -0],
        [0.2, -0.1, 0, -30],
        [0.3, 0.0, 0, -0],
        [0.4, 0.0, 0, -30],
        [0.5, 0.0, 0, 0],
    ]

    # 生成步态消息并发布
    msg = get_multiple_steps_msg(body_poses, dt, is_left_first_default, collision_check)
    pub.publish(msg)
