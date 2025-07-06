#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
from kuavo_msgs.msg import footPose, footPoseTargetTrajectories

class StairClimbingPlanner:
    def __init__(self):
        self.dt = 0.8  # 步态周期
        self.foot_width = 0.10  # 宽
        self.step_height = 0.13  # 台阶高度
        self.step_length = 0.32  # 台阶长度
        self.total_step = 0  # 总步数
        
    def move_to_down_stairs(self, step = 3, current_torso_pos = np.array([0.0, 0.0, 0.0, 0.0]), current_foot_pos = np.array([0.0, 0.0, 0.0])):
        time_traj = []
        foot_idx_traj = []
        foot_traj = []
        torso_traj = []
        
        # current_torso_pos = np.array([0.0, 0.0, 0.0, 0.0])
        # current_foot_pos = np.array([0.0, 0.0, 0.0])
        step_length = 0.11
        for i in range(step):
            self.total_step += 1
            time_traj.append(self.total_step*self.dt)
            # 左右脚交替
            is_left_foot = ((self.total_step -1) % 2 == 0)
            foot_idx_traj.append(0 if is_left_foot else 1)
            if i == 0:
                current_torso_pos[0] += step_length/2
                current_foot_pos[0] = current_torso_pos[0] +step_length/2  # 脚掌相对躯干前移
            elif i == step - 1:
                current_torso_pos[0] += step_length/2
                current_foot_pos[0] = current_torso_pos[0]
            else:
                current_torso_pos[0] += step_length
                current_foot_pos[0] = current_torso_pos[0] +step_length/2  # 脚掌相对躯干前移
            
            foot_traj.append([current_foot_pos[0],self.foot_width if is_left_foot else -self.foot_width, current_foot_pos[2],0])
            torso_traj.append([*current_torso_pos, 0.0])
            
        return time_traj, foot_idx_traj, foot_traj, torso_traj
    
    
    def plan_up_stairs(self, num_steps=5, current_torso_pos = np.array([0.0, 0.0, 0.0]), current_foot_pos = np.array([0.0, 0.0, 0.0])):
        time_traj = []
        foot_idx_traj = []
        foot_traj = []
        torso_traj = []
        
        # 初始位置
        torso_height_offset = -0.05  # 躯干高度偏移
        current_torso_pos[2] = torso_height_offset
        torso_yaw = 0.0
        # current_foot_pos = np.array([0.0, 0.0, 0.0])
        offset_x = [0.0, -0.0, -0.0, -0.0, -0.0]
        first_step_offset = 0.35
        
        # 为每一步生成落脚点
        for step in range(num_steps):
            # 更新时间
            self.total_step += 1
            time_traj.append(self.total_step * self.dt)
            
            # 左右脚交替
            is_left_foot = ((self.total_step -1) % 2 == 0)
            foot_idx_traj.append(0 if is_left_foot else 1)
            
            # 计算躯干位置
            if step == 0:
                
                current_foot_pos[0] = current_torso_pos[0] + self.step_length  # 脚掌相对躯干前移
                current_foot_pos[1] = current_torso_pos[1] + self.foot_width if is_left_foot else -self.foot_width  # 左右偏移
                current_foot_pos[2] = self.step_height  # 脚掌高度
                current_torso_pos[0] += self.step_length/3
                
            elif step == num_steps - 1: # 最后一步
                # current_torso_pos[0] += self.step_length/2  # 向前移动
                # current_torso_pos[2] += self.step_height/2  # 向上移动
                # current_foot_pos[0] = current_torso_pos[0] # 最后一步x不动
                current_torso_pos[0] = current_foot_pos[0] # 最后一步躯干x在双脚上方
                current_foot_pos[1] = current_torso_pos[1] + self.foot_width if is_left_foot else -self.foot_width  # 左右偏移
                current_torso_pos[2] += self.step_height 
            else:
                current_torso_pos[0] += self.step_length  # 向前移动
                current_torso_pos[2] += self.step_height  # 向上移动
            
                # 计算落脚点位置
                current_foot_pos[0] = current_torso_pos[0] + self.step_length/2  # 脚掌相对躯干前移
                current_foot_pos[1] = current_torso_pos[1] + self.foot_width if is_left_foot else -self.foot_width  # 左右偏移
                current_foot_pos[2] += self.step_height
                
            if step < len(offset_x) and not step == num_steps - 1:    # 脚掌偏移
                current_foot_pos[0] += offset_x[step]
                # current_torso_pos[0] += offset_x[step]
            # 添加轨迹点
            foot_traj.append([*current_foot_pos, torso_yaw])
            torso_traj.append([*current_torso_pos, torso_yaw])
            
        return time_traj, foot_idx_traj, foot_traj, torso_traj
    

def publish_foot_pose_traj(time_traj, foot_idx_traj, foot_traj, torso_traj):
    rospy.init_node('stair_climbing_planner', anonymous=True)
    pub = rospy.Publisher('/humanoid_mpc_foot_pose_target_trajectories', 
                         footPoseTargetTrajectories, queue_size=10)
    rospy.sleep(1)

    msg = footPoseTargetTrajectories()
    msg.timeTrajectory = time_traj
    msg.footIndexTrajectory = foot_idx_traj
    msg.footPoseTrajectory = []

    for i in range(len(time_traj)):
        foot_pose_msg = footPose()
        foot_pose_msg.footPose = foot_traj[i]
        foot_pose_msg.torsoPose = torso_traj[i]
        msg.footPoseTrajectory.append(foot_pose_msg)

    pub.publish(msg)
    rospy.sleep(1.5)

if __name__ == '__main__':
    try:
        planner = StairClimbingPlanner()
        time_traj_0, foot_idx_traj_0, foot_traj_0, torso_traj_0 = planner.plan_up_stairs()
        print("Up stairs plan done.")
        print("Time trajectory:", time_traj_0)
        print("Foot index trajectory:", foot_idx_traj_0)
        print("Foot pose trajectory:", foot_traj_0)
        print("Torso pose trajectory:", torso_traj_0)
        print(torso_traj_0[-1][0:3])
        time_traj, foot_idx_traj, foot_traj, torso_traj = time_traj_0, foot_idx_traj_0, foot_traj_0, torso_traj_0
        # time_traj, foot_idx_traj, foot_traj, torso_traj = planner.predifine_traj()
        # time_traj_1, foot_idx_traj_1, foot_traj_1, torso_traj_1 = planner.move_to_down_stairs(current_torso_pos=np.array(torso_traj_0[-1][0:3]), current_foot_pos=np.array(foot_traj_0[-1][0:3]))
        # print("\nMove to down stairs plan done.")
        # print("Time trajectory:", time_traj_1)
        # print("Foot index trajectory:", foot_idx_traj_1)
        # print("Foot pose trajectory:", foot_traj_1)
        # print("Torso pose trajectory:", torso_traj_1)
        # time_traj += time_traj_1
        # foot_idx_traj += foot_idx_traj_1
        # foot_traj += foot_traj_1
        # torso_traj += torso_traj_1
        # print(torso_traj_1[-1][0:3])
        # time_traj_2, foot_idx_traj_2, foot_traj_2, torso_traj_2 = planner.plan_down_stairs(current_torso_pos=np.array(torso_traj_1[-1][0:3]), current_foot_pos=np.array(foot_traj_1[-1][0:3]))
        # print("\nDown stairs plan done.")
        # print("Time trajectory:", time_traj_2)
        # print("Foot index trajectory:", foot_idx_traj_2)
        # print("Foot pose trajectory:", foot_traj_2)
        # print("Torso pose trajectory:", torso_traj_2)
        # time_traj += time_traj_2
        # foot_idx_traj += foot_idx_traj_2
        # foot_traj += foot_traj_2
        # torso_traj += torso_traj_2
        # # 打印规划结果
        # print("\nTime trajectory:", time_traj)
        # print("Foot index trajectory:", foot_idx_traj)
        # print("Foot pose trajectory:", foot_traj)
        # print("Torso pose trajectory:", torso_traj)
        
        publish_foot_pose_traj(time_traj, foot_idx_traj, foot_traj, torso_traj)
    except rospy.ROSInterruptException:
        pass
