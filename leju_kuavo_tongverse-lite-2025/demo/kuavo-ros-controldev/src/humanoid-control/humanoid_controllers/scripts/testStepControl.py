#!/usr/bin/env python3

import rospy
import numpy as np
from kuavo_msgs.msg import footPose, footPoseTargetTrajectories

class StairClimbingPlanner:
    def __init__(self):
        self.dt = 0.4  # 步态周期
        self.foot_width = 0.10  # 宽
        self.step_height = 0.1  # 台阶高度
        self.step_length = 0.37  # 台阶长度
        self.total_step = 0  # 总步数
        
    def move_to_down_stairs(self, step = 3, current_torso_pos = np.array([0.0, 0.0, 0.0]), current_foot_pos = np.array([0.0, 0.0, 0.0])):
        time_traj = []
        foot_idx_traj = []
        foot_traj = []
        torso_traj = []
        
        # current_torso_pos = np.array([0.0, 0.0, 0.0, 0.0])
        # current_foot_pos = np.array([0.0, 0.0, 0.0])
        step_length = 0.3
        for i in range(step):
            self.total_step += 1
            time_traj.append((time_traj[-1] if len(time_traj) > 0 else 0) + self.dt)
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
            
            # 双足支撑
            time_traj.append(time_traj[-1] + self.dt)
            foot_idx_traj.append(2)
            foot_traj.append(foot_traj[-1])
            torso_traj.append(torso_traj[-1])
            
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
        time_traj_0, foot_idx_traj_0, foot_traj_0, torso_traj_0 = planner.move_to_down_stairs()
        print("Up stairs plan done.")
        print("Time trajectory:", time_traj_0)
        print("Foot index trajectory:", foot_idx_traj_0)
        print("Foot pose trajectory:", foot_traj_0)
        print("Torso pose trajectory:", torso_traj_0)
        print(torso_traj_0[-1][0:3])
        time_traj, foot_idx_traj, foot_traj, torso_traj = time_traj_0, foot_idx_traj_0, foot_traj_0, torso_traj_0

        # 打印规划结果
        print("\nTime trajectory:", time_traj)
        print("Foot index trajectory:", foot_idx_traj)
        print("Foot pose trajectory:", foot_traj)
        print("Torso pose trajectory:", torso_traj)
        
        publish_foot_pose_traj(time_traj, foot_idx_traj, foot_traj, torso_traj)
    except rospy.ROSInterruptException:
        pass
