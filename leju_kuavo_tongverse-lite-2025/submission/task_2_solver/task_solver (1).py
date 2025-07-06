#!/usr/bin/env python3
# -*- coding: utf-8 -*-
CONTORLLER_PATH = "/TongVerse/biped_challenge/demo/kuavo-ros-controldev"

import sys
import os
sys.path.append("/opt/ros/noetic/lib/python3/dist-packages")
sys.path.append("/usr/lib/python3/dist-packages")
sys.path.append(os.path.join(CONTORLLER_PATH, "devel/lib/python3/dist-packages"))
import time
import rospy
import subprocess
from std_msgs.msg import Float32
from kuavo_msgs.msg import jointCmd, sensorsData
import numpy as np
from typing import Dict, Any, Optional
import signal
from scipy.spatial.transform import Rotation
from std_srvs.srv import SetBool, SetBoolResponse
import termios
import tty
import select
from geometry_msgs.msg import Pose, Point, Quaternion
from kuavo_msgs.srv import SetTagPose
from geometry_msgs.msg import Twist
from kuavo_msgs.msg import footPose, footPoseTargetTrajectories

class TaskSolver:
    def __init__(self, task_params, agent_params) -> None:
        # implement your own TaskSolver here
        
        # init_pos, init_quat = self.init_robot_pose("Kuavo")
        # 存储当前action
        self.current_action = None
        
        # 用于计算加速度的历史数据
        self.last_joint_velocities = None
        self.last_time = None
        
        # 添加新的成员变量
        self.last_obs = None
        self.last_published_time = None
        self.control_freq = 500
        self.dt = 1 / self.control_freq
        
        # 添加子进程存储变量
        self.launch_process = None
        
        # 仿真相关变量
        self.sim_running = True
        self.sensor_time = 0
        self.last_sensor_time = 0
        self.is_grab_box_demo = False
        self.stair_climb_started = False
        self.move_down_started = False
        self.task_params = task_params
        self.agent_params = agent_params
        self.step_count = 0 
        self.goal = np.array([
            [10.2, -9.3, 0.0, 0.3],
            [9.3, -9.3, 0.0, 0.1],
            [9.1, -9.3, 0.0, 0.2],
            [8.2, -9.3, 0.0, 0.1],
            [8.05, -9.3, 0.0, 0.2],
            [7.1, -9.3, 0.0, 0.1],
            [6.7, -9.3, 0.0, 0.3],
            [5.3, -8.7758, 0.0, 0.3],
            [4.93, -8.7758, 0.0, 0.1],
            [0,0,0,0],
            [0,0,0,0],
            [0,0,0,0],
            [3.15, -8.7758, 0.0, 0.3],
            [2.8, -8.7758, 0.0, 0.1],
            [0,0,0,0]
        ])
        self.dt = 0.5  # 步态周期
        self.foot_width = 0.10  # 宽
        self.step_height = 0.13  # 台阶高度
        self.step_length = 0.28  # 台阶长度
        self.total_step = 0  # 总步数
        self.down_dt = 0.5  # 步态周期
        self.down_step_height = - 0.20 * np.tan(np.pi/12)  # 下楼梯时脚掌高度
        self.down_step_length = 0.20  
        self.down_total_step = 0  # 总步数
        self.count = 0
        self.start_launch()

    def init_ros(self):
        """初始化ROS相关的组件"""
        # 初始化ROS节点
        rospy.init_node('velocity_publisher', anonymous=True)
        # rospy.init_node('cmd_vel_listener', anonymous=True)
        # 发布器和订阅器
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.sensor_pub = rospy.Publisher('/sensors_data_raw', sensorsData, queue_size=2)
        self.joint_cmd_sub = rospy.Subscriber('/joint_cmd', jointCmd, self.joint_cmd_callback)
        
        # 设置发布频率
        self.publish_rate = rospy.Rate(self.control_freq)  # 500Hz的发布频率
      
        # 添加仿真启动服务
        self.sim_start_srv = rospy.Service('sim_start', SetBool, self.sim_start_callback)
      
        # 添加退出处理
        rospy.on_shutdown(self.cleanup)
      
        # 添加频率统计的发布器
        self.freq_pub = rospy.Publisher('/controller_freq', Float32, queue_size=10)

    def start_launch(self, reset_ground: bool = True) -> None:
        """启动指定的命令行命令并保持进程存活
      
        Args:
            reset_ground: 是否重置地面高度，默认为True
        """
        # 使用bash执行命令
        command = f"bash -c 'source {CONTORLLER_PATH}/devel/setup.bash && roslaunch humanoid_controllers load_kuavo_isaac_sim.launch reset_ground:={str(reset_ground).lower()}'"
        print(command)
        try:
            # 使用shell=True允许执行完整的命令字符串，并将输出直接连接到当前终端
            self.launch_process = subprocess.Popen(
                command,
                shell=True,
                stdout=None,
                stderr=None,
                stdin=subprocess.PIPE,
                preexec_fn=os.setsid
            )
            rospy.loginfo(f"Successfully started command: {command}")
          
            # 检查进程是否立即失败
            if self.launch_process.poll() is not None:
                raise Exception(f"Process failed to start with return code: {self.launch_process.returncode}")
              
        except Exception as e:
            rospy.logerr(f"Failed to start command: {str(e)}")
            if self.launch_process is not None:
                try:
                    os.killpg(os.getpgid(self.launch_process.pid), signal.SIGTERM)
                except:
                    pass
                self.launch_process = None
        # 初始化ROS相关组件
        self.init_ros()

    def sim_start_callback(self, req: SetBool) -> SetBoolResponse:
        """仿真启动服务的回调函数
      
        Args:
            req: SetBool请求，data字段为True表示启动仿真，False表示停止仿真
          
        Returns:
            SetBoolResponse: 服务响应
        """
        response = SetBoolResponse()
      
        self.sim_running = req.data
      
        if req.data:
            rospy.loginfo("Simulation started")
        else:
            rospy.loginfo("Simulation stopped")
      
        response.success = True
        response.message = "Simulation control successful"
      
        return response

    def cleanup(self):
        """清理资源，在节点关闭时调用"""
        if self.launch_process is not None:
            try:
                rospy.loginfo("Cleaning up launch process...")
                os.killpg(os.getpgid(self.launch_process.pid), signal.SIGTERM)
                self.launch_process.wait()
                self.launch_process = None
                rospy.loginfo("Launch process cleaned up")
            except Exception as e:
                rospy.logerr(f"Error cleaning up launch process: {str(e)}")
          
            # 清理爬楼梯进程
            if hasattr(self, 'stair_process') and self.stair_process is not None:
                try:
                    rospy.loginfo("Cleaning up stair climbing process...")
                    os.killpg(os.getpgid(self.stair_process.pid), signal.SIGTERM)
                    self.stair_process.wait()
                    self.stair_process = None
                    rospy.loginfo("Stair climbing process cleaned up")
                except Exception as e:
                    rospy.logerr(f"Error cleaning up stair climbing process: {str(e)}")

            # 清理抓箱子进程
            if hasattr(self, 'grab_box_process') and self.grab_box_process is not None:
                try:
                    rospy.loginfo("Cleaning up grab box process...")
                    os.killpg(os.getpgid(self.grab_box_process.pid), signal.SIGTERM)
                    self.grab_box_process.wait()
                    self.grab_box_process = None
                    rospy.loginfo("Grab box process cleaned up")
                except Exception as e:
                    rospy.logerr(f"Error cleaning up grab box process: {str(e)}")

    def get_cmd(self, linear_x, linear_y, linear_z, angular_z):
        """
        发送速度指令到 /cmd_vel 话题
        :param linear_x: x 方向线性速度 (m/s)
        :param linear_y: y 方向线性速度 (m/s)
        :param linear_z: z 方向增量高度 (m)
        :param angular_z: yaw 方向角速度 (radian/s)
        """
        # 创建 Twist 消息
        twist_msg = Twist()
        twist_msg.linear.x = linear_x
        twist_msg.linear.y = linear_y
        twist_msg.linear.z = linear_z
        twist_msg.angular.z = angular_z
        twist_msg.angular.x = 0.0  # 未使用
        twist_msg.angular.y = 0.0  # 未使用
        # 发布消息
        self.cmd_vel_pub.publish(twist_msg)
        # rospy.loginfo(f"Published velocity: linear=({linear_x}, {linear_y}, {linear_z}), angular=({angular_z})")
        # self.listen()
        # return self.current_action

    def joint_cmd_callback(self, msg: jointCmd) -> None:
        """处理关节命令回调
      
        Args:
            msg: 关节命令消息
        """
        # 构建action字典，按照README.md中的格式
        action = {
            "arms": {
                "ctrl_mode": "position",
                "joint_values": np.zeros(14),  # 14 arm joints
                "stiffness": [100.0] * 14 if self.is_grab_box_demo else [200.0] * 14,  # 搬箱子需要更低刚度的手臂
                "dampings": [20.2, 20.2, 20.5, 20.5, 10.2, 10.2, 20.1, 20.1, 10.1, 10.1, 10.1, 10.1, 10.1, 10.1],
            },
            "legs": {
                "ctrl_mode": "effort",
                "joint_values": np.zeros(12),  # 12 leg joints
                "stiffness": [0.0] * 12,  # Not setting stiffness
                "dampings": [0.2] * 12,  # Not setting dampings
            },
            "head": {
                "ctrl_mode": "position",
                "joint_values": np.zeros(2),  # 2 head joints
                "stiffness": None,  # Not setting stiffness
                "dampings": None,  # Not setting dampings
            }
        }

        # 处理腿部力矩数据
        for i in range(6):
            action["legs"]["joint_values"][i*2] = msg.tau[i]        # 左腿
            action["legs"]["joint_values"][i*2+1] = msg.tau[i+6]    # 右腿

        # 处理手臂位置数据
        for i in range(7):
            action["arms"]["joint_values"][i*2] = msg.joint_q[i+12]    # 左臂
            action["arms"]["joint_values"][i*2+1] = msg.joint_q[i+19]  # 右臂
        # action["arms"]["joint_values"][1] = 100
        # print(action["arms"]["joint_values"])
        # 处理头部位置数据（如果有的话）
        if len(msg.joint_q) >= 28:  # 确保消息中包含头部数据
            action["head"]["joint_values"][0] = msg.joint_q[26]  # 头部第一个关节
            action["head"]["joint_values"][1] = msg.joint_q[27]  # 头部第二个关节

        # 更新当前action
        self.current_action = action
        # print(self.current_action)
  
    def quat2euler(self, q, axes):
        """
        四元数转欧拉角的简单实现
        """
        w, x, y, z = q
        # roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)  # use 90 degrees if out of range
        else:
            pitch = np.arcsin(sinp)

        # yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        if axes == 'sxyz':
            return np.array([roll, pitch, yaw])
        # 可以添加其他轴顺序的处理
        return np.array([roll, pitch, yaw])
  
    def process_obs(self, obs: Dict[str, Any], republish = False) -> None:
        """处理观测数据并发布传感器数据
      
        Args:
            obs: 观测数据字典，包含IMU和关节状态信息
        """
        sensor_data = sensorsData()
      
        # 设置时间戳
        current_time = rospy.Time.now()
        sensor_data.header.stamp = current_time
        sensor_data.header.frame_id = "world"
        sensor_data.sensor_time = rospy.Duration(self.sensor_time)
        if republish:
            pass
            # self.sensor_time += self.dt
        else:
            self.sensor_time += obs["imu_data"]["imu_time"] - self.last_sensor_time
        self.last_sensor_time = obs["imu_data"]["imu_time"]
        # print(f"sensor_time: {self.sensor_time}")
        # 处理IMU数据
        if "imu_data" in obs:
            imu_data = obs["imu_data"]
            sensor_data.imu_data.acc.x = imu_data["linear_acceleration"][0]
            sensor_data.imu_data.acc.y = imu_data["linear_acceleration"][1]
            sensor_data.imu_data.acc.z = imu_data["linear_acceleration"][2]
            sensor_data.imu_data.gyro.x = imu_data["angular_velocity"][0]
            sensor_data.imu_data.gyro.y = imu_data["angular_velocity"][1]
            sensor_data.imu_data.gyro.z = imu_data["angular_velocity"][2]
            sensor_data.imu_data.quat.w = imu_data["orientation"][0]
            sensor_data.imu_data.quat.x = imu_data["orientation"][1]
            sensor_data.imu_data.quat.y = imu_data["orientation"][2]
            sensor_data.imu_data.quat.z = imu_data["orientation"][3]

        # 处理关节数据
        if "Kuavo" in obs and "joint_state" in obs["Kuavo"]:
            joint_state = obs["Kuavo"]["joint_state"]
          
            # 初始化关节数据数组
            sensor_data.joint_data.joint_q = [0.0] * 28
            sensor_data.joint_data.joint_v = [0.0] * 28
            sensor_data.joint_data.joint_vd = [0.0] * 28
            sensor_data.joint_data.joint_current = [0.0] * 28

            # 处理腿部数据
            if "legs" in joint_state:
                legs_data = joint_state["legs"]
                legs_pos = legs_data["positions"]
                legs_vel = legs_data["velocities"]
                legs_effort = legs_data["applied_effort"]
              
                for i in range(6):
                    # 左腿
                    sensor_data.joint_data.joint_q[i] = legs_pos[i*2]
                    sensor_data.joint_data.joint_v[i] = legs_vel[i*2]
                    sensor_data.joint_data.joint_current[i] = legs_effort[i*2]
                    # 右腿
                    sensor_data.joint_data.joint_q[i+6] = legs_pos[i*2+1]
                    sensor_data.joint_data.joint_v[i+6] = legs_vel[i*2+1]
                    sensor_data.joint_data.joint_current[i+6] = legs_effort[i*2+1]

            # 处理手臂数据
            if "arms" in joint_state:
                arms_data = joint_state["arms"]
                arms_pos = arms_data["positions"]
                arms_vel = arms_data["velocities"]
                arms_effort = arms_data["applied_effort"]
              
                for i in range(7):
                    # 左臂
                    sensor_data.joint_data.joint_q[i+12] = arms_pos[i*2]
                    sensor_data.joint_data.joint_v[i+12] = arms_vel[i*2]
                    sensor_data.joint_data.joint_current[i+12] = arms_effort[i*2]
                    # 右臂
                    sensor_data.joint_data.joint_q[i+19] = arms_pos[i*2+1]
                    sensor_data.joint_data.joint_v[i+19] = arms_vel[i*2+1]
                    sensor_data.joint_data.joint_current[i+19] = arms_effort[i*2+1]

            # 处理头部数据
            if "head" in joint_state:
                head_data = joint_state["head"]
                head_pos = head_data["positions"]
                head_vel = head_data["velocities"]
                head_effort = head_data["applied_effort"]
              
                for i in range(2):
                    sensor_data.joint_data.joint_q[26+i] = head_pos[i]
                    sensor_data.joint_data.joint_v[26+i] = head_vel[i]
                    sensor_data.joint_data.joint_current[26+i] = head_effort[i]
         
        # 发布传感器数据
        self.sensor_pub.publish(sensor_data)
    
    def start_stair_climb(self) -> None:
        """启动爬楼梯规划器"""
        # # 使用bash执行命令
        # command = f"env -i bash -c 'source {CONTORLLER_PATH}/devel/setup.bash && rosrun humanoid_controllers stairClimbPlanner.py'"
        # print(command)
        # try:
        #     # 使用shell=True允许执行完整的命令字符串，并将输出直接连接到当前终端
        #     self.stair_process = subprocess.Popen(
        #         command,
        #         shell=True,
        #         stdout=None,  
        #         stderr=None,
        #         stdin=subprocess.PIPE,
        #         preexec_fn=os.setsid  # 使用新的进程组，便于后续清理
        #     )
        #     rospy.loginfo(f"Successfully started stair climbing planner")
            
        #     # 检查进程是否立即失败
        #     if self.stair_process.poll() is not None:
        #         raise Exception(f"Process failed to start with return code: {self.stair_process.returncode}")
        # except Exception as e:
        #     rospy.logerr(f"Failed to start stair climbing planner: {str(e)}")
        #     if self.stair_process is not None:
        #         try:
        #             os.killpg(os.getpgid(self.stair_process.pid), signal.SIGTERM)
        #         except:
        #             pass
        #         self.stair_process = None
        time_traj_0, foot_idx_traj_0, foot_traj_0, torso_traj_0 = self.plan_up_stairs()
        print("Up stairs plan done.")
        print("Time trajectory:", time_traj_0)
        print("Foot index trajectory:", foot_idx_traj_0)
        print("Foot pose trajectory:", foot_traj_0)
        print("Torso pose trajectory:", torso_traj_0)
        print(torso_traj_0[-1][0:3])
        time_traj, foot_idx_traj, foot_traj, torso_traj = time_traj_0, foot_idx_traj_0, foot_traj_0, torso_traj_0
        self.publish_foot_pose_traj(time_traj, foot_idx_traj, foot_traj, torso_traj)
    
    def plan_up_stairs(self, num_steps=8, current_torso_pos = np.array([0.0, 0.0, 0.0]), current_foot_pos = np.array([0.0, 0.0, 0.0])):
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
        first_step_offset = 0.29
        
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
                current_foot_pos[0] = current_torso_pos[0] + first_step_offset  # 脚掌相对躯干前移
                current_foot_pos[1] = current_torso_pos[1] + self.foot_width if is_left_foot else -self.foot_width  # 左右偏移
                current_foot_pos[2] = self.step_height  # 脚掌高度
                current_torso_pos[0] += self.step_length * 0.7
            elif step%2 == 0:
                current_foot_pos[0] = current_torso_pos[0] + self.step_length  # 脚掌相对躯干前移
                current_foot_pos[1] = current_torso_pos[1] + self.foot_width if is_left_foot else -self.foot_width  # 左右偏移
                current_foot_pos[2] = self.step_height  # 脚掌高度
                current_torso_pos[0] += self.step_length * 0.7
            else:
            # elif step == num_steps - 1: # 最后一步
                # current_torso_pos[0] += self.step_length/2  # 向前移动
                # current_torso_pos[2] += self.step_height/2  # 向上移动
                # current_foot_pos[0] = current_torso_pos[0] # 最后一步x不动
                current_torso_pos[0] = current_foot_pos[0] # 最后一步躯干x在双脚上方
                current_foot_pos[1] = current_torso_pos[1] + self.foot_width if is_left_foot else -self.foot_width  # 左右偏移
                # current_torso_pos[2] += self.step_height 
            # else:
            #     current_torso_pos[0] += self.step_length  # 向前移动
            #     current_torso_pos[2] += self.step_height  # 向上移动
            
            #     # 计算落脚点位置
            #     current_foot_pos[0] = current_torso_pos[0] + self.step_length/2  # 脚掌相对躯干前移
            #     current_foot_pos[1] = current_torso_pos[1] + self.foot_width if is_left_foot else -self.foot_width  # 左右偏移
            #     current_foot_pos[2] += self.step_height
                
            if step < len(offset_x) and not step == num_steps - 1:    # 脚掌偏移
                current_foot_pos[0] += offset_x[step]
                # current_torso_pos[0] += offset_x[step]
            # 添加轨迹点
            foot_traj.append([*current_foot_pos, torso_yaw])
            torso_traj.append([*current_torso_pos, torso_yaw])
            
        return time_traj, foot_idx_traj, foot_traj, torso_traj
    
    def start_move_down(self) -> None:
        time_traj_0, foot_idx_traj_0, foot_traj_0, torso_traj_0 = self.plan_move_down()
        print("Up stairs plan done.")
        print("Time trajectory:", time_traj_0)
        print("Foot index trajectory:", foot_idx_traj_0)
        print("Foot pose trajectory:", foot_traj_0)
        print("Torso pose trajectory:", torso_traj_0)
        print(torso_traj_0[-1][0:3])
        time_traj, foot_idx_traj, foot_traj, torso_traj = time_traj_0, foot_idx_traj_0, foot_traj_0, torso_traj_0
        self.publish_foot_pose_traj(time_traj, foot_idx_traj, foot_traj, torso_traj)
    
    def plan_move_down(self, num_steps=40, current_torso_pos = np.array([0.0, 0.0, 0.0]), current_foot_pos = np.array([0.0, 0.0, 0.0])):
        time_traj = []
        foot_idx_traj = []
        foot_traj = []
        torso_traj = []
        
        # 初始位置
        torso_height_offset = -0.15  # 躯干高度偏移
        current_torso_pos[2] = torso_height_offset
        torso_yaw = 0.0
        # current_foot_pos = np.array([0.0, 0.0, 0.0])
        offset_x = [0.0, -0.0, -0.0, -0.0, -0.0]
        first_step_offset = 0.30
        
        # 为每一步生成落脚点
        for step in range(num_steps):
            # 更新时间
            self.down_total_step += 1
            time_traj.append(self.down_total_step * self.down_dt)
            
            # 左右脚交替
            is_left_foot = ((self.down_total_step -1) % 2 == 0)
            foot_idx_traj.append(0 if is_left_foot else 1)
            
            # 计算躯干位置
            if step == 0:
                current_foot_pos[0] = current_torso_pos[0] + first_step_offset  # 脚掌相对躯干前移
                current_foot_pos[1] = current_torso_pos[1] + self.foot_width if is_left_foot else -self.foot_width  # 左右偏移
                current_foot_pos[2] = self.down_step_height  # 脚掌高度
                current_torso_pos[0] += self.down_step_length * 0.7
            elif step%2 == 0:
                current_foot_pos[0] = current_torso_pos[0] + self.down_step_length  # 脚掌相对躯干前移
                current_foot_pos[1] = current_torso_pos[1] + self.foot_width if is_left_foot else -self.foot_width  # 左右偏移
                current_foot_pos[2] = self.down_step_height  # 脚掌高度
                current_torso_pos[0] += self.down_step_length * 0.7
            else:
            # elif step == num_steps - 1: # 最后一步
                # current_torso_pos[0] += self.down_step_length/2  # 向前移动
                # current_torso_pos[2] += self.down_step_height/2  # 向下移动
                # current_foot_pos[0] = current_torso_pos[0] # 最后一步x不动
                current_torso_pos[0] = current_foot_pos[0] # 最后一步躯干x在双脚上方
                current_foot_pos[1] = current_torso_pos[1] + self.foot_width if is_left_foot else -self.foot_width  # 左右偏移
                # current_torso_pos[2] += self.down_step_height 
            # else:
            #     current_torso_pos[0] += self.down_step_length  # 向前移动
            #     current_torso_pos[2] += self.down_step_height / 2 # 向下移动
            
            #     # 计算落脚点位置
            #     current_foot_pos[0] = current_torso_pos[0] + self.down_step_length/2  # 脚掌相对躯干前移
            #     current_foot_pos[1] = current_torso_pos[1] + self.foot_width if is_left_foot else -self.foot_width  # 左右偏移
            #     current_foot_pos[2] += self.down_step_height
                
            if step < len(offset_x) and not step == num_steps - 1:    # 脚掌偏移
                current_foot_pos[0] += offset_x[step]
                # current_torso_pos[0] += offset_x[step]
            # 添加轨迹点
            foot_traj.append([*current_foot_pos, torso_yaw])
            torso_traj.append([*current_torso_pos, torso_yaw])
            
        return time_traj, foot_idx_traj, foot_traj, torso_traj

    def publish_foot_pose_traj(self, time_traj, foot_idx_traj, foot_traj, torso_traj):
        # rospy.init_node('stair_climbing_planner', anonymous=True)
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

    def next_action(self, obs: dict) -> dict:
        
        body_state = obs["Kuavo"]["body_state"]
        pos = body_state["world_position"]          # 位置
        quat = body_state["world_orient"]         # 姿态
        linear_vel = body_state["root_linear_velocity"]  # 线速度
        angular_vel = body_state["root_angular_velocity"] # 角速度

        rpy = self.quat2euler(quat, "sxyz") 
        # print(f"yaw: {rpy[2]}")

        if self.step_count == 9 :
            if linear_vel[0] < 0.01 :
                self.step_count += 1
                print(f"x: {pos[0]}, y: {pos[1]}, yaw: {rpy[2]}")
            self.get_cmd(0, 0, 0, 0)
            self.last_obs = obs
            self.process_obs(obs)
            st = time.time()
            while self.current_action is None and not rospy.is_shutdown():
                # 发布传感器数据
                self.process_obs(self.last_obs,republish=True)
                
                # 等待一个发布周期
                self.publish_rate.sleep()
                
            freq = Float32()
            freq.data = 1
            self.freq_pub.publish(freq)
            return self.current_action 
        
        elif self.step_count == 10:
            if rpy[2] > 0 :
                theta_e = rpy[2] - np.pi
            else:
                theta_e = np.pi + rpy[2]
            if abs(theta_e) < 0.01:
                self.step_count += 1
                self.get_cmd(0, 0, 0, 0)
                print(f"x: {pos[0]}, y: {pos[1]}, yaw: {rpy[2]}")
                self.last_obs = obs
                self.process_obs(obs)
                st = time.time()
                while self.current_action is None and not rospy.is_shutdown():
                    # 发布传感器数据
                    self.process_obs(self.last_obs,republish=True)
                    
                    # 等待一个发布周期
                    self.publish_rate.sleep()
                freq = Float32()
                freq.data = 1
                self.freq_pub.publish(freq)
                return self.current_action 
            
            self.get_cmd(0, 0, 0, -0.5 * theta_e)
            print(f"yaw: {rpy[2]}")
            self.last_obs = obs
            self.process_obs(obs)
            st = time.time()
            while self.current_action is None and not rospy.is_shutdown():
                # 发布传感器数据
                self.process_obs(self.last_obs,republish=True)
                
                # 等待一个发布周期
                self.publish_rate.sleep()
            freq = Float32()
            freq.data = 1
            self.freq_pub.publish(freq)
            return self.current_action 
        
        elif self.step_count == 11:
            if self.count < 250:
                self.get_cmd(0, 0, 0, 0)
                self.count += 1
                print(self.count)
                self.last_obs = obs
                self.process_obs(obs)
                st = time.time()
                while self.current_action is None and not rospy.is_shutdown():
                    # 发布传感器数据
                    self.process_obs(self.last_obs,republish=True)
                    
                    # 等待一个发布周期
                    self.publish_rate.sleep()
                    
                freq = Float32()
                freq.data = 1
                self.freq_pub.publish(freq)
                return self.current_action
            if self.stair_climb_started == False:
                print("start stair climb")
                self.stair_climb_started = True
                self.start_stair_climb()
            # print(pos[0], pos[1])       # 3.827, -8.844
            # self.step_count += 1
            self.count += 1
            if self.count > 4500:
                self.step_count += 1
            self.last_obs = obs
            self.process_obs(obs)            
            while self.current_action is None and not rospy.is_shutdown():
                # 发布传感器数据
                self.process_obs(self.last_obs,republish=True)
                
                # 等待一个发布周期
                self.publish_rate.sleep()
                
            freq = Float32()
            freq.data = 1
            self.freq_pub.publish(freq)
                
            # 如果没有收到action，持续发布上一次的观测数据
            # self.current_action = None # 清空当前action
            
            return self.current_action 
        
        elif self.step_count == 14:
            if rpy[2] > 0 :
                theta_e = rpy[2] - np.pi
            else:
                theta_e = np.pi + rpy[2]
            if abs(theta_e) < 0.01:
                self.step_count += 1
                self.get_cmd(0, 0, 0, 0)
                print(f"x: {pos[0]}, y: {pos[1]}, yaw: {rpy[2]}")
                self.last_obs = obs
                self.process_obs(obs)
                st = time.time()
                while self.current_action is None and not rospy.is_shutdown():
                    # 发布传感器数据
                    self.process_obs(self.last_obs,republish=True)
                    
                    # 等待一个发布周期
                    self.publish_rate.sleep()
                freq = Float32()
                freq.data = 1
                self.freq_pub.publish(freq)
                return self.current_action 
            
            self.get_cmd(0, 0, 0, -0.5 * theta_e)
            print(f"yaw: {rpy[2]}")
            self.last_obs = obs
            self.process_obs(obs)
            st = time.time()
            while self.current_action is None and not rospy.is_shutdown():
                # 发布传感器数据
                self.process_obs(self.last_obs,republish=True)
                
                # 等待一个发布周期
                self.publish_rate.sleep()
            freq = Float32()
            freq.data = 1
            self.freq_pub.publish(freq)
            return self.current_action

        elif self.step_count == 15:
            if self.count < 4750:
                self.get_cmd(0, 0, 0, 0)
                self.count += 1
                print(self.count)
                self.last_obs = obs
                self.process_obs(obs)
                st = time.time()
                while self.current_action is None and not rospy.is_shutdown():
                    # 发布传感器数据
                    self.process_obs(self.last_obs,republish=True)
                    
                    # 等待一个发布周期
                    self.publish_rate.sleep()
                    
                freq = Float32()
                freq.data = 1
                self.freq_pub.publish(freq)
                return self.current_action
            if self.move_down_started == False:
                print("start move down")
                self.move_down_started = True
                self.start_move_down()
            # print(pos[0], pos[1])
            # self.step_count += 1
            self.count += 1
            if self.count > 30000:
                self.step_count += 1
            self.last_obs = obs
            self.process_obs(obs)            
            while self.current_action is None and not rospy.is_shutdown():
                # 发布传感器数据
                self.process_obs(self.last_obs,republish=True)
                
                # 等待一个发布周期
                self.publish_rate.sleep()
                
            freq = Float32()
            freq.data = 1
            self.freq_pub.publish(freq)

            return self.current_action 
        
        # 目标点检查
        distance = (self.goal[self.step_count][0] - pos[0])**2 + (self.goal[self.step_count][1] - pos[1])**2
        # print(np.sqrt(distance))

        # self.start_stair_climb()

        if distance < 0.0025:  # 到达阈值
            self.step_count += 1
    
        dx = self.goal[self.step_count][0] - pos[0]
        dy = self.goal[self.step_count][1] - pos[1]
        # print(f"x: {pos[0]}, y: {pos[1]}")
    
        target_dir = np.arctan2(dy, dx)
        # print(f"target_dir: {target_dir}")
    
        diff_rot = target_dir - rpy[2]
    
        if diff_rot > np.pi:
            diff_rot -= 2 * np.pi
        elif diff_rot < -np.pi:
            diff_rot += 2 * np.pi
        # print(f"diff_rot: {diff_rot}")

        if self.step_count == 8 or self.step_count == 13:
            target_vel = 0.25 * np.sqrt(distance)
        # elif self.step_count == 13:
        #     target_vel = 0
        else:
            target_vel= self.goal[self.step_count][3]
            # target_vel = 0

        if abs(diff_rot) > 0.3:
            if diff_rot > 0:
                self.get_cmd(0, 0, 0, 0.3)
            else:
                self.get_cmd(0, 0, 0, -0.3)
        # elif abs(diff_rot) > 0.1:
        #     if diff_rot > 0:
        #         self.get_cmd(0.5*target_vel, 0, 0, 0.1)
        #     else:
        #         self.get_cmd(0.5*target_vel, 0, 0, -0.1)
        elif diff_rot > 0:
            self.get_cmd(target_vel, 0, 0, 0.1)
        else:
            self.get_cmd(target_vel, 0, 0, -0.1)

        self.last_obs = obs
            
        self.process_obs(obs)
            
        # 如果没有收到action，持续发布上一次的观测数据
        # self.current_action = None # 清空当前action
        st = time.time()
        while self.current_action is None and not rospy.is_shutdown():
            # 发布传感器数据
            self.process_obs(self.last_obs,republish=True)
            
            # 等待一个发布周期
            self.publish_rate.sleep()
            
        freq = Float32()
        freq.data = 1
        self.freq_pub.publish(freq)

        return self.current_action
