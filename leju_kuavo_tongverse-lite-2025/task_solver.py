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
import os
import signal
from scipy.spatial.transform import Rotation
from std_srvs.srv import SetBool, SetBoolResponse
from geometry_msgs.msg import Pose, Point, Quaternion
from kuavo_msgs.srv import SetTagPose
from geometry_msgs.msg import Twist

class TaskSolver:
    def __init__(self, task_params, agent_params) -> None:
        # implement your own TaskSolver here
        self.start_launch()
        # init_pos, init_quat = self.init_robot_pose("Kuavo")
        self.is_grab_box_demo = False
        self.current_action = None
        self.task_params = task_params
        self.agent_params = agent_params
        self.step_count = 0 
        self.goal = np.array([
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ])
      
        # raise NotImplementedError("Implement your own TaskSolver here")
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
        self.control_freq = 500
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

    def init_robot_pose(self, robot, init_pos = None, init_yaw: float = 0, height: float = 0.82):
        """初始化机器人姿态
      
        Args:
            robot: 机器人对象
            height: 机器人初始高度，默认为0.82
        """
        # 获取当前机器人位姿
        robot_init_pos, robot_init_orient = robot.get_world_pose()
        print(f"robot_init_pos: {robot_init_pos}")
        print(f"robot_init_orient: {robot_init_orient}")
        # 从二维数组中提取四元数值
        quat = robot_init_orient[0]  # [w, x, y, z]
        current_euler = Rotation.from_quat([quat[0], quat[1], quat[2], quat[3]]).as_euler("xyz", degrees=True)
      
        print(f"current_euler: {current_euler}")
        # 设置roll=0, pitch=3度, 保持原来的yaw角度
        new_euler = [0, 3, init_yaw]  # [roll, pitch+3, yaw], 拿不到正确的yaw
        initial_quat = Rotation.from_euler("xyz", new_euler, degrees=True).as_quat()
      
        # 设置位置和姿态，使用传入的高度值
        if init_pos is None:
            init_pos = [robot_init_pos[0][0], robot_init_pos[0][1], height]  # 从二维数组中提取位置值
        else:
            init_pos = [init_pos[0], init_pos[1], height]
        robot.set_world_pose(init_pos, initial_quat)
      
        print("Robot pose set to:", init_pos, initial_quat) 
        return init_pos, initial_quat

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
        return self.current_action

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
        print(self.current_action)
  
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

    def next_action(self, obs: dict) -> dict:
        # implement your own TaskSolver here
        body_state = obs["Kuavo"]["body_state"]
        pos = body_state["world_position"]          # 位置
        quat = body_state["world_orient"]         # 姿态
        linear_vel = body_state["root_linear_velocity"]  # 线速度
        angular_vel = body_state["root_angular_velocity"] # 角速度

        # 目标点检查
        # distance = np.linalg.norm(self.goal[self.step_count] - pos[:2])
        # if distance < 0.1:  # 到达阈值
        #     self.step_count += 1
        #     if self.step_count >= len(self.goal):
        #         return self.get_cmd(0, 0, 0, 0)
      
        rpy = self.quat2euler(quat, "sxyz")
      
        dx = self.goal[self.step_count][0] - pos[0]
        dy = self.goal[self.step_count][1] - pos[1]
      
        target_dir = np.arctan2(dy, dx)
      
        diff_rot = target_dir - rpy[2]
      
        if diff_rot > np.pi:
            diff_rot -= 2 * np.pi
        elif diff_rot < -np.pi:
            diff_rot += 2 * np.pi
      
        if abs(diff_rot) < np.pi / 90:
            return self.get_cmd(0, 0, 0, 0)
        elif diff_rot > 0:
            return self.get_cmd(0, 0, 0, -0.1)
        else:
            return self.get_cmd(0, 0, 0, 0.1)
      
        # if self.step_count == 0:

        # raise NotImplementedError("Implement your own TaskSolver here")
    # def cmd_vel_callback(msg):
    #     print("Received cmd_vel message:")
    #     print("  Linear:", msg.linear)
    #     print("  Angular:", msg.angular)

    # def listen(self):

    #     rospy.Subscriber('/cmd_vel', Twist, self.cmd_vel_callback)
    #     rospy.spin()