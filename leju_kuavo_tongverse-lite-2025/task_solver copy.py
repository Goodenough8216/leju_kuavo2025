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
#------------------------------------
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
#------------------------------------
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
        self.task_params = task_params
        self.agent_params = agent_params
        self.step_count = 0 
        self.goal = np.array([
            [-9.5283,-8.4758,0.78]
        ])
        self.target_color = ['red', 'blue', 'yellow']
        self.start_launch()
        self.i=0
        self.j=0
        self.init_yaw = 0
        self.init_pos = np.zeros(3)
        # raise NotImplementedError("Implement your own TaskSolver here")

        self.started = False
        self.boxes_info   = None 
        self.shelves_info = None
    def init_ros(self):
        """初始化ROS相关的组件"""
        # 初始化ROS节点
        rospy.init_node('velocity_publisher', anonymous=True)
        # rospy.init_node('cmd_vel_listener', anonymous=True)
        # 发布器和订阅器
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.sensor_pub = rospy.Publisher('/sensors_data_raw', sensorsData, queue_size=2)
        self.joint_cmd_sub = rospy.Subscriber('/joint_cmd', jointCmd, self.joint_cmd_callback)
        
        #--------------------------------------
        # 添加用于图像转换的桥接器
        self.bridge = CvBridge()
        
        # 添加图像发布器
        self.camera_rgb_pub = rospy.Publisher('/camera/rgb', Image, queue_size=10)
        self.camera_depth_pub = rospy.Publisher('/camera/depth', Image, queue_size=10)
        #---------------------------------------
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

        #--------------------------------------
        # 处理相机数据
        if "camera" in obs:
            camera_data = obs["camera"]
            
            # 发布 RGB 图像
            if "rgb" in camera_data:
                try:
                    rgb_image = camera_data["rgb"].reshape((-1, 3))  # 假设 RGB 数据是 (N, 3)
                    rgb_msg = self.bridge.cv2_to_imgmsg(rgb_image, encoding="8UC1")
                    self.camera_rgb_pub.publish(rgb_msg)
                except Exception as e:
                    rospy.logerr(f"Failed to publish RGB image: {str(e)}")
            
            # 发布深度图像
            if "depth" in camera_data:
                try:
                    depth_image = camera_data["depth"]  # 假设深度数据是 (n, m, 1)
                    depth_msg = self.bridge.cv2_to_imgmsg(depth_image, encoding="passthrough")
                    self.camera_depth_pub.publish(depth_msg)
                except Exception as e:
                    rospy.logerr(f"Failed to publish depth image: {str(e)}")
        #--------------------------------------
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

    def process_task_info(self, robot_init_pos, color):
        """处理任务信息，提取箱子和货架位置
        
        Args:
            task_params: 任务参数字典
            robot_init_pos: 机器人初始位置
            
        Returns:
            self.boxes_info: 箱子信息字典
            self.shelves_info: 货架信息字典
        """
        # 提取箱子位置
        self.boxes_info = {
            'red_box': self.task_params['initial_box_world_position'][0],
            'blue_box': self.task_params['initial_box_world_position'][1],
            'yellow_box': self.task_params['initial_box_world_position'][2]
        }
        print("original self.boxes_info", self.boxes_info)
        
        # 提取货架位置
        self.shelves_info = self.task_params['shelve_world_position']
        
        # 计算位置偏移
        robot_offset = robot_init_pos  # 使用机器人初始位置作为参考
        
        # # 调整所有位置
        # for key in self.boxes_info:
        #     # print(f"key：{key}, self.boxes_info[key]：{self.boxes_info[key]} robot_offset：{robot_offset}")    
        #     boxes_pos_temp = self.boxes_info[key] - robot_offset
        #     boxes_pos_temp[0] = boxes_pos_temp[0] - 0.18
        #     boxes_pos_temp[1] += boxes_pos_temp[1] * 0.02
        #     self.boxes_info[key][0] = ( boxes_pos_temp[0]  ) * np.cos(self.init_yaw) + boxes_pos_temp[1] * np.sin(self.init_yaw)
        #     self.boxes_info[key][1] = - ( boxes_pos_temp[0] ) * np.sin(self.init_yaw) + boxes_pos_temp[1] * np.cos(self.init_yaw)
        #     self.boxes_info[key][2] = 0.76
        #     # print(f"key：{key}, self.boxes_info[key]：{self.boxes_info[key]}")    
        # for key in self.shelves_info:
        #     shelves_pos_temp = self.shelves_info[key] - robot_offset
        #     self.shelves_info[key][0] = shelves_pos_temp[0] * np.cos(self.init_yaw) + shelves_pos_temp[1] * np.sin(self.init_yaw)
        #     self.shelves_info[key][1] = -shelves_pos_temp[0] * np.sin(self.init_yaw) + shelves_pos_temp[1] * np.cos(self.init_yaw)
        #     self.shelves_info[key][2] = 1.2
        if color == 'red':
        # 调整所有位置
            for key in self.boxes_info:
                # print(f"key：{key}, self.boxes_info[key]：{self.boxes_info[key]} robot_offset：{robot_offset}")    
                boxes_pos_temp = self.boxes_info[key] - robot_offset
                boxes_pos_temp[0] = boxes_pos_temp[0] - 0.19
                boxes_pos_temp[1] += boxes_pos_temp[1] * 0.002 + 0.02
                self.boxes_info[key][0] = ( boxes_pos_temp[0]  ) * np.cos(self.init_yaw) + boxes_pos_temp[1] * np.sin(self.init_yaw)
                self.boxes_info[key][1] = - ( boxes_pos_temp[0] ) * np.sin(self.init_yaw) + boxes_pos_temp[1] * np.cos(self.init_yaw)
                self.boxes_info[key][2] = 0.8
                # print(f"key：{key}, self.boxes_info[key]：{self.boxes_info[key]}")    
            for key in self.shelves_info:
                shelves_pos_temp = self.shelves_info[key] - robot_offset
                self.shelves_info[key][0] = shelves_pos_temp[0] * np.cos(self.init_yaw) + shelves_pos_temp[1] * np.sin(self.init_yaw)
                self.shelves_info[key][1] = -shelves_pos_temp[0] * np.sin(self.init_yaw) + shelves_pos_temp[1] * np.cos(self.init_yaw)
                self.shelves_info[key][2] = 1.2
        else :pass
            # # 调整所有位置
            # for key in self.boxes_info:
            #     print(f"key：{key}, self.boxes_info[key]：{self.boxes_info[key]} robot_offset：{robot_offset}")    

            #     self.boxes_info[key] = self.boxes_info[key] - robot_offset
            #     self.boxes_info[key][2] = 0.8
            #     print(f"key：{key}, self.boxes_info[key]：{self.boxes_info[key]}")    
            # for key in self.shelves_info:
            #     self.shelves_info[key] = self.shelves_info[key] - robot_offset
            #     self.shelves_info[key][2] = 1.2

        return self.boxes_info, self.shelves_info
    

    def set_box_and_shelf_positions_1(self, box_pose: Dict[str, float], shelf_pose: Dict[str, float], color) -> bool:
        """设置箱子和架子的位置（包含朝向旋转）
        
        Args:
            box_pose: 箱子位置字典，包含 x,y,z 坐标和 orientation
            shelf_pose: 架子位置字典，包含 x,y,z 坐标和 orientation
            
        Returns:
            bool: 设置是否成功
        """
        try:
            # 等待服务可用
            rospy.wait_for_service('/set_tag_pose', timeout=10.0)
            set_tag_pose = rospy.ServiceProxy('set_tag_pose', SetTagPose)
            
            # ====================== 箱子朝向旋转 ======================
            # 原始四元数（相对于机器人初始坐标系）
            original_box_quat = [0.0, -0.707, 0.0, 0.707]  # w, x, y, z
            
            # 创建绕Z轴旋转init_yaw的四元数
            yaw_rotation = Rotation.from_euler('z', -self.init_yaw)
            box_rotation = Rotation.from_quat([original_box_quat[1], original_box_quat[2], 
                                            original_box_quat[3], original_box_quat[0]])
            
            # 组合旋转
            adjusted_box_rot = yaw_rotation * box_rotation
            adjusted_box_quat = adjusted_box_rot.as_quat()  # 返回顺序x,y,z,w
            
            # ====================== 货架朝向旋转 ======================
            # 原始四元数（相对于机器人初始坐标系）
            original_shelf_quat = [0.707, 0.0, 0.707, 0.0]  # w, x, y, z
            
            shelf_rotation = Rotation.from_quat([original_shelf_quat[1], original_shelf_quat[2],
                                            original_shelf_quat[3], original_shelf_quat[0]])
            
            # 组合旋转
            adjusted_shelf_rot = yaw_rotation * shelf_rotation
            adjusted_shelf_quat = adjusted_shelf_rot.as_quat()

            # 设置箱子位置 (tag_id = 1)
            box_pose_msg = Pose()
            box_pose_msg.position = Point(
                x=box_pose.get('x', 3.15),
                y=box_pose.get('y', 0.00245),
                z=box_pose.get('z', 0.78)
            )
            
            if color == 'red' :
                box_pose_msg.orientation = Quaternion(
                    x=adjusted_box_quat[0],
                    y=adjusted_box_quat[1],
                    z=adjusted_box_quat[2],
                    w=adjusted_box_quat[3]
                )
            else:
                box_pose_msg.orientation = Quaternion(
                    x=original_box_quat[0],
                    y=original_box_quat[1],
                    z=original_box_quat[2],
                    w=original_box_quat[3]
                )
            response = set_tag_pose(tag_id=1, pose=box_pose_msg)
            if not response.success:
                rospy.logerr(f"Failed to set box position: {response.message}")
                return False
            rospy.loginfo(f"Setting box position: {box_pose_msg}")

            # 设置架子位置 (tag_id = 2)
            shelf_pose_msg = Pose()
            shelf_pose_msg.position = Point(
                x=shelf_pose.get('x', -2.004896),
                y=shelf_pose.get('y', 0.00245),
                z=shelf_pose.get('z', 0.85)
            )
            if color == 'red':
                shelf_pose_msg.orientation = Quaternion(
                    x=adjusted_shelf_quat[0],
                    y=adjusted_shelf_quat[1],
                    z=adjusted_shelf_quat[2],
                    w=adjusted_shelf_quat[3]
                )
            else:
                box_pose_msg.orientation = Quaternion(
                    x=original_shelf_quat[0],
                    y=original_shelf_quat[1],
                    z=original_shelf_quat[2],
                    w=original_shelf_quat[3]
                )
            response = set_tag_pose(tag_id=2, pose=shelf_pose_msg)
            if not response.success:
                rospy.logerr(f"Failed to set shelf position: {response.message}")
                return False
            
            # response = set_tag_pose(tag_id=3, pose=box_pose_msg)#干什么用的？？？？？
                
            rospy.loginfo("Successfully set box and shelf positions with adjusted orientation")
            return True
        
        except rospy.ROSException as e:
            rospy.logerr(f"Service call failed: {e}")
            return False
        
    def set_box_and_shelf_positions(self, robot_pos, color):
        # 处理任务信息
        self.boxes_info, self.shelves_info = self.process_task_info(robot_pos, color)
        
        # 设置第一个箱子(红箱子)和对应货架的位置
        # select_color = 'red' # red blue yellow
        select_color = color
        box_pose = {
            'x': float(self.boxes_info[select_color + '_box'][0]),
            'y': float(self.boxes_info[select_color + '_box'][1]),
            'z': float(self.boxes_info[select_color + '_box'][2])
        }
        
        shelf_pose = {
            'x': float(self.shelves_info[select_color + '_shelf'][0]),
            'y': float(self.shelves_info[select_color + '_shelf'][1]),
            'z': float(self.shelves_info[select_color + '_shelf'][2])
        }
        
        # # 设置位置
        success = self.set_box_and_shelf_positions_1(box_pose, shelf_pose, color)
        
        if not success:
            print("Failed to set box and shelf positions")
            exit(0)

    def launch_grab_box(self) -> None:
        """启动抓箱子任务的launch文件"""
        # 使用bash执行命令
        self.is_grab_box_demo = True
        command = f"env -i bash -c 'source {CONTORLLER_PATH}/devel/setup.bash && roslaunch grab_box grab_box_mm.launch'"
        print(command)
        try:
            # 使用shell=True允许执行完整的命令字符串，并将输出直接连接到当前终端
            self.grab_box_process = subprocess.Popen(
                command,
                shell=True,
                stdout=None,  # 不捕获输出，让它们直接显示在终端
                stderr=None,
                stdin=subprocess.PIPE,
                preexec_fn=os.setsid  # 使用新的进程组，便于后续清理
            )
            rospy.loginfo(f"Successfully started grab box launch")
            
            # 检查进程是否立即失败
            if self.grab_box_process.poll() is not None:
                raise Exception(f"Process failed to start with return code: {self.grab_box_process.returncode}")
                
        except Exception as e:
            rospy.logerr(f"Failed to start grab box launch: {str(e)}")
            if self.grab_box_process is not None:
                try:
                    os.killpg(os.getpgid(self.grab_box_process.pid), signal.SIGTERM)
                except:
                    pass
                self.grab_box_process = None

    def next_action(self, obs: dict) -> dict:
        # implement your own TaskSolver here
        body_state = obs["Kuavo"]["body_state"]
        pos = body_state["world_position"]          # 位置
        quat = body_state["world_orient"]         # 姿态
        linear_vel = body_state["root_linear_velocity"]  # 线速度
        angular_vel = body_state["root_angular_velocity"] # 角速度
        
        if self.started==False:
        #     #目标点检查
        #     distance = (self.goal[self.step_count][0] - pos[0])**2 + (self.goal[self.step_count][1] - pos[1])**2
        #     print(np.sqrt(distance))
        #     if np.sqrt(distance) < 0.15:  # 到达阈值
        #         rpy = self.quat2euler(quat, "sxyz") 
        #         print(f"yaw: {rpy[2]}")
        #         if rpy[2] > 0.1:
        #             self.get_cmd(0, 0, 0, -0.3)
        #         elif rpy[2] < -0.1:
        #             self.get_cmd(0, 0, 0, 0.3)
        #         else:
        #             self.started = True
        #     else:
        #         rpy = self.quat2euler(quat, "sxyz") 
        #         print(f"yaw: {rpy[2]}")
            
        #         dx = self.goal[self.step_count][0] - pos[0]
        #         dy = self.goal[self.step_count][1] - pos[1]
        #         print(f"x: {pos[0]}, y: {pos[1]}")
            
        #         target_dir = np.arctan2(dy, dx)
        #         # print(f"target_dir: {target_dir}")
            
        #         diff_rot = target_dir - rpy[2]
            
        #         if diff_rot > np.pi:
        #             diff_rot -= 2 * np.pi
        #         elif diff_rot < -np.pi:
        #             diff_rot += 2 * np.pi
        #         print(f"diff_rot: {diff_rot}")
            
        #         if abs(diff_rot) < 0.1:
        #             self.get_cmd(0.5, 0, 0, 0)
        #         elif abs(diff_rot) < 0.9:
        #             if(diff_rot > 0):
        #                 self.get_cmd(0, 0, 0, 0.3)
        #                 print("turn right")
        #             else:
        #                 self.get_cmd(0, 0, 0, -0.3)
        #         elif diff_rot > 0:
        #             self.get_cmd(0, 0, 0, 0.6)
        #         else:
        #             self.get_cmd(0, 0, 0, -0.6)
        #     # if rpy[2] > 0:
        #     #     self.get_cmd(0, 0, 0, -0.3)
        #     # elif rpy[2] < 0:
        #     #     self.get_cmd(0, 0, 0, 0.3)
            self.started = True
            self.launch_grab_box()
            self.target_color = ['red', 'blue', 'yellow']
            # self.get_cmd(0, 0, 0, 0)
            print(f"init_yaw: {self.init_yaw}")

        else:
            if self.j == 0:
                self.init_yaw = self.quat2euler(quat, "sxyz")[2]
                self.init_pos = pos
                self.set_box_and_shelf_positions(pos, self.target_color[0])
            elif self.j == 21500:
                # self.init_yaw = self.quat2euler(quat, "sxyz")[2]
                # self.init_pos = pos
                self.set_box_and_shelf_positions(self.init_pos, self.target_color[2])
            elif self.j == 43000:
                # self.init_yaw = self.quat2euler(quat, "sxyz")[2]
                # self.init_pos = pos
                self.set_box_and_shelf_positions(self.init_pos, self.target_color[1])
            if self.j == 0 or self.j == 21500 or self.j == 43000:
                command = f"env -i bash -c 'source {CONTORLLER_PATH}/devel/setup.bash && rosrun grab_box grab_box_demo '"
                print(command)
                try:
                    # 使用shell=True允许执行完整的命令字符串，并将输出直接连接到当前终端
                    self.stair_process = subprocess.Popen(
                        command,
                        shell=True,
                        stdout=None,  
                        stderr=None,
                        stdin=subprocess.PIPE,
                        preexec_fn=os.setsid  # 使用新的进程组，便于后续清理
                    )
                    rospy.loginfo(f"Successfully crab the boxes")
                    
                    # 检查进程是否立即失败
                    if self.stair_process.poll() is not None:
                        raise Exception(f"Process failed to start with return code: {self.stair_process.returncode}")
                except Exception as e:
                    rospy.logerr(f"Failed to start crab: {str(e)}")
                    if self.stair_process is not None:
                        try:
                            os.killpg(os.getpgid(self.stair_process.pid), signal.SIGTERM)
                        except:
                            pass
                        self.stair_process = None
            self.j+=1
            print(f"self.j: {self.j}")
        
        # 更新最新的观测数据
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