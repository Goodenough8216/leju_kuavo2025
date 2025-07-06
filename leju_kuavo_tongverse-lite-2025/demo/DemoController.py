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
import termios
import tty
import select
from geometry_msgs.msg import Pose, Point, Quaternion
from kuavo_msgs.srv import SetTagPose

class DemoController:
    def __init__(self):
        """初始化控制器"""
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
        
        # 添加初始位姿配置
        self.SCENE_CFG = {
            "point_1": {"position": [11.8283, -8.7758, 0.0], "orientation": [0, 0, -180]}, # 沙地前
            "point_2": {"position": [2.3283, 9.458, -0.051], "orientation": [0, 0, -90]},  # 导航场景
            "point_3": {"position": [9.5283, 8.4758, -0.052], "orientation": [0, 0, 0]},   # 搬运场景
            "point_4": {"position": [0.2283, 8.7758, 0.0], "orientation": [0, 0, 0]},      # 上斜坡前
            "point_5": {"position": [4.9083, -8.7758, 0.0], "orientation": [0, 0, -180]},  # 上楼梯前
        }
        
        # 仿真相关变量
        self.sim_running = True
        self.sensor_time = 0
        self.last_sensor_time = 0
        self.is_grab_box_demo = False
        
        # 添加按键监听相关变量
        self.old_settings = termios.tcgetattr(sys.stdin)

    def init_ros(self):
        """初始化ROS相关的组件"""
        # 初始化ROS节点
        rospy.init_node('demo_controller', anonymous=True)
        
        # 发布器和订阅器
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

    def get_action(self, obs: Dict[str, Any]) -> Optional[Dict]:
        """获取当前action，如果没有新的action则等待并持续发布上一次的观测数据
        
        Args:
            obs: 当前的观测数据字典
            
        Returns:
            当前action字典或None
        """
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

    def start_stair_climb(self) -> None:
        """启动爬楼梯规划器"""
        # 使用bash执行命令
        command = f"env -i bash -c 'source {CONTORLLER_PATH}/devel/setup.bash && rosrun humanoid_controllers stairClimbPlanner.py'"
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
            rospy.loginfo(f"Successfully started stair climbing planner")
            
            # 检查进程是否立即失败
            if self.stair_process.poll() is not None:
                raise Exception(f"Process failed to start with return code: {self.stair_process.returncode}")
                
        except Exception as e:
            rospy.logerr(f"Failed to start stair climbing planner: {str(e)}")
            if self.stair_process is not None:
                try:
                    os.killpg(os.getpgid(self.stair_process.pid), signal.SIGTERM)
                except:
                    pass
                self.stair_process = None

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
        initial_quat = Rotation.from_euler("xyz", new_euler, degrees=True).as_quat()[[3, 0, 1, 2]]
        
        # 设置位置和姿态，使用传入的高度值
        if init_pos is None:
            init_pos = [robot_init_pos[0][0], robot_init_pos[0][1], height]  # 从二维数组中提取位置值
        else:
            init_pos = [init_pos[0], init_pos[1], height]
        robot.set_world_pose(init_pos, initial_quat)
        
        print("Robot pose set to:", init_pos, initial_quat) 
        return init_pos, initial_quat
    def __del__(self):
        """析构函数，确保清理子进程"""
        self.cleanup()
        
    def getKey(self):
        tty.setraw(sys.stdin.fileno())
        rlist, _, _ = select.select([sys.stdin], [], [], 0.001)
        if rlist:
            key = sys.stdin.read(1)
        else:
            key = ''
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
        return key

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

    def set_box_and_shelf_positions(self, box_pose: Dict[str, float], shelf_pose: Dict[str, float]) -> bool:
        """设置箱子和架子的位置
        
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
            
            # 设置箱子位置 (tag_id = 1)
            box_pose_msg = Pose()
            box_pose_msg.position = Point(
                x=box_pose.get('x', 3.15),
                y=box_pose.get('y', 0.00245),
                z=box_pose.get('z', 0.78)
            )
            box_pose_msg.orientation = Quaternion(
                x=0.0,
                y=-0.707,
                z=0.0,
                w=0.707
            )
            
            response = set_tag_pose(tag_id=1, pose=box_pose_msg)
            if not response.success:
                rospy.logerr(f"Failed to set box position: {response.message}")
                return False
            
            # 设置架子位置 (tag_id = 2)
            shelf_pose_msg = Pose()
            shelf_pose_msg.position = Point(
                x=shelf_pose.get('x', -2.004896),
                y=shelf_pose.get('y', 0.00245),
                z=shelf_pose.get('z', 0.85)
            )
            shelf_pose_msg.orientation = Quaternion(
                x=0.707,
                y=0.0,
                z=0.707,
                w=0.0
            )
            
            response = set_tag_pose(tag_id=2, pose=shelf_pose_msg)
            if not response.success:
                rospy.logerr(f"Failed to set shelf position: {response.message}")
                return False
            
            rospy.loginfo("Successfully set box and shelf positions")
            return True
            
        except rospy.ROSException as e:
            rospy.logerr(f"Service call failed: {e}")
            return False
