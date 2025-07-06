#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import math
import rospy
import numpy as np
from enum import Enum
from pynput import keyboard

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "motion_capture_ik_packaged/scripts/ik"))
from torso_ik import *
from kuavo_msgs.msg import sensorsData, armTargetPoses
from kuavo_msgs.srv import changeArmCtrlMode, changeArmCtrlModeRequest
from pydrake.all import StartMeshcat, AddMultibodyPlantSceneGraph, MeshcatVisualizer

class ArmType(Enum):
    Right = 0,
    Left = 1
class KeyBoardArmController:
    def __init__(self, x_gap = 0.03, y_gap = 0.03, z_gap = 0.03, time_gap = 0.5,  which_hand=ArmType.Right, interpolate_num=50):
        rospy.init_node("arm_control_keyboard_node", anonymous=True)
        
        """ Variables """
        self._control_xyz_keys = {                      # 控制位置的按键映射
            'w': {'axis':'x', 'index':0, 'gap': x_gap},
            's': {'axis':'x', 'index':0, 'gap': -x_gap},
            'd': {'axis':'y', 'index':1, 'gap': y_gap},
            'a': {'axis':'y', 'index':1, 'gap': -y_gap},
            'q': {'axis':'z', 'index':2, 'gap': z_gap},
            'e': {'axis':'z', 'index':2, 'gap': -z_gap}
        }
        self.eef_current = [np.zeros(3), np.zeros(3)]  # 当前末端位置
        self.eef_target = [np.zeros(3), np.zeros(3)]   # 目标末端位置
        self.current_joint_values = [0] * 14           # 当前关节数值
        self.which_hand = which_hand                   # 左/右手
        self._flag_pose_inited = False
        self.exec_time = 0
        self.time_gap = time_gap    # 每段执行时间
        self.interpolate_num = interpolate_num
        
        # 初始化ik
        self.meshcat = StartMeshcat()
        self.eef_z_bias = -0.15
        
        # 给定modelfile
        self.model_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                       "motion_capture_ik_packaged/models/biped_gen4.0/urdf/biped_v3_arm.urdf")
        self.end_frames_name = ["torso", "l_hand_roll", "r_hand_roll", "l_forearm_pitch", "r_forearm_pitch"]
        
        # 实例化ArmIK类
        self.arm_ik = ArmIk(self.model_file, self.end_frames_name, self.meshcat, 1e-3, 1e-3, 1000, eef_z_bias=self.eef_z_bias, as_mc_ik=True)
        torso_yaw_deg = 0.0
        torso_height = 0.0
        self.arm_ik.init_state(torso_yaw_deg, torso_height)
        
        rospy.sleep(2)
        self.joint_state_subscriber = rospy.Subscriber('/sensors_data_raw', sensorsData, self.update_joint_state_callback)
        self._arm_traj_change_mode_client = rospy.ServiceProxy("/arm_traj_change_mode", changeArmCtrlMode)
        
        self.set_arm_control_mode(2) # 切换手臂控制模式
        
        self._traj_pub = rospy.Publisher("/kuavo_arm_target_poses", armTargetPoses, queue_size=10)
        
        # FIXME todebug
        self.counter = 0
        self.time_thershold = 200

    def update_joint_state_callback(self, data):
        arm_joint_data = data.joint_data.joint_q[12:]
        self.current_joint_values = arm_joint_data
        if self.which_hand == ArmType.Left:
            self.eef_current = self.arm_ik.left_hand_pose(arm_joint_data)
        else:
            self.eef_current = self.arm_ik.right_hand_pose(arm_joint_data)
        
        if not self._flag_pose_inited:
            np.set_printoptions(suppress=True)
            print("------- Init joint data deg:", np.array(arm_joint_data) * 180 / math.pi)
            self.eef_target[0] = self.eef_current[0].copy()
            self.eef_target[1] = self.eef_current[1].copy()
            print("------- Init Pose:", self.eef_target)
            self._flag_pose_inited = True

    def IK(self, l_hand_pose=None, r_hand_pose=None, l_hand_RPY=None, r_hand_RPY=None, l_elbow_pos=None, r_elbow_pos=None):
        """
        根据目标点逆解IK
        参数：
        l_hand_pose: 左臂末端位置
        r_hand_pose: 右臂末端位置
        l_hand_RPY: 左臂末端RPY
        r_hand_RPY: 右臂末端RPY
        l_elbow_pos: 左臂肘部位置
        r_elbow_pos: 右臂肘部位置
        """
        try:
            if self.which_hand == ArmType.Right:
                r_hand_pose = r_hand_pose.tolist()
                r_hand_RPY = r_hand_RPY.tolist()
            else:
                l_hand_pose = l_hand_pose.tolist()
                l_hand_RPY = l_hand_RPY.tolist()

            q_target_rad = self.arm_ik.computeIK(list(self.current_joint_values), l_hand_pose, r_hand_pose, l_hand_RPY, r_hand_RPY, l_elbow_pos, r_elbow_pos)
            if q_target_rad is None:
                print("=====IK失败！！=====")
                return None

            q_target_degree = list(np.round(np.array(q_target_rad) * 180 / math.pi, 2))
            # print("当前末端位置", self.eef_current)
            # print("设定的末端位置", r_hand_pose, r_hand_RPY)
            # print("当前关节位置", self.current_joint_values)
            print("逆解出的关节位置", q_target_degree)
            # print("逆解出的关节位置正解得到的末端位置", self.arm_ik.right_hand_pose(q_target_rad))
            return q_target_degree
        except Exception as e:
            print(e)
    
    def on_press(self, key):
        self.exec_time = self.exec_time + self.time_gap
        if hasattr(key, 'char') and key.char in self._control_xyz_keys:
            try:
                ctrl = self._control_xyz_keys[key.char]
                index = ctrl['index']
                self.eef_target[0][index] += ctrl['gap']
                print(f"按下{key.char}键, {ctrl['axis']} 值变化 {ctrl['gap']}, 当前位置: {self.eef_target[0]}")
            except Exception as e:
                print(e)
        
    def on_release(self, key):
        try:
            if hasattr(key, 'char') and key.char in self._control_xyz_keys:
                if self.which_hand == ArmType.Left:
                    target_joint_pose = self.IK(l_hand_pose=self.eef_target[0], l_hand_RPY=self.eef_target[1])
                else:
                    target_joint_pose = self.IK(r_hand_pose=self.eef_target[0], r_hand_RPY=self.eef_target[1])
                
                if target_joint_pose is None:
                    print("IK fail!")
                else:
                    message_torque = []
                    message_time = []
                    
                    # 对关节数值插值
                    message_torque_temp = self.linear_interpolate_joint(np.array(self.current_joint_values)*180/math.pi, target_joint_pose, int(self.interpolate_num*self.exec_time/self.time_gap))
                    for torque_data in message_torque_temp:
                        for i in torque_data:
                            message_torque.append(i)
                    
                    # 对时间插值
                    message_time = self.linear_interpolate_time(0, self.exec_time, int(self.interpolate_num*self.exec_time/self.time_gap))
                    print("length of torque message and time: ", len(message_torque)/14, len(message_time))
                    # 运行
                    self.move_with_trajactory(message_time, message_torque)
                    self.exec_time = 0
        except Exception as e:
            print(e)

    def linear_interpolate_time(self, start, end, num_points):
        """
        在指定的时间范围内进行线性插值。
        参数:
        start (float): 时间范围的起始点。
        end (float): 时间范围的结束点。
        num_points (int): 需要生成的时间点数量。

        返回:
        list: 包含按时间顺序排列的等间隔时间点的列表。
        """
        step = (end - start) / (num_points - 1)
        return [start + i * step for i in range(num_points)]

    def linear_interpolate_joint(self, start_joints, end_joints, num_points):
        """
        执行关节角度的线性插值。
        参数:
        - start_joints: 起始关节角度的列表或元组。
        - end_joints: 结束关节角度的列表或元组，与 start_joints 对应。
        - num_points: 需要生成的插值点数，包括起始点和结束点。
 
        返回:
        - 一个插值关节角度集的列表，每个集合对应于特定插值点的所有关节角度。
        """
        result = []
        for start, end in zip(start_joints, end_joints):
            step = (end - start) / (num_points - 1)
            points = [start + i * step for i in range(num_points)]
            result.append(points)
        return list(zip(*result))

    def move_with_trajactory(self, time_list, torque_list):
        arm_traj_msg = armTargetPoses()
        arm_traj_msg.times = time_list
        arm_traj_msg.values = torque_list
        self._traj_pub.publish(arm_traj_msg)
        print("=====publish success!=====")
        
    def set_arm_control_mode(self, mode):
        """设置OCS2手臂控制模式"""
        try:
            # 使用初始化时的服务代理
            request = changeArmCtrlModeRequest()
            request.control_mode = mode

            # 发送请求并接收响应
            response = self._arm_traj_change_mode_client(request)
            
            if response.result:
                rospy.loginfo(f"Successfully changed arm control mode to {mode}: {response.message}")
            else:
                rospy.logwarn(f"Failed to change arm control mode to {mode}: {response.message}")

        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")
            
    def _message_torque_to_series(self, message_torque):
        final_message_torque_series = []
        for i in range(len(message_torque)):
            for j in range(len(message_torque[i])):
                final_message_torque_series.append(message_torque[i][j])
        return final_message_torque_series
    
    def _message_time_to_series(self, message_time):
        return [message_time[i][0] for i in range(len(message_time))]
    
    def run(self):
        # 创建监听器实例
        listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        listener.start()
        
        print("-------- 按键控制手臂运动：")
        print(" ws: 控制 x 轴")
        print(" ad: 控制 y 轴")
        print(" qe: 控制 z 轴")
        print(" 按 Ctrl-C 退出")
        
        # 保持程序运行
        while not rospy.is_shutdown():
            pass
        
if __name__ == "__main__":
    # Right Arm
    keyboard_arm_controller = KeyBoardArmController(x_gap = 0.03, y_gap = 0.03, z_gap = 0.03, 
                                                    time_gap = 0.5,  
                                                    which_hand = ArmType.Right, 
                                                    interpolate_num=50)
    keyboard_arm_controller.run()