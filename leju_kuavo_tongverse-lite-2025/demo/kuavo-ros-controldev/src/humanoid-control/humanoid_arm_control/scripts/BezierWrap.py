#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import sys 
import os
import rospy
import drake
import math
import traceback

from pydrake.all import StartMeshcat, AddMultibodyPlantSceneGraph, MeshcatVisualizer

from cubic_bezier import CubicBezier
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "motion_capture_ik_packaged/scripts/ik"))

from torso_ik import *
from data_utils import *

config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config/armcontrol_config.json")
"""
    ArmControl class
    封装了加载预设轨迹、根据目标点生成曲线、IK、开启手臂规划、将曲线发送至响应话题执行等函数
    
    Extern Args: 
    1. model file
    2. bezier_interpolate_point_num: 插值点数量
    3. bezier_interpolate_gap_time:  插值间隔时间
    
"""
class BezierWrap():
    def __init__(self) -> None:
        # 1. 获取配置文件
        try:
            with open(config_path, 'r') as config_file:
                self.config = json.load(config_file)
                print("- Load armcontrol config success -")
        except FileNotFoundError:
            print("BezierWrap: 未找到配置文件")
            exit(0)
        except json.JSONDecodeError:
            print("BezierWrap: JSON 格式错误")
            exit(0)
       
        # 轨迹构造和插值
        self._pre_action_data = None
        self._q0 = None
        self._q0_rad = None
        self._cubic_bezier = CubicBezier()
        
        # IK 参数及实例化
        self.meshcat = StartMeshcat()
        self.eef_z_bias = -0.15
        self.model_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.config["model_file_path"])
        self.end_frames_name = ["torso", "l_hand_roll", "r_hand_roll", "l_forearm_pitch", "r_forearm_pitch"]
        self.arm_ik = ArmIk(self.model_file, self.end_frames_name, self.meshcat, 1e-3, 1e-3, 1000, eef_z_bias=self.eef_z_bias, as_mc_ik=True)
        torso_yaw_deg = 0.0
        torso_height = 0.0
        self.arm_ik.init_state(torso_yaw_deg, torso_height)
        
    """
        加载固定轨迹
    """
    def load_pre_bezier_traj(self, filepath):
        print("- Load fix trajectory -")
        data = load_json_file(filepath)
        self._pre_action_data = frames_to_custom_action_data(data["frames"])
        self._q0 = get_q_from_action_file(self._pre_action_data)[0:14]
        self._q0_rad = (np.array(self._q0) * math.pi / 180)
        print("- Load success, initial joint value：", self._q0, "from file ", filepath)
    
    """ 根据目标点，在原贝塞尔曲线后接上一段，添加的点的形式仍然是end_point, left_control, right_control, 

    """
    def add_new_bezier_subtraj(self, q_target_rad, time_gap):
        # 将新控制点加入
        for i in self._pre_action_data:
            if i <= 14:
                # action_data_i 的 shape 为(n, 3, 2)
                action_data_i = self._pre_action_data[i]
                q_idx = i - 1
                # 提取控制点并计算新控制点
                last_bezier_curve = [action_data_i[-2][0], action_data_i[-2][2], action_data_i[-1][1], action_data_i[-1][0]]
                ctrl_points = self._cubic_bezier.calc_cubic_bezier_ctrl_point(*last_bezier_curve, last_bezier_curve[3], q_target_rad[q_idx])
                print("add new ctrl point: ", ctrl_points)
                # 改变上一段曲线的右控制点
                self._pre_action_data[i][-1][2] = list(ctrl_points[0])
                end_point = [action_data_i[-1][0][0]+time_gap, q_target_rad[q_idx]]
                # 添加最后一段曲线
                self._pre_action_data[i].append([end_point, list(ctrl_points[1]), end_point])
    
    """ 得到标准贝塞尔曲线形式，两个端点、两个控制点
    
    """
    def get_standard_bezier_line_format(self):
        curves = []
        for i in self._pre_action_data:
            curve_points = get_points_from_action_data(self._pre_action_data[i])
            curves.append(curve_points)
        return curves
    
    def get_traj_points_after_interpolate(self, standard_curves, interpolate_point_num):
        points_all = [] # 是每个关节的数值集合
        for i in range(len(standard_curves)):  # 14
            # 每个关节
            curve = standard_curves[i]
            points_curve = []   # 某个关节的曲线
            for j in range(len(curve)):
                # 每段曲线
                sub_curve = curve[j]
                # 插值
                points_subcurve = self._cubic_bezier.interpolate_cubic_bezier(sub_curve[0], sub_curve[1], sub_curve[2], sub_curve[3], num_points=interpolate_point_num)
                for item in points_subcurve:
                    points_curve.append(item)
            points_all.append(points_curve)
        return points_all
    
    def get_traj_points_after_seperate_time_q(self, points_all):
        # 将点拆分为时间和关节位置
        final_message_torque = []
        final_message_time = []
        
        # 取出时间
        for i in range(len(points_all[0])):
            final_message_time.append([points_all[0][i][0]])
        
        # final_message_torque 应该是n个14维的数组
        for i in range(len(points_all[0])): # 18
            final_message_torque.append([])
        for i in range(len(points_all[0])):    # 18
            for j in range(len(points_all)): # 14
                final_message_torque[i].append(points_all[j][i][1])
        return (final_message_time, final_message_torque)

        
        
    """ 从源文件加载的数据和目标关节点创建新贝塞尔曲线，需要先使用IK解出关节空间位置，再使用这个函数

        Args:
            1. 目标关节角度
            2. 插值点数量
            3. 间隔时间
    """
    def create_new_bezier_curve(self, q_target_rad, interpolate_point_num, time_gap):
        try:
            print("- start create new bezier line -")
            
            # 添加一段贝塞尔曲线,all of the calculation is in the format of 3point
            self.add_new_bezier_subtraj(q_target_rad=q_target_rad, time_gap=time_gap)
            
            # 将曲线的各个点转换形式，变为两个end_point和两个control_point的格式
            # curves的dimension为(28, n, 4, 2)，28个关节，3段曲线，每段4个点，每个点两个维度
            curves = self.get_standard_bezier_line_format()
            curves = curves[0:14]
            
            # 获取插值后的点
            # points_all 的 shape 为(14, n, 2), 14个关节，18个点，2
            points_all = self.get_traj_points_after_interpolate(curves, interpolate_point_num)
            
            # 把time和q分开
            result = self.get_traj_points_after_seperate_time_q(points_all)
            
            return result
                    
        except Exception as e:
            print(e)
            tb = traceback.extract_tb(e.__traceback__)
            for filename, line, function, text in tb:
                print(f"File: {filename}, Line: {line}, Function: {function}")
    
    """ 从预设轨迹终点解目标点的关节数值
    
    """
    def IK(self, l_hand_pose=None, r_hand_pose=None, l_hand_RPY=None, r_hand_RPY=None, l_elbow_pos=None, r_elbow_pos=None, use_preset_q0=True, q0=None):
        print("设置的目标位置为： " , r_hand_pose, r_hand_RPY, "\n")
        
        q_target_rad = self.arm_ik.computeIK(self._q0_rad, l_hand_pose=self.arm_ik.left_hand_pose(self._q0_rad)[0],r_hand_pose=r_hand_pose,l_hand_RPY=None,r_hand_RPY=r_hand_RPY,l_elbow_pos=l_elbow_pos,r_elbow_pos=r_elbow_pos)
        if q_target_rad is None:
            print("=====IK失败!!=====")
            return None
        
        q_target_degree = list(np.round(np.array(q_target_rad) * 180 / math.pi, 2))
        print("=====IK 成功！逆解得到的目标位置:=====")
        # 当前末端位置
        print("当前末端位置", self.arm_ik.right_hand_pose(self._q0_rad))
        # 设定的末端位置/
        print("设定的末端位置", r_hand_pose, r_hand_RPY)
        # 当前关节位置
        print("当前关节位置", self._q0)
        # 逆解出的关节位置
        print("逆解出的关节位置", q_target_degree)
        # 逆解出的关节位置正解得到的末端位置
        print("逆解出的关节位置正解得到的末端位置", self.arm_ik.right_hand_pose(q_target_rad))
        return q_target_degree

    



r_hand_pose = np.array([0.37312414, -0.23108104,  0.13475034])
r_hand_RPY = np.array([1.3272946 ,  0.03376237, -1.39988757])
HAND_POSE_RPY = None

pre_action_data_path = "/home/lab/singapore/kuavo_ros_application/src/ros_package/scripts/action_files/hand_up_side.tact"

if __name__ == "__main__":
    arm_control = BezierWrap()
    print("********************************************开始执行********************************************")
    print("target position: ", np.array([0.37312414, -0.23108104,  0.13475034]))
    # 2. 加载原轨迹
    arm_control.load_pre_bezier_traj(pre_action_data_path)
    # 3. 给定位置，计算IK， 获取目标关节位姿
    # print("target position: ")
    q_target_rad = arm_control.IK(r_hand_pose=np.array([0.37312414, -0.23108104,  0.13475034]), r_hand_RPY=HAND_POSE_RPY)
    # print("q_target: ", q_target_rad)
    # 4. 获取新曲线
    time_gap = 3    # 时间
    (time_list, torque_list) = arm_control.create_new_bezier_curve(q_target_rad, time_gap)
    # 5. 按照新曲线执行
    # print("torque_list: ", torque_list)
    arm_control.move_with_trajactory(time_list=time_list, torque_list=torque_list)
    # 6. 手臂运动到位后关闭灵巧手
    print("********************************************执行结束********************************************")