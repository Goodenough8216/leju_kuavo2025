#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import rospy
import json
import os
from apriltag_ros.msg import AprilTagDetectionArray

from humanoid_arm_control.srv import armControl, armControlResponse
import numpy as np

import linear_interpolate

from BezierWrap import BezierWrap
from KuavoSDK import kuavo

from threading import Thread

config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config/armservice_config.json")

""" ArmControl的rosservice封装类

封装了ArmControl类,能够获取tag位置信息,然后接受请求,执行固定轨迹和生成的轨迹

Extern Args:
    1. tag_id: 识别tag的id号,可以从req中给出,也可以从配置文件中给出
    2. X/Y/Z_OFFSET: 三个方向的偏移值，使用配置文件
    3. HAND_POSE_RPY: 最终抓取的手部旋转rpy,可以是none,也可以通过逻辑给出来
    4. pre_action_data_path: 预设动作路径 
    5. preset_target: 是否使用预设位置
"""

class ArmService:
    def __init__(self):
        # 1. 加载config配置文件
        try:
            with open(config_path, 'r') as file:
                self.config = json.load(file)
                print("- Load armservice config success")
        except FileNotFoundError:
            print("- 配置文件未找到")
            exit(0)
        except json.JSONDecodeError:
            print("- JSON格式错误")
            exit(0)
        
        # 实例化BezierWrap
        print("- Initialize BezierWrap -")
        self.bezier_wrap = BezierWrap()
        
        # 实例化kuavo
        print("- Initialize Kuavo -")
        self.kuavo = kuavo()
        
        # set arm mode
        print("- Set Arm Mode to Extern Mode -")
        self.kuavo.set_arm_control_mode(2)
        
        # 加载固定轨迹
        print("- start to load fix trajectory -")
        self._fix_trajectorys = {}
        for key in self.config["fix_trajectory_paths"]:
            action_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.config["fix_trajectory_paths"][key])
            self.bezier_wrap.load_pre_bezier_traj(action_file_path)
            curves = self.bezier_wrap.get_standard_bezier_line_format()
            curves = curves[0:14]
            
            # 获取插值后的点
            # points_all 的 shape 为(14, n, 2), 14个关节，18个点，2
            points_all = self.bezier_wrap.get_traj_points_after_interpolate(curves, 100)
            
            # 把time和q分开
            result = self.bezier_wrap.get_traj_points_after_seperate_time_q(points_all)
            self._fix_trajectorys[key] = result
        
        # 初始化AprilTag数据变量
        # self._has_apriltag = self.config["preset_target"]["use"]
        self._has_apriltag = False
        self._tag_position = self.config["preset_target"]["point"]
        self._tag_position = {}
        self._tag_id = 0
        print("tag_position: ", self._tag_position)
        
        # 创建AprilTag话题订阅者
        self._sub_tag_pose = rospy.Subscriber("/robot_tag_info", AprilTagDetectionArray, self.tag_callback, queue_size=1)
        
        # 创建ROS服务
        print("- Start service on /arm_control -")
        self._service = rospy.Service('/arm_control', armControl, self.arm_control_callback)
    
    # tag目标更新函数
    def tag_callback(self,msg):
        if not msg.detections:
            self._has_apriltag = False
            # rospy.logwarn("No apriltag detected")
            return
        
        for detection in msg.detections:
            # print(detection)
            self._has_apriltag = True
            self._tag_position["{}".format(detection.id[0])] = [detection.pose.pose.pose.position.x, detection.pose.pose.pose.position.y, detection.pose.pose.pose.position.z]
            # print("update tag position: {}".format(self._tag_position))
            
        
        if not self._has_apriltag:
            rospy.logwarn("No apriltag with ID {} detected".format(self.config["tag_id"]))

    # 0-middle 1-left 2-right
    def arm_control_callback(self, req):
        self._tag_id = req.tagid
        # 使用tag抓取
        if req.req == 0:
            print("\n！！！执行tag抓取轨迹！！！\n")
            success, duration = self.run_flexible_trajectory()
            return armControlResponse(success, duration)
            # FIXME 移动轨迹写好后可以放入
        # 使用固定轨迹抓取
        else:
            fix_thread = Thread(target=self.run_fix_trajectory, args=(req.req,))
            fix_thread.start()
            # print(self._fix_trajectorys["left"][0][-1][0])
            return armControlResponse(0, self._fix_trajectorys["left"][0][-1][0])
    
    # 固定轨迹执行函数
    def run_fix_trajectory(self, which_hand):
        print("enter run fix trajectory")
        
        zeros = [0,0,0,0,0,0]
        thumb = [0,100,0,0,0,0]
        fist = [100,100,100,100,100,100]
        if which_hand == 1:
            print("\n- Start execute trajectory {}".format("left"))
            time_list = self._fix_trajectorys["left"][0]
            torque_list = self._fix_trajectorys["left"][1]
            self.kuavo.move_with_trajactory(time_list, torque_list)
            rospy.sleep(4)
            self.kuavo.publish_hand_position(thumb,zeros)
            rospy.sleep(3.3)
            self.kuavo.publish_hand_position(fist, zeros)
            rospy.sleep(7)
            self.kuavo.publish_hand_position(thumb, zeros)
            rospy.sleep(3)
            self.kuavo.publish_hand_position(zeros, zeros)
        else:
            print("\n- Start execute trajectory {}".format("right"))
            torque_list = self._fix_trajectorys["right"][0]
            time_list = self._fix_trajectorys["right"][1]
            self.kuavo.move_with_trajactory(torque_list, time_list)
            rospy.sleep(4)
            self.kuavo.publish_hand_position(zeros, thumb)
            rospy.sleep(3.3)
            self.kuavo.publish_hand_position(zeros, fist)
            rospy.sleep(7)
            self.kuavo.publish_hand_position(zeros, thumb)
            rospy.sleep(3)
            self.kuavo.publish_hand_position(zeros, zeros)
    
    # 构造tag抓取轨迹
    def run_flexible_trajectory(self):
        # 检查是否有AprilTag数据
        if self._tag_position == {}:
            print("未检测到tag")
            zero_arm_rad = np.array([20,0,0,-30,0,0,0]) * math.pi / 180
            zero_arm_rad = zero_arm_rad.tolist()
            print("---zero arm rad---", zero_arm_rad)
            return 1, None
        try:
            print("- Start to construct the trajectory -")
            print("target position: ", self._tag_position)
            
            # 设置插值点数量和持续时间
            time_gap=self.config["interpolate_config"]["bezier_interpolate_gap_time"]
            interpolate_point_num=self.config["interpolate_config"]["bezier_interpolate_point_num"]
            
            zero_arm_rad = np.array([20,0,0,-30,0,0,0]) * math.pi / 180
            zero_arm_rad = zero_arm_rad.tolist()
            print("---zero arm rad---", zero_arm_rad)
            
            # 如果在右边，执行右手逻辑
            print("tag position: ", self._tag_position)
            if self._tag_position["{}".format(self._tag_id)][1] < 0:
                # TODO 在这里设置路径点并使用IK进行解算，注意IK之间的关系
                # NOTE 注意这里要用深拷贝，否则会是一样的值
                # 如果超出范围，返回false
                origin_target = np.array([self._tag_position["{}".format(self._tag_id)][0] , \
                                        self._tag_position["{}".format(self._tag_id)][1] , \
                                        self._tag_position["{}".format(self._tag_id)][2]])
                
                print("origin_target: ", origin_target)

                if not self.judge_boundary(origin_target, 2):
                    print("Beyond Boundary!!!")
                    return 2, None
                                
                # 抓取点
                target1 = np.array([self._tag_position["{}".format(self._tag_id)][0] + self.config["target_offsets"]["right_hand"]["x"], \
                                    self._tag_position["{}".format(self._tag_id)][1] + self.config["target_offsets"]["right_hand"]["y"], \
                                    self._tag_position["{}".format(self._tag_id)][2] + self.config["target_offsets"]["right_hand"]["z"],])
                print("grab in right hand")
                print("target1: ", target1)
                
                # 抬起的目标点，向上抬一点
                target2 = target1.copy()
                target2[2] += 0.1
                print("target2: ", target2)
                # 放置点
                
                # 1 抬起手臂
                r_hand_up_tact_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.config["preset_action_path"]["right_hand"]["hand_up"])
                self.bezier_wrap.load_pre_bezier_traj(r_hand_up_tact_path)
                curves = self.bezier_wrap.get_standard_bezier_line_format()
                curves = curves[0:14]
                points_all = self.bezier_wrap.get_traj_points_after_interpolate(curves, interpolate_point_num)
                time_list_up, torque_list_up = self.bezier_wrap.get_traj_points_after_seperate_time_q(points_all)
                
                # NOTE 一定要记得把rad改成degree再发送请求
                # 2 到抓取点
                q_grab_rad_temp = self.bezier_wrap.arm_ik.computeIK(self.bezier_wrap._q0_rad, l_hand_pose=self.bezier_wrap.arm_ik.left_hand_pose(self.bezier_wrap._q0_rad)[0],r_hand_pose=target1,l_hand_RPY=None,r_hand_RPY=None,l_elbow_pos=None,r_elbow_pos=None)
                q_grab_rad_temp = q_grab_rad_temp.tolist()
                q_grab_rad = [0.3490658503988659, 0.0, 0.0, -0.5235987755982988, 0.0, 0.0, 0.0]
                q_grab_rad.extend(q_grab_rad_temp[7:])
                print("q_grab_rad: ", q_grab_rad)
                if q_grab_rad is None:
                    print("==========IK fail1===========")
                    return 2, None
                grab_time = time_list_up[-1][0]
                grab_up_time = grab_time + time_gap
                linear_interpolate_time1 = [[e] for e in linear_interpolate.linear_interpolate_time(grab_time, grab_up_time, 100)]
                linear_interpolate_torque1 = np.array(linear_interpolate.linear_interpolate_joint(self.bezier_wrap._q0_rad, q_grab_rad, 100)) * 180 / math.pi
                
                # FIXME
                zero_arm_rad = np.array([20,0,0,-30,0,0,0]) * math.pi / 180
                zero_arm_rad = zero_arm_rad.tolist()
                
                # 3 到中间点
                q_up_rad_temp = self.bezier_wrap.arm_ik.computeIK(q_grab_rad, l_hand_pose=self.bezier_wrap.arm_ik.left_hand_pose(q_grab_rad)[0],r_hand_pose=target2,l_hand_RPY=None,r_hand_RPY=self.config["target_pose_rpy"],l_elbow_pos=None,r_elbow_pos=None)
                q_up_rad_temp = q_up_rad_temp.tolist()
                q_up_rad = [0.3490658503988659, 0.0, 0.0, -0.5235987755982988, 0.0, 0.0, 0.0]
                q_up_rad.extend(q_up_rad_temp[7:])
                print("q_up_rad_temp: ", q_up_rad_temp)
                print("zero_arm_rad: ", zero_arm_rad)
                print("q_up_rad: ", q_up_rad)
                if q_up_rad is None:
                    print("==========IK fail2===========")
                    return 2, None
                put_time = grab_up_time + 3
                linear_interpolate_time2 = [[e] for e in linear_interpolate.linear_interpolate_time(grab_up_time, put_time, 100)]
                linear_interpolate_torque2 = np.array(linear_interpolate.linear_interpolate_joint(q_grab_rad, q_up_rad, 100)) * 180 / math.pi
                
                # 4 到放置点
                put_end_time = put_time + time_gap
                q_put_rad = np.array(self.config["put_joint_pose"]["right_hand"]) * math.pi / 180
                linear_interpolate_time3 = [[e] for e in linear_interpolate.linear_interpolate_time(put_time, put_end_time, 100)]
                linear_interpolate_torque3 = np.array(linear_interpolate.linear_interpolate_joint(q_up_rad, q_put_rad, 100)) * 180 / math.pi
                
                # 5 放下
                r_hand_down_tact_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.config["preset_action_path"]["right_hand"]["hand_down"])
                self.bezier_wrap.load_pre_bezier_traj(r_hand_down_tact_path)
                curves = self.bezier_wrap.get_standard_bezier_line_format()
                curves = curves[0:14]
                points_all = self.bezier_wrap.get_traj_points_after_interpolate(curves, interpolate_point_num)
                time_list_down, torque_list_down = self.bezier_wrap.get_traj_points_after_seperate_time_q(points_all)
                time_list_down = [[e[0]+put_end_time] for e in time_list_down] # 时间列表要加上前面的时间
                
                # 6 合并轨迹
                time_list = time_list_up + linear_interpolate_time1 + linear_interpolate_time2 + linear_interpolate_time3 + time_list_down
                torque_list = torque_list_up + linear_interpolate_torque1.tolist() + linear_interpolate_torque2.tolist() + linear_interpolate_torque3.tolist() + torque_list_down
                
                # 5. 按照新曲线执行
                self.kuavo.move_with_trajactory(time_list=time_list, torque_list=torque_list)
                
                tag_thread = Thread(target=self.hand_control, args=(2, grab_time, grab_up_time-1, put_end_time, put_end_time+3))
                tag_thread.start()
            # 如果在左边，执行左手逻辑
            else:
                # TODO 在这里设置路径点并使用IK进行解算，注意IK之间的关系
                # NOTE 注意这里要用深拷贝，否则会是一样的值
                # 抓取点
                origin_target = np.array([self._tag_position["{}".format(self._tag_id)][0] , \
                                        self._tag_position["{}".format(self._tag_id)][1] , \
                                        self._tag_position["{}".format(self._tag_id)][2]])
                if not self.judge_boundary(origin_target, 1):
                    print("Beyond Boundary!!!")
                    return 2, None
                
                target1 = np.array([self._tag_position["{}".format(self._tag_id)][0] + self.config["target_offsets"]["left_hand"]["x"], \
                                    self._tag_position["{}".format(self._tag_id)][1] + self.config["target_offsets"]["left_hand"]["y"], \
                                    self._tag_position["{}".format(self._tag_id)][2] + self.config["target_offsets"]["left_hand"]["z"],])
                                # 抬起的目标点，向上抬一点
                target2 = target1.copy()
                print("grab in left hand")
                print("target1: ", target1)
                target2[2] += 0.1
                # 放置点
                
                # 1 抬起手臂
                l_hand_up_tact_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.config["preset_action_path"]["left_hand"]["hand_up"])
                self.bezier_wrap.load_pre_bezier_traj(l_hand_up_tact_path)
                curves = self.bezier_wrap.get_standard_bezier_line_format()
                curves = curves[0:14]
                points_all = self.bezier_wrap.get_traj_points_after_interpolate(curves, interpolate_point_num)
                time_list_up, torque_list_up = self.bezier_wrap.get_traj_points_after_seperate_time_q(points_all)
                
                # NOTE 一定要记得把rad改成degree再发送请求
                # 2 到抓取点
                q_grab_rad_temp = self.bezier_wrap.arm_ik.computeIK(self.bezier_wrap._q0_rad, l_hand_pose=target1, r_hand_pose=self.bezier_wrap.arm_ik.right_hand_pose(self.bezier_wrap._q0_rad)[0],l_hand_RPY=self.config["target_pose_rpy"], r_hand_RPY=None, l_elbow_pos=None,r_elbow_pos=None)
                q_grab_rad = q_grab_rad_temp[0:7].tolist()
                q_grab_rad.extend([0.3490658503988659, 0.0, 0.0, -0.5235987755982988, 0.0, 0.0, 0.0])
                
                print("q_grab_rad: ", np.array(q_grab_rad)*180/math.pi)
                if q_grab_rad is None:
                    print("==========IK fail1===========")
                    return 2, None
                grab_time = time_list_up[-1][0]
                grab_up_time = grab_time + time_gap
                linear_interpolate_time1 = [[e] for e in linear_interpolate.linear_interpolate_time(grab_time, grab_up_time, 100)]
                linear_interpolate_torque1 = np.array(linear_interpolate.linear_interpolate_joint(self.bezier_wrap._q0_rad, q_grab_rad, 100)) * 180 / math.pi
                
                # 3 到中间点
                q_up_rad_temp = self.bezier_wrap.arm_ik.computeIK(q_grab_rad, l_hand_pose=target2, r_hand_pose=self.bezier_wrap.arm_ik.right_hand_pose(q_grab_rad)[0], l_hand_RPY=self.config["target_pose_rpy"], r_hand_RPY=None, l_elbow_pos=None, r_elbow_pos=None)
                q_up_rad = q_up_rad_temp[0:7].tolist()
                q_up_rad.extend([0.3490658503988659, 0.0, 0.0, -0.5235987755982988, 0.0, 0.0, 0.0])
                if q_up_rad is None:
                    print("==========IK fail2===========")
                    return 2, None
                put_time = grab_up_time + 3
                linear_interpolate_time2 = [[e] for e in linear_interpolate.linear_interpolate_time(grab_up_time, put_time, 100)]
                linear_interpolate_torque2 = np.array(linear_interpolate.linear_interpolate_joint(q_grab_rad, q_up_rad, 100)) * 180 / math.pi
                
                # 4 到放置点
                put_end_time = put_time + time_gap
                q_put_rad = np.array(self.config["put_joint_pose"]["left_hand"]) * math.pi / 180
                linear_interpolate_time3 = [[e] for e in linear_interpolate.linear_interpolate_time(put_time, put_end_time, 100)]
                linear_interpolate_torque3 = np.array(linear_interpolate.linear_interpolate_joint(q_up_rad, q_put_rad, 100)) * 180 / math.pi
                
                # 5 放下
                l_hand_down_tact_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.config["preset_action_path"]["left_hand"]["hand_down"])
                self.bezier_wrap.load_pre_bezier_traj(l_hand_down_tact_path)
                curves = self.bezier_wrap.get_standard_bezier_line_format()
                curves = curves[0:14]
                points_all = self.bezier_wrap.get_traj_points_after_interpolate(curves, interpolate_point_num)
                time_list_down, torque_list_down = self.bezier_wrap.get_traj_points_after_seperate_time_q(points_all)
                time_list_down = [[e[0]+put_end_time] for e in time_list_down] # 时间列表要加上前面的时间
                
                # 6 合并轨迹
                time_list = time_list_up + linear_interpolate_time1 + linear_interpolate_time2 + linear_interpolate_time3 + time_list_down
                torque_list = torque_list_up + linear_interpolate_torque1.tolist() + linear_interpolate_torque2.tolist() + linear_interpolate_torque3.tolist() + torque_list_down
                
                # 5. 按照新曲线执行
                self.kuavo.move_with_trajactory(time_list=time_list, torque_list=torque_list)
                
                tag_thread = Thread(target=self.hand_control, args=(1, grab_time, grab_up_time-1, put_end_time-0.5, put_end_time+3))
                tag_thread.start()
            # 6. 手臂运动到位后关闭灵巧手
            print("- Construct Success! -")
            return 0, time_list_down[-1][0]
        except Exception as e:
            print("Arm Service Fail: ", e)
            return False, None
    
    # CAUTION: 传入的时间应该是总时间，而不是sleep的时间
    def hand_control(self, which_hand, thumb_time, grab_time, put_time, zero_time):
        print("enter run fix trajectory")
        zeros = [0,0,0,0,0,0]
        thumb = [0,100,0,0,0,0]
        fist = [100,100,100,100,55,50]
        # left hand
        if which_hand == 1:
            print("\n- Start execute trajectory {}".format("left"))
            rospy.sleep(thumb_time)
            self.kuavo.publish_hand_position(thumb,zeros)
            rospy.sleep(grab_time-thumb_time)
            self.kuavo.publish_hand_position(fist, zeros)
            rospy.sleep(1)
            self.kuavo.publish_hand_position(fist, zeros)
            rospy.sleep(put_time-grab_time-1)
            self.kuavo.publish_hand_position(thumb, zeros)
            rospy.sleep(zero_time-put_time)
            self.kuavo.publish_hand_position(zeros, zeros)
        else:
            print("\n- Start execute trajectory {}".format("right"))
            rospy.sleep(thumb_time)
            self.kuavo.publish_hand_position(zeros, thumb)
            rospy.sleep(grab_time-thumb_time)
            self.kuavo.publish_hand_position(zeros, fist)
            rospy.sleep(1)
            self.kuavo.publish_hand_position(zeros, fist)
            rospy.sleep(put_time-grab_time-1)
            self.kuavo.publish_hand_position(zeros, thumb)
            rospy.sleep(zero_time-put_time)
            self.kuavo.publish_hand_position(zeros, zeros)
        
        # 将有apriltag的标志位归false
        self._tag_position = {}
    
    def judge_boundary(self, target, which_hand):
        hand = "right_hand"
        if which_hand == 1:
            hand = "left_hand"
        if target[0] < self.config["boundary"][hand]["x"][0] or target[0] > self.config["boundary"][hand]["x"][1]:
            return False
        if target[1] < self.config["boundary"][hand]["y"][0] or target[0] > self.config["boundary"][hand]["y"][1]:
            return False
        return True


if __name__ == '__main__':
    rospy.init_node('arm_service')
    arm_service = ArmService()
    rospy.spin()
    
    # 软件导出来的是角度；IK相关的是弧度