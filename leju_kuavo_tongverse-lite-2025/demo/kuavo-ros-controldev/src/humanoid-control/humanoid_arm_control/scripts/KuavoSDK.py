#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy

from kuavo_msgs.msg import armTargetPoses
from kuavo_msgs.srv import changeArmCtrlMode, changeArmCtrlModeRequest, changeArmCtrlModeResponse
from data_utils import *
from kuavo_msgs.msg import sensorsData
from kuavo_msgs.msg import robotHandPosition, robotHeadMotionData

class kuavo():
    def __init__(self) -> None:
        self._traj_pub = rospy.Publisher("/kuavo_arm_target_poses", armTargetPoses, queue_size=10)
        self._arm_traj_change_mode_client = rospy.ServiceProxy("/arm_traj_change_mode", changeArmCtrlMode)
        self._pub_hand_pose = rospy.Publisher('/control_robot_hand_position', robotHandPosition, queue_size=10)
        self._pub_head_pose = rospy.Publisher('/robot_head_motion_data', robotHeadMotionData, queue_size=10)
        
    def update_joint_state_callback(self, data):
        arm_joint_data = data.joint_data.joint_q[12:]
        self.current_joint_values = arm_joint_data
        self.l_hand_info = self.arm_ik.right_hand_pose(self.current_joint_values)
        self.r_hand_info = self.arm_ik.left_hand_pose(self.current_joint_values)
        
    """ 将关节列表和时间列表发送到话题执行
    
    """
    def move_with_trajactory(self, time_list, torque_list):
        # self.set_arm_control_mode(2)
        # time.sleep(0.5)
        arm_traj_msg = armTargetPoses()
        # print(1)
        arm_traj_msg.times = self._message_time_to_series(time_list)
        # print(2)
        arm_traj_msg.values = self._message_torque_to_series(torque_list)
        # print(3)
        while self._traj_pub.get_num_connections() == 0 and not rospy.is_shutdown():
            rospy.loginfo("等待订阅者连接...")
            rospy.sleep(2)
        print("开始发布")
        self._traj_pub.publish(arm_traj_msg)
        print("=====publish success!=====")
    
    """ 切换ocs2手臂控制模式
    
    """
    def set_arm_control_mode(self, mode):
        """ 切换手臂规划模式 
        :param control_mode: uint8, # 0: keep pose, 1: auto_swing_arm, 2: external_control 
        :return: bool, 服务调用结果
        """
        rospy.wait_for_service('/arm_traj_change_mode')
        try:
            # 使用初始化时的服务代理
            request = changeArmCtrlModeRequest()
            request.control_mode = mode

            # 发送请求并接收响应
            response = self._arm_traj_change_mode_client(request)
            
            if response.result:
                rospy.loginfo(f"Successfully changed arm control mode to {mode}: {response.message}")
                # self.move_with_trajactory([0.1], [20, 0, 0, -30, 0, 0, 0, 20, 0, 0, -30, 0, 0])
            else:
                rospy.logwarn(f"Failed to change arm control mode to {mode}: {response.message}")

        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")
    
    
    """ 将关节数值转为一系列列表
    
    """
    def _message_torque_to_series(self, message_torque):
        final_message_torque_series = []
        for i in range(len(message_torque)):
            for j in range(len(message_torque[i])):
                final_message_torque_series.append(message_torque[i][j])
        # print(final_message_torque_series)
        return final_message_torque_series
    
    def _message_time_to_series(self, message_time):
        # print("time", [message_time[i][0] for i in range(len(message_time))])
        return [message_time[i][0] for i in range(len(message_time))]
    
    def set_head_target(self, yaw, pitch):
        head_target_msg = robotHeadMotionData()
        head_target_msg.joint_data = [yaw, pitch]
        self._pub_head_pose.publish(head_target_msg)
    
    def publish_hand_position(self, left_hand_pos, right_hand_pos):
        msg = robotHandPosition()
        msg.left_hand_position = left_hand_pos
        msg.right_hand_position = right_hand_pos
        self._pub_hand_pose.publish(msg)
        

if __name__ == "__main__":
    rospy.init_node("kuavoSDK_test", anonymous=True)
    kuavo_instance = kuavo()
    rospy.sleep(1)
    kuavo_instance.set_head_target(0.0, 0.0)