#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from kuavo_sdk.srv import fkSrv


# FK正解服务
def fk_srv_client(joint_angles):
  # 确保要调用的服务可用
  rospy.wait_for_service('/ik/fk_srv')
  try:
      # 初始化服务代理
      fk_srv = rospy.ServiceProxy('/ik/fk_srv', fkSrv)
      # 调取服务并获得响应
      fk_result = fk_srv(joint_angles)
      # 打印是否求解成功
      print("FK result:", fk_result.success)
      # 返回正解结果（手臂位置姿态等信息）
      return fk_result.hand_poses
  except rospy.ServiceException as e:
      print("Service call failed: %s"%e)


if __name__ == "__main__":
  # 初始化ROS节点
  rospy.init_node("robot_arm_fk_srv_node")

  # 创建请求对象（单位：弧度）
  joint_angles = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.38, -1.39, -0.29, -0.43, 0.0, -0.17, 0.0]

  # 调用 FK 正解服务
  hand_poses = fk_srv_client(joint_angles)

  # 打印部分正解结果
  if hand_poses is not None:
      print("left eef position:", hand_poses.left_pose.pos_xyz)
      print("\nright eef position: ", hand_poses.right_pose.pos_xyz)
  else:
      print("No hand poses returned")

