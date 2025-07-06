#!/usr/bin/env python

import rospy
from kuavo_sdk.srv import changeArmCtrlMode, changeArmCtrlModeRequest, changeArmCtrlModeResponse


def set_arm_control_mode(mode):
    """
    设置OCS2手臂控制模式。
    修改手臂控制模式，control_mode 有三种模式
    - 0: keep pose 保持姿势 
    - 1: auto_swing_arm 行走时自动摆手，切换到该模式会自动运动到摆手姿态
    - 2: external_control 外部控制，手臂的运动由外部控制
    :param mode: 要设置的控制模式
    """
    try:
        # 创建服务代理，用于与服务通信
        arm_traj_change_mode_client = rospy.ServiceProxy("/arm_traj_change_mode", changeArmCtrlMode)

        # 创建请求对象
        request = changeArmCtrlModeRequest()
        request.control_mode = mode  # 设置请求的控制模式

        # 发送请求并接收响应
        response = arm_traj_change_mode_client(request)

        if response.result:
            # 如果响应结果为真，表示成功更改控制模式
            rospy.loginfo(f"Successfully changed arm control mode to {mode}: {response.message}")
        else:
            # 如果响应结果为假，表示更改控制模式失败
            rospy.logwarn(f"Failed to change arm control mode to {mode}: {response.message}")

    except rospy.ServiceException as e:
        # 如果服务调用失败，记录错误信息
        rospy.logerr(f"Service call failed: {e}")

def main():
    """
    主函数，初始化ROS节点并调用设置手臂控制模式的函数。
    """
    # 初始化ROS节点
    rospy.init_node('arm_control_mode_client', anonymous=True)

    mode = 2
    set_arm_control_mode(mode)


if __name__ == '__main__':
    # 如果此脚本是主程序，则调用main函数
    main()