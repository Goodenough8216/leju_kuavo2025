#!/usr/bin/env python3
import rospy
import math
from sensor_msgs.msg import Joy
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from enum import Enum

"""
控制模式枚举
"""
class ControlMode(Enum):
    POSE = 0        # 姿态控制模式
    VELOCITY = 1    # 速度控制模式

class KuavoSDK:
    def __init__(self):
        self.pub_cmd_vel = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.pub_torso_pose = rospy.Publisher('/cmd_pose', Twist, queue_size=10)

    def publish_command(self, mode:ControlMode, x, y, z, yaw):
        """
        根据给定的控制模式发布机器人命令。

        :param mode: ControlMode 枚举值，指定使用速度还是姿态控制
        :param x: float, 沿x轴的速度或位置
        :param y: float, 沿y轴的速度或位置
        :param z: float, 沿z轴的速度或位置
        :param yaw: float, 旋转角速度或角度
        """
        if mode == ControlMode.POSE:
            twist = Twist()
            # 发布torso的位姿
            twist.linear.x = x  
            twist.linear.y = y  
            twist.linear.z = z 
            twist.angular.z = yaw  
            self.pub_torso_pose.publish(twist)

        elif mode == ControlMode.VELOCITY:
            twist = Twist()
            twist.linear.x = x  # 线速度，沿x轴
            twist.linear.y = y  # 线速度，沿y轴
            twist.linear.z = z  # 线速度，沿z轴
            twist.angular.z = yaw
            self.pub_cmd_vel.publish(twist)
        
def circle_trajectory_case(kuavo:KuavoSDK, radius, yaw_v):
    """
    控制机器人执行圆形轨迹运动。

    :param kuavo: KuavoSDK 实例
    :param radius: float, 圆形轨迹的半径
    :param yaw_v: float, 旋转角速度
    """

    print("----- start circle trajectory")
    x_v = radius * yaw_v
    while not rospy.is_shutdown():
        kuavo.publish_command(ControlMode.VELOCITY, x_v, 0, 0, yaw_v)

def square_trajectory_case(kuavo: KuavoSDK, side_length, x_v):
    """
    控制机器人执行方形轨迹运动。

    :param kuavo: KuavoSDK 实例
    :param side_length: float, 方形边长
    :param x_v: float, 沿x轴前进的速度
    """

    print("----- start square trajectory")
    
    yaw_v = 15  # degree
    j = 0
    while j < 4 and not rospy.is_shutdown(): 
        # go straight
        i = 0
        while i < int(side_length/x_v) and not rospy.is_shutdown(): 
            kuavo.publish_command(ControlMode.VELOCITY, x_v, 0, 0, 0)
            i+=1
            rospy.sleep(1.0)

        # turn left 90 degree
        i = 0
        while i < int(90/yaw_v) and not rospy.is_shutdown():   
            kuavo.publish_command(ControlMode.VELOCITY, 0, 0, 0, math.radians(yaw_v))
            rospy.sleep(1.0)
            i+=1
        rospy.sleep(2.0)
        j += 1

def s_shape_trajectory_case(kuavo: KuavoSDK, radius, yaw_v):
    """
    控制机器人执行S形轨迹运动。

    :param kuavo: KuavoSDK 实例
    :param radius: float, 半圆的半径
    :param yaw_v: float, 旋转角速度
    """

    print("----- start S shape trajectory")
    x_v = radius * yaw_v
    # 第一个半圆向左转
    i = 0
    while i < (math.pi / yaw_v) and not rospy.is_shutdown():
        kuavo.publish_command(ControlMode.VELOCITY, x_v, 0, 0, yaw_v)
        rospy.sleep(1.0)
        i += 1
    
    # 直行一段距离作为过渡
    transition_distance = radius/3.0
    i = 0
    while i < (transition_distance / x_v) and not rospy.is_shutdown():
        kuavo.publish_command(ControlMode.VELOCITY, x_v, 0, 0, 0)
        rospy.sleep(1.0)
        i += 1

    # 第二个半圆向右转
    i = 0
    while i < (math.pi / yaw_v) and not rospy.is_shutdown():
        kuavo.publish_command(ControlMode.VELOCITY, x_v, 0, 0, -yaw_v)
        rospy.sleep(1.0)
        i += 1

if __name__ == '__main__':
    rospy.init_node('walk_trajectory_demo_node')
    kuavo = KuavoSDK()
    print("========================Init========================")
    
    # 正方形轨迹
    # square_trajectory_case(kuavo, side_length=2.0, x_v=0.2)

    # S 曲线
    # 尽可能让速度小一些
    # s_shape_trajectory_case(kuavo, radius=0.75, yaw_v=0.25)
    
    # 圆型轨迹
    # 尽可能让速度小一些
    circle_trajectory_case(kuavo, radius=0.7, yaw_v=0.25)
