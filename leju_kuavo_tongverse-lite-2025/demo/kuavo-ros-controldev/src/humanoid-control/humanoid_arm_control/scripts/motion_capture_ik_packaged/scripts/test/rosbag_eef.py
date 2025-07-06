import rosbag
import sys
import os

import numpy as np
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
tools_dir = os.path.abspath(os.path.join(current_dir, '..', 'tools'))
print(tools_dir)
sys.path.append(tools_dir)

from drake_trans import *

def read_rosbag(bag_file, topic_list):
    # 打开ROS bag文件
    bag = rosbag.Bag(bag_file)

    # 遍历bag文件中的消息
    rpy1, rpy2 = None, None
    mat1 = None
    mat2 = None
    # 初始化存储rpy的列表
    roll_values = []
    pitch_values = []
    yaw_values = []
    rot_agl = []
    for topic, msg, t in bag.read_messages(topics=topic_list):
        if topic == topic_list[0]:
            rpy1 = quaternion_to_RPY(msg.left_pose.quat_xyzw)
            mat1 = quaternion_to_matrix(msg.left_pose.quat_xyzw)
        if topic == topic_list[1]:
            rpy2 = quaternion_to_RPY(msg.left_pose.quat_xyzw)
            mat2 = quaternion_to_matrix(msg.left_pose.quat_xyzw)
        if mat1 is not None and mat2 is not None:
            delta_mat = np.dot(mat1.T, mat2)
            rpy = matrix_to_rpy(delta_mat)
            axis, angle = matrix_to_axis_angle(delta_mat)
            rpy = rpy2 - rpy1
            print(f"axis: {axis}, angle: {angle}")
            # print(f"rpy delta: {rpy}")
            # 存储rpy的值
            rot_agl.append(angle)
            
            roll_values.append(rpy[0])
            pitch_values.append(rpy[1])
            yaw_values.append(rpy[2])
    # 关闭bag文件
    bag.close()
    # 使用matplotlib绘制rpy的三条线
    time_indices = range(len(roll_values))

    plt.plot(time_indices, rot_agl, label="Axis Angle[norm]")
    plt.plot(time_indices, roll_values, label="Roll")
    plt.plot(time_indices, pitch_values, label="Pitch")
    plt.plot(time_indices, yaw_values, label="Yaw")

    # 显示图例和图形
    plt.legend()
    plt.xlabel("Time Index")
    plt.ylabel("RPY Value")
    plt.title("RPY Delta Over Time")
    plt.show()

if __name__ == "__main__":
    # 获取当前目录
    current_directory = os.getcwd()
    
    # 获取当前目录下的所有bag文件
    bag_files = [f for f in os.listdir(current_directory) if f.endswith('.bag')]
    
    if not bag_files:
        print("当前目录下没有ROS bag文件。")
        sys.exit(1)
    
    # 选择第一个bag文件
    bag_file = os.path.join(current_directory, bag_files[0])
    
    # 设置要读取的话题
    topic_list = ["/drake_ik/cmd_arm_hand_pose", "/drake_ik/real_arm_hand_pose"]  # 替换为你要读取的话题名称
    
    # 读取bag文件并打印消息
    read_rosbag(bag_file, topic_list)
