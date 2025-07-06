#!/usr/bin/env python
#encoding: utf-8
import time
import rospy
from ocs2_msgs.msg import mpc_flattened_controller
import matplotlib.pyplot as plt

draw = False
draw_data = None

def mpc_policy_callback(data):
    global draw, draw_data
    if not draw:
        return
    draw = False
    draw_data = data
    print("update plot")

def subscribe_to_mpc_policy_topic():
    global draw
    rospy.init_node('mpc_policy_subscriber', anonymous=True)
    rospy.Subscriber('/humanoid_mpc_policy', mpc_flattened_controller, mpc_policy_callback)
    fig, ax = plt.subplots()
    ax.set_xlabel('step')
    ax.set_ylabel('Trajectory')
    ax.grid(True)  # 开启网格
    ax.legend()  # 显示图例
    while True:
        user_input = input("按回车键绘制图表，输入'q'退出：")
        if user_input.lower() == "q":
            break
        draw = True
        while draw_data is None:
            time.sleep(0.01)
        ax.clear()
        
        # 开始和结束的位置，目前是取出z
        start_clo = 8
        end_clo = 9
        for i in range(start_clo,end_clo):
            values = [state.value[i] for state in draw_data.stateTrajectory]
            ax.plot(values, label=f'Value {i}')
        
        plt.draw()
        plt.show(block=False)
        plt.pause(0.001) 

if __name__ == '__main__':
    subscribe_to_mpc_policy_topic()
