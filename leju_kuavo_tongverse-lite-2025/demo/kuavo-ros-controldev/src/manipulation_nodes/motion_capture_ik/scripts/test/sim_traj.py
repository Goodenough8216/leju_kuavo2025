#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from std_msgs.msg import Float32, Float32MultiArray
from sensor_msgs.msg import JointState
import numpy as np
import time

pub = rospy.Publisher("/kuavo_arm_traj", JointState, queue_size=10)

def publish_joint_states(q_now):
    msg = JointState()
    msg.name = ["arm_joint_" + str(i) for i in range(1, 14)]
    msg.header.stamp = rospy.Time.now()
    msg.position = np.array(q_now)
    # msg.position = 180.0 / np.pi * np.array(q_now)
    pub.publish(msg)

if __name__ == '__main__':
    rospy.init_node('sim_traj', anonymous=True)
    rate = rospy.Rate(10) # 10hz
    q_list = []
    q0 = [0.0]*14
    q1 = [0.0]*14
    q1[:7] = [-30, 60, 0, -30, 0, -30, 30]
    num = 90
    for i in range(num):
        q_tmp = [0.0]*14
        for j in range(14):
            q_tmp[j] = q0[j] + i/float(num)*(q1[j] - q0[j])
        q_list.append(q_tmp)
    for q in q_list:
        publish_joint_states(q)
        print(f"q: {q}")
        time.sleep(0.02)