#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import math
import numpy as np

class JointError:
    def __init__(self):
        # data
        self.ik_joint = np.zeros(14)
        self.mpc_joint = np.zeros(14)
        self.estimate_joint = np.zeros(14)

        self.sub_ik_joint = rospy.Subscriber('/kuavo_arm_traj', JointState, self.ikJointCallback)
        self.sub_mpc_jonit = rospy.Subscriber('/humanoid_controller/optimizedState_mrt/joint_pos', Float64MultiArray, self.mpcJointCallback)
        self.sub_estimate = rospy.Subscriber('/state_estimate/joint/pos', Float64MultiArray, self.estimateCallback)


        self.pub_ik_real = rospy.Publisher('/monitor/ik_real_joint_error', Float64MultiArray, queue_size=10)
        self.pub_ik_mpc = rospy.Publisher('/monitor/ik_mpc_joint_error', Float64MultiArray, queue_size=10)

    def np_to_float64multiarray(self, data):
        msg = Float64MultiArray()
        msg.data = data.tolist()
        return msg

    def ikJointCallback(self, msg):
        self.ik_joint = msg.position
        self.ik_joint = np.array(self.ik_joint)
        self.ik_joint = np.deg2rad(self.ik_joint)
        # print(self.ik_joint)
    
    def mpcJointCallback(self, msg):
        # print(msg.data)
        mpc_data = np.array(msg.data)
        self.mpc_joint = mpc_data[-14:]
        # print(self.mpc_joint)
        # print(len(self.mpc_joint))

        error_ik_mpc = self.ik_joint - self.mpc_joint
        error_ik_mpc = np.rad2deg(error_ik_mpc)
        error_ik_mpc = np.abs(error_ik_mpc)
        self.pub_ik_mpc.publish(self.np_to_float64multiarray(error_ik_mpc))

    def estimateCallback(self, msg):        
        est_data = np.array(msg.data)
        self.estimate_joint = est_data[-14:]
        # print(self.estimate_joint)
        error_ik_real = self.ik_joint - self.estimate_joint
        error_ik_real = np.rad2deg(error_ik_real)
        error_ik_real = np.abs(error_ik_real)
        self.pub_ik_real.publish(self.np_to_float64multiarray(error_ik_real))


if __name__ == '__main__':
    rospy.init_node('joint_error_monitor')
    print("JointError node started")
    joint_error = JointError()
    rospy.spin()
