#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import time
import rospy
from kuavo_msgs.msg import lejuClawState, lejuClawCommand

claw_state = [lejuClawState.kUnknown, lejuClawState.kUnknown]

pub = rospy.Publisher('/leju_claw_command', lejuClawCommand, queue_size=10)
def pub_leju_claw_comand(pos, vel, effort):
    msg = lejuClawCommand()
    msg.data.name = ['left_claw', 'right_claw']
    msg.data.position = pos
    msg.data.velocity = vel
    msg.data.effort = effort
    pub.publish(msg)
    
def leju_calw_state_callback(msg):
    global claw_state 
    claw_state = msg.state

if __name__ == '__main__':
    rospy.init_node('leju_claw_client_node')
    claw_state_sub = rospy.Subscriber('/leju_claw_state', lejuClawState, leju_calw_state_callback)
    time.sleep(1)

    pub_leju_claw_comand([10.0, 10.0], [90, 90], [1.0, 1.0]) 

    option_open = False
    # while not rospy.is_shutdown():
    #     # wait for claw state to be reached
    #     if not (claw_state[0] >= lejuClawState.kReached and claw_state[1] >= lejuClawState.kReached):
    #         continue
    #     print("claw state:", claw_state)
    #     if option_open:
    #         print("open claw")
    #         pub_leju_claw_comand([10.0, 10.0], [90, 90], [1.0, 1.0]) 
    #         option_open = False
    #     else:       
    #         print("close claw")
    #         pub_leju_claw_comand([90, 90], [90, 90], [1.0, 1.0])
    #         option_open = True
    #     time.sleep(2.5)    

    while not rospy.is_shutdown():
        if option_open:
            print("open claw")
            pub_leju_claw_comand([10.0, 10.0], [90, 90], [1.0, 1.0]) 
            option_open = False
        else:       
            print("close claw")
            pub_leju_claw_comand([90, 90], [90, 90], [1.0, 1.0])
            option_open = True
        time.sleep(0.1)    