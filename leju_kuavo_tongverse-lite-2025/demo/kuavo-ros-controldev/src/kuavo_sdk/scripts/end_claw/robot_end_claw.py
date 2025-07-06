#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import time
import rospy
from kuavo_sdk.srv import controlLejuClaw, controlLejuClawRequest, controlLejuClawResponse
from kuavo_sdk.msg import lejuClawState

claw_state = [lejuClawState.kUnknown, lejuClawState.kUnknown]

def call_leju_claw_client(pos, vel, effort):
    try:
        req = controlLejuClawRequest()
        req.data.name = ['left_claw', 'right_claw']
        req.data.position = pos
        req.data.velocity = vel
        req.data.effort = effort
        rospy.wait_for_service('/control_robot_leju_claw')
        control_leju_claw = rospy.ServiceProxy('/control_robot_leju_claw', controlLejuClaw)
        res = control_leju_claw(req)
        # print(f"res.result:{res.success}, message:{res.message}")
    except rospy.ServiceException as e:
        print("Service call failed:", e)
        return False
    
def leju_calw_state_callback(msg):
    global claw_state 
    claw_state = msg.state

if __name__ == '__main__':
    rospy.init_node('leju_claw_client_node')
    claw_state_sub = rospy.Subscriber('/leju_claw_state', lejuClawState, leju_calw_state_callback)
    time.sleep(1)
    
    call_leju_claw_client([10.0, 10.0], [50, 50], [1.0, 1.0]) 

    option_open = False
    while not rospy.is_shutdown():
        # wait for claw state to be reached
        if not (claw_state[0] >= lejuClawState.kReached and claw_state[1] >= lejuClawState.kReached):
            continue
        print("claw state:", claw_state)
        if option_open:
            print("open claw")
            call_leju_claw_client([10.0, 10.0], [50, 50], [1.0, 1.0]) 
            option_open = False
        else:       
            print("close claw")
            call_leju_claw_client([90, 90], [50, 50], [1.0, 1.0])
            option_open = True
        time.sleep(2.5)    