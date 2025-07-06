#!/usr/bin/env python3

import rospy
import numpy as np
from kuavo_msgs.msg import headBodyPose

def get_head_body_pose_msg(delta_height, body_pitch):
    msg = headBodyPose()
    msg.body_height = max(-0.4, min(delta_height, 0.2))
    msg.body_yaw = 0.0
    msg.body_pitch = max(3*np.pi/180.0, min(body_pitch, 60*np.pi/180.0))
    return msg

if __name__ == '__main__':
    rospy.init_node('torso_control_demo')
    head_body_pose_puber = rospy.Publisher('/kuavo_head_body_orientation', headBodyPose, queue_size=10)
    rospy.sleep(1)
    
    delta_height = -0.1
    body_pitch = 20.0 * np.pi/180.0
    msg = get_head_body_pose_msg(delta_height=delta_height, body_pitch=body_pitch)
    head_body_pose_puber.publish(msg)
