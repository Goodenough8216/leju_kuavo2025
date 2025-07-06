#!/usr/bin/env python3

import rospy
import time
from kuavo_msgs.msg import robotHandPosition

rospy.init_node("control_hand_demo_node", anonymous=True)
control_hand_pub = rospy.Publisher('/control_robot_hand_position', robotHandPosition, queue_size=10)

def control_hand_pose(left_hand_position, right_hand_position):
    hand_pose_msg = robotHandPosition()
    hand_pose_msg.left_hand_position = left_hand_position
    hand_pose_msg.right_hand_position = right_hand_position
    control_hand_pub.publish(hand_pose_msg)
            
if __name__ == "__main__":
    # waiting ros init.
    rospy.sleep(2.0)
    
    # ok
    print("control_robot_hand_position execute gesture: `ok`.")
    ok_pos = [60, 90, 60, 5, 5, 5]
    control_hand_pose(ok_pos, ok_pos)

    time.sleep(1.5)

    # reset
    empty_pos = [0, 0, 0, 0, 0, 0]
    control_hand_pose(empty_pos, empty_pos)
    print("control_robot_hand_position reset gesture.")