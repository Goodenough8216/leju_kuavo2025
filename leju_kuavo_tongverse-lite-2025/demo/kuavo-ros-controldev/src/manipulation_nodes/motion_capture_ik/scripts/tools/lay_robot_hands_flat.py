#!/usr/bin/env python3
import math
import numpy as np
import rospy
from sensor_msgs.msg import JointState
from kuavo_msgs.srv import changeArmCtrlMode, changeArmCtrlModeRequest
from kuavo_msgs.msg import sensorsData
import time

joint_state = JointState()
joint_state.name = [
    "l_arm_pitch",
    "l_arm_roll",
    "l_arm_yaw",
    "l_forearm_pitch",
    "l_hand_yaw",
    "l_hand_pitch",
    "l_hand_roll",
    "r_arm_pitch",
    "r_arm_roll",
    "r_arm_yaw",
    "r_forearm_pitch",
    "r_hand_yaw",
    "r_hand_pitch",
    "r_hand_roll",
]


cur_joint_pos = [20, 0, 0, -30, 0, 0, 0, 20, 0, 0, -30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
cur_joint_pos = [math.radians(pos) for pos in cur_joint_pos]


def sensors_data_callback(msg):
    global cur_joint_pos
    cur_joint_pos = msg.joint_data.joint_q[12:26]
    cur_joint_pos = [round(pos, 2) for pos in cur_joint_pos]


target_joint_pos = [-0.55, 0.15, 0.00, -0.35, -0.35, -1.20, -0.4, -0.55, -0.15, 0.00, -0.35, 0.35, -1.20, 0.4]

start_time = 0
end_time = 2
interpolation_rate = 100  # Publish at 100Hz, adjust as needed


def cubic_spline_interpolation(start_pos, end_pos, start_time, end_time, rate):
    """
    Performs cubic spline interpolation between start and end positions.

    Args:
        start_pos: Initial joint positions (list of floats).
        end_pos: Target joint positions (list of floats).
        start_time: Start time of interpolation (float).
        end_time: End time of interpolation (float).
        rate: Interpolation rate in Hz

    Returns:
      A list of interpolated joint positions, where each element is list of floats.
    """

    num_steps = int((end_time - start_time) * rate)
    times = np.linspace(start_time, end_time, num_steps + 1)
    interpolated_positions = []

    for i in range(len(start_pos)):
      y0 = start_pos[i]
      y1 = end_pos[i]
      x0 = start_time
      x1 = end_time

      # Calculate coefficients for cubic spline
      a = y0
      b = 0
      c = (3 * (y1 - y0) / (x1 - x0)**2)
      d = (-2 * (y1-y0) / (x1-x0)**3)

      interpolated_pos = [a + b * (t - x0) + c * (t - x0)**2 + d * (t - x0)**3 for t in times]
      interpolated_positions.append(interpolated_pos)
    return np.array(interpolated_positions).T.tolist()

def call_change_arm_ctrl_mode_service(arm_ctrl_mode):
    result = True
    service_name = "humanoid_change_arm_ctrl_mode"
    try:
        rospy.wait_for_service(service_name, timeout=0.5)
        change_arm_ctrl_mode = rospy.ServiceProxy(
            "humanoid_change_arm_ctrl_mode", changeArmCtrlMode
        )
        change_arm_ctrl_mode(control_mode=arm_ctrl_mode)
        rospy.loginfo("Service call successful")
    except rospy.ServiceException as e:
        rospy.loginfo("Service call failed: %s", e)
        result = False
    except rospy.ROSException:
        rospy.logerr(f"Service {service_name} not available")
        result = False
    finally:
        return result
 
def main():
    rospy.init_node("arm_trajectory_publisher")

    pub = rospy.Publisher("/kuavo_arm_traj", JointState, queue_size=10)
    rospy.Subscriber("/sensors_data_raw", sensorsData, sensors_data_callback)
    rate = rospy.Rate(interpolation_rate)
    
    call_change_arm_ctrl_mode_service(2)
    global cur_joint_pos
    # Wait for the initial sensor data to be available
    while len(cur_joint_pos) != 14 and not rospy.is_shutdown():
      rospy.loginfo("Waiting for initial sensor data...")
      rate.sleep()
    
    rospy.loginfo("Initial sensor data received, starting interpolation.")
    
    interpolated_positions = cubic_spline_interpolation(cur_joint_pos, target_joint_pos, start_time, end_time, interpolation_rate)
    
    for pos in interpolated_positions:
      if rospy.is_shutdown():
        break
        
      joint_state.position = [math.degrees(p) for p in pos]
      joint_state.header.stamp = rospy.Time.now()
      pub.publish(joint_state)
      rate.sleep()

    rospy.loginfo("Trajectory published.")    
    # time.sleep(0.5)
    # call_change_arm_ctrl_mode_service(1)

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
