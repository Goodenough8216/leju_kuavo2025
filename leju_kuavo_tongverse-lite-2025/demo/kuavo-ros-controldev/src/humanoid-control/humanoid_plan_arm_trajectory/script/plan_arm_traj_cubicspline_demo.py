#!/usr/bin/env python3

import rospy
import json
import math
import numpy as np
from humanoid_plan_arm_trajectory.srv import planArmTrajectoryCubicSpline, planArmTrajectoryCubicSplineRequest
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from kuavo_msgs.srv import changeArmCtrlMode, changeArmCtrlModeRequest
from kuavo_msgs.msg import sensorsData
# from ocs2_msgs.msg import mpc_observation

current_arm_joint_state = []

def deg_to_rad(deg):
    return math.radians(deg)

def sensors_data_callback(msg):
    global current_arm_joint_state
    current_arm_joint_state = msg.joint_data.joint_q[12:26]
    current_arm_joint_state = [round(pos, 2) for pos in current_arm_joint_state]


# def mpc_obs_callback(msg):
#     global current_arm_joint_state
#     current_arm_joint_state = msg.state.value[24:]
#     current_arm_joint_state = [round(pos, 2) for pos in current_arm_joint_state]

positions = [

    [deg_to_rad(angle) for angle in [0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
    [deg_to_rad(angle) for angle in [20,0,0,-30,0,0,0,20,0,0,-30,0,0,0]],
    [deg_to_rad(angle) for angle in [20,0,0,-30,0,0,0,-30,0,30,-88,8,-22,-4]],
    [deg_to_rad(angle) for angle in [20,0,0,-30,0,0,0,-30,-25,-54,-15,-6,-22,-4]],
    [deg_to_rad(angle) for angle in [10,10,-20,-70,0,0,-24,-30,-25,-54,-15,-6,-22,-4]],
    [deg_to_rad(angle) for angle in [14,20,33,-35,76,-18,3.5,-30,-25,-54,-15,-6,-22,-4]],
    [deg_to_rad(angle) for angle in [20,0,0,-30,0,0,0,20,0,0,-30,0,0,0]],
    [deg_to_rad(angle) for angle in [0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
]
position_size = len(positions)
times = [3 + 1.5 * i for i in range(position_size)]
joint_state = JointState()

def traj_callback(msg):
    global joint_state
    if len(msg.points) == 0:
        return
    point = msg.points[0]
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
    joint_state.position = [math.degrees(pos) for pos in point.positions[:14]]
    joint_state.velocity = [math.degrees(vel) for vel in point.velocities[:14]]
    joint_state.effort = [0] * 14

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

def plan_arm_traj_cubicspline_demo():
    rospy.wait_for_service('/cubic_spline/plan_arm_trajectory')
    plan_arm_traj_cubicspline = rospy.ServiceProxy('/cubic_spline/plan_arm_trajectory', planArmTrajectoryCubicSpline)
    request = planArmTrajectoryCubicSplineRequest()
    joint_trajectory = JointTrajectory()
    print(times)
    print(positions[0])
    for i in range(len(times)):
        joint_trajectory.points.append(JointTrajectoryPoint())
        joint_trajectory.points[-1].positions = positions[i]
        joint_trajectory.points[-1].time_from_start = rospy.Duration(times[i])
    request.joint_trajectory = joint_trajectory
    request.joint_trajectory.joint_names = ["l_arm_pitch", "l_arm_roll", "l_arm_yaw", "l_forearm_pitch", "l_hand_yaw", "l_hand_pitch", "l_hand_roll", "r_arm_pitch", "r_arm_roll", "r_arm_yaw", "r_forearm_pitch", "r_hand_yaw", "r_hand_pitch", "r_hand_roll"]
    response = plan_arm_traj_cubicspline(request)

    return response.success

def main():
    rospy.init_node('arm_trajectory_cubicspline_demo')
    traj_sub = rospy.Subscriber('/cubic_spline/arm_traj', JointTrajectory, traj_callback, queue_size=1, tcp_nodelay=True)
    kuavo_arm_traj_pub = rospy.Publisher('/kuavo_arm_traj', JointState, queue_size=1, tcp_nodelay=True)
    # mpc_obs_sub = rospy.Subscriber('/humanoid_mpc_observation', mpc_observation, mpc_obs_callback)
    sensor_data_sub = rospy.Subscriber(
            '/sensors_data_raw', 
            sensorsData, 
            sensors_data_callback, 
            queue_size=1, 
            tcp_nodelay=True
    )
    call_change_arm_ctrl_mode_service(2)
    
    while len(current_arm_joint_state) != 0:
        break
    times.insert(0, 2)
    positions.insert(0, current_arm_joint_state)
    success = plan_arm_traj_cubicspline_demo()
    if success:
        rospy.loginfo("Arm trajectory planned successfully")
    else:
        rospy.logerr("Failed to plan arm trajectory")

    rate = 100
    while not rospy.is_shutdown():
        try:
            global joint_state
            if len(joint_state.position) == 0:
                continue
            kuavo_arm_traj_pub.publish(joint_state)
        except Exception as e:
            rospy.logerr(f"Failed to publish arm trajectory: {e}")
        except KeyboardInterrupt:
            break
        rospy.sleep(1/rate)

if __name__ == "__main__":
    main()
