#!/usr/bin/env python3

import rospy
import json
import math
import numpy as np
from humanoid_plan_arm_trajectory.srv import planArmTrajectoryBezierCurve, planArmTrajectoryBezierCurveRequest
from humanoid_plan_arm_trajectory.msg import bezierCurveCubicPoint, jointBezierTrajectory
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory
from kuavo_msgs.srv import changeArmCtrlMode, changeArmCtrlModeRequest
from kuavo_msgs.msg import sensorsData
# from ocs2_msgs.msg import mpc_observation
import math
INIT_ARM_POS = [20, 0, 0, -30, 0, 0, 0, 20, 0, 0, -30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
START_FRAME_TIME = 0
END_FRAME_TIME = 22

x_shift = START_FRAME_TIME - 1
joint_state = JointState()
current_arm_joint_state = []


def sensors_data_callback(msg):
    global current_arm_joint_state
    current_arm_joint_state = msg.joint_data.joint_q[12:26]
    current_arm_joint_state = [round(pos, 2) for pos in current_arm_joint_state]
    current_arm_joint_state.extend([0] * 14)


# def mpc_obs_callback(msg):
#     global current_arm_joint_state
#     current_arm_joint_state = msg.state.value[24:]
#     current_arm_joint_state = [round(pos, 2) for pos in current_arm_joint_state]
#     current_arm_joint_state.extend([0] * 14)

import numpy as np

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
    
def load_json_file(file_path):
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except IOError as e:
        rospy.logerr(f"Error reading file {file_path}: {e}")
        return None

def add_init_frame(frames):
    action_data = {}
    for frame in frames:
        servos, keyframe, attribute = frame["servos"], frame["keyframe"], frame["attribute"]
        for index, value in enumerate(servos):
            key = index + 1
            if key == 15:
                break
            if key not in action_data:
                action_data[key] = []
                if keyframe != 0 and len(action_data[key]) == 0:
                    if key <= len(INIT_ARM_POS):
                        action_data[key].append([
                            [0, math.radians(INIT_ARM_POS[key-1])],
                            [0, math.radians(INIT_ARM_POS[key-1])],
                            [0, math.radians(INIT_ARM_POS[key-1])],
                ])
            if value is not None:
                CP = attribute[str(key)]["CP"]
                left_CP, right_CP = CP
                action_data[key].append([
                    [round(keyframe/100, 1), math.radians(value)],
                    [round((keyframe+left_CP[0])/100, 1), math.radians(value+left_CP[1])],
                    [round((keyframe+right_CP[0])/100, 1), math.radians(value+right_CP[1])],
                ])
    return action_data

def frames_to_custom_action_data(frames):
    action_data = {}
    for frame in frames:
        servos, keyframe, attribute = frame["servos"], frame["keyframe"], frame["attribute"]
        for index, value in enumerate(servos):
            key = index + 1
            if key not in action_data:
                action_data[key] = []
                if keyframe != 0 and len(action_data[key]) == 0:
                    if key <= len(INIT_ARM_POS):
                        action_data[key].append([
                            [0, math.radians(INIT_ARM_POS[key-1])],
                            [0, 0],
                            [0, math.radians(INIT_ARM_POS[key-1])],
                ])
            if value is not None:
                CP = attribute[str(key)]["CP"]
                left_CP, right_CP = CP
                action_data[key].append([
                    [round(keyframe/100, 1) - x_shift, math.radians(value)],
                    [round((keyframe+left_CP[0])/100, 1) - x_shift, math.radians(value+left_CP[1])],
                    [round((keyframe+right_CP[0])/100, 1) - x_shift, math.radians(value+right_CP[1])],
                ])
    return action_data


def filter_data(action_data):
    filtered_action_data = {}
    for key, frames in action_data.items():
        filtered_frames = []
        found_start = False
        skip_next = False
        for i in range(-1, len(frames)):
            frame = frames[i]
            if i == len(frames) - 1:
                next_frame = frame
            else:
                next_frame = frames[i+1]
            end_time = next_frame[0][0]
            if not found_start and end_time >= START_FRAME_TIME:
                found_start = True
                p0 = np.array([0, current_arm_joint_state[key-1]])
                p3 = np.array([next_frame[0][0] - x_shift, next_frame[0][1]])
                
                # Calculate control points for smooth transition
                curve_length = np.linalg.norm(p3 - p0)
                p1 = p0 + curve_length * 0.25 * np.array([1, 0])  # Move 1/4 curve length to the right
                p2 = p3 - curve_length * 0.25 * np.array([1, 0])  # Move 1/4 curve length to the left
                
                # Create new frame
                frame1 = [
                    p0.tolist(),
                    p0.tolist(),
                    p1.tolist()
                ]
                
                # Modify next_frame's left control point
                next_frame[1] = p2.tolist()
                
                filtered_frames.append(frame1)
                skip_next = True
            
            if found_start:
                if skip_next:
                    skip_next = False
                    continue
                end_point = [round(frame[0][0] - x_shift, 1), round(frame[0][1], 1)]
                left_control_point = [round(frame[1][0] - x_shift, 1), round(frame[1][1], 1)]
                right_control_point = [round(frame[2][0] - x_shift, 1), round(frame[2][1], 1)]
                filtered_frames.append([end_point, left_control_point, right_control_point])

        filtered_action_data[key] = filtered_frames
    return filtered_action_data

def create_bezier_request(action_data):
    req = planArmTrajectoryBezierCurveRequest()
    for key, value in action_data.items():
        msg = jointBezierTrajectory()
        for frame in value:
            point = bezierCurveCubicPoint()
            point.end_point, point.left_control_point, point.right_control_point = frame
            msg.bezier_curve_points.append(point)
        req.multi_joint_bezier_trajectory.append(msg)
    req.start_frame_time = START_FRAME_TIME
    req.end_frame_time = END_FRAME_TIME
    req.joint_names = ["l_arm_pitch", "l_arm_roll", "l_arm_yaw", "l_forearm_pitch", "l_hand_yaw", "l_hand_pitch", "l_hand_roll", "r_arm_pitch", "r_arm_roll", "r_arm_yaw", "r_forearm_pitch", "r_hand_yaw", "r_hand_pitch", "r_hand_roll"]
    return req

def plan_arm_trajectory_bezier_curve_client(req):
    service_name = '/bezier/plan_arm_trajectory'
    rospy.wait_for_service(service_name)
    try:
        plan_service = rospy.ServiceProxy(service_name, planArmTrajectoryBezierCurve)
        res = plan_service(req)
        return res.success
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")
        return False

def stop_arm_trajectory_client():
    service_name = '/stop_plan_arm_trajectory'
    rospy.wait_for_service(service_name)
    try:
        stop_service = rospy.ServiceProxy(service_name, stopPlanArmTrajectory)
        res = stop_service()
        return res.result
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")
        return False

def main():
    rospy.init_node('arm_trajectory_bezier_demo')
    traj_sub = rospy.Subscriber('/bezier/arm_traj', JointTrajectory, traj_callback, queue_size=1, tcp_nodelay=True)
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
    file_path = rospy.get_param('~tact_file', './action_files/welcome.tact')
    data = load_json_file(file_path)
    if not data:
        return

    # rospy.sleep(2) # ensure the mpc_obs_sub has enough time to get the current arm joint state
    action_data = add_init_frame(data["frames"])
    req = create_bezier_request(filter_data(action_data))
    # req = create_bezier_request(action_data)

    rospy.loginfo("Planning arm trajectory...")
    success = plan_arm_trajectory_bezier_curve_client(req)
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

