import rospy
import os
import asyncio
import json
from queue import Queue
from dataclasses import dataclass
from typing import Any, Union, Dict
import websockets
import threading
import time
import math
from utils import calculate_file_md5, frames_to_custom_action_data, get_start_end_frame_time, frames_to_custom_action_data_ocs2

from kuavo_ros_interfaces.srv import planArmTrajectoryBezierCurve, stopPlanArmTrajectory, planArmTrajectoryBezierCurveRequest, ocs2ChangeArmCtrlMode
from kuavo_ros_interfaces.msg import planArmState, jointBezierTrajectory, bezierCurveCubicPoint, robotHandPosition, robotHeadMotionData
from std_srvs.srv import Trigger
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory
from kuavo_msgs.msg import sensorsData
# Replace multiprocessing values with simple variables
plan_arm_state_progress = 0
plan_arm_state_status = False
response_queue = Queue()
active_threads: Dict[websockets.WebSocketServerProtocol, threading.Event] = {}
ACTION_FILE_FOLDER = "~/.config/lejuconfig/action_files"
g_robot_type = ""
ocs2_current_joint_state = []
joint_names = [
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
ocs2_joint_state = JointState()
ocs2_hand_state = robotHandPosition()
ocs2_head_state = robotHeadMotionData()
robot_settings = {
    "kuavo":{
        "plan_arm_trajectory_bezier_service_name": "/plan_arm_trajectory_bezier_curve",
        "stop_plan_arm_trajectory_service_name": "/stop_plan_arm_trajectory",
        "arm_traj_state_topic_name": "/robot_plan_arm_state",
    },
    "ocs2":{
        "plan_arm_trajectory_bezier_service_name": "/bezier/plan_arm_trajectory",
        "stop_plan_arm_trajectory_service_name": "/bezier/stop_plan_arm_trajectory",
        "arm_traj_state_topic_name": "/bezier/arm_traj_state",
    }
}

def call_change_arm_ctrl_mode_service(arm_ctrl_mode):
    result = True
    service_name = "humanoid_change_arm_ctrl_mode"
    try:
        rospy.wait_for_service(service_name, timeout=0.5)
        change_arm_ctrl_mode = rospy.ServiceProxy(
            "humanoid_change_arm_ctrl_mode", ocs2ChangeArmCtrlMode
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

def sensors_data_callback(msg):
    global current_arm_joint_state
    current_arm_joint_state = msg.joint_data.joint_q[12:26]
    current_arm_joint_state = [round(pos, 2) for pos in current_arm_joint_state]
    current_arm_joint_state.extend([0] * 14)

def traj_callback(msg):
    global ocs2_joint_state
    if len(msg.points) == 0:
        return
    point = msg.points[0]
    ocs2_joint_state.name = msg.joint_names
    ocs2_joint_state.position = [math.degrees(pos) for pos in point.positions[:14]]
    ocs2_joint_state.velocity = [math.degrees(vel) for vel in point.velocities[:14]]
    ocs2_joint_state.effort = [0] * 14

    ocs2_hand_state.left_hand_position = [int(math.degrees(pos)) for pos in point.positions[14:20]]
    ocs2_hand_state.right_hand_position = [int(math.degrees(pos)) for pos in point.positions[20:26]]

    ocs2_head_state.joint_data = [math.degrees(pos) for pos in point.positions[26:]]

kuavo_arm_traj_pub = None
control_hand_pub = None
control_head_pub = None

def timer_callback(event):
    global kuavo_arm_traj_pub, control_hand_pub, control_head_pub
    if g_robot_type == "ocs2" and len(ocs2_joint_state.position) > 0 and plan_arm_state_status is False:
        kuavo_arm_traj_pub.publish(ocs2_joint_state)
        control_hand_pub.publish(ocs2_hand_state)
        control_head_pub.publish(ocs2_head_state)

def init_publishers():
    global kuavo_arm_traj_pub, control_hand_pub, control_head_pub
    kuavo_arm_traj_pub = rospy.Publisher('/kuavo_arm_traj', JointState, queue_size=1, tcp_nodelay=True)
    control_hand_pub = rospy.Publisher('/control_robot_hand_position', robotHandPosition, queue_size=1, tcp_nodelay=True)
    control_head_pub = rospy.Publisher('/robot_head_motion_data', robotHeadMotionData, queue_size=1, tcp_nodelay=True)

async def init_ros_node():
    print("Initializing ROS node")
    rospy.init_node("arm_action_server", anonymous=True, disable_signals=True)
    robot_plan_arm_state_topic_name = robot_settings[g_robot_type]["arm_traj_state_topic_name"]
    rospy.Subscriber(robot_plan_arm_state_topic_name, planArmState, plan_arm_state_callback)
    rospy.Subscriber('/bezier/arm_traj', JointTrajectory, traj_callback, queue_size=1, tcp_nodelay=True)
    # rospy.Subscriber('/humanoid_mpc_observation', mpc_observation, mpc_obs_callback)
    rospy.Subscriber('/sensors_data_raw', sensorsData, sensors_data_callback, queue_size=1, tcp_nodelay=True)
    
    init_publishers()
    
    # Create a timer that calls timer_callback every 10ms (100Hz)
    rospy.Timer(rospy.Duration(0.01), timer_callback)

    print("ROS node initialized")

def create_bezier_request(action_data, start_frame_time, end_frame_time):
    req = planArmTrajectoryBezierCurveRequest()
    for key, value in action_data.items():
        msg = jointBezierTrajectory()
        for frame in value:
            point = bezierCurveCubicPoint()
            point.end_point, point.left_control_point, point.right_control_point = frame
            msg.bezier_curve_points.append(point)
        req.multi_joint_bezier_trajectory.append(msg)
    req.start_frame_time = start_frame_time
    req.end_frame_time = end_frame_time
    req.joint_names = ["l_arm_pitch", "l_arm_roll", "l_arm_yaw", "l_forearm_pitch", "l_hand_yaw", "l_hand_pitch", "l_hand_roll", "r_arm_pitch", "r_arm_roll", "r_arm_yaw", "r_forearm_pitch", "r_hand_yaw", "r_hand_pitch", "r_hand_roll"]
    return req

def plan_arm_trajectory_bezier_curve_client(req):
    service_name = robot_settings[g_robot_type]["plan_arm_trajectory_bezier_service_name"]
    rospy.wait_for_service(service_name)
    try:
        plan_service = rospy.ServiceProxy(service_name, planArmTrajectoryBezierCurve)
        res = plan_service(req)
        return res.success
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")
        return False

def stop_plan_arm_trajectory_client():
    service_name = robot_settings[g_robot_type]["stop_plan_arm_trajectory_service_name"]
    rospy.wait_for_service(service_name)
    try:
        if g_robot_type == "ocs2":
            stop_service = rospy.ServiceProxy(service_name, Trigger)
        else:
            stop_service = rospy.ServiceProxy(service_name, stopPlanArmTrajectory)

        stop_service()
        return
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")
        return False
@dataclass
class Payload:
    cmd: str
    data: dict


@dataclass
class Response:
    payload: Any
    target: Union[str, websockets.WebSocketServerProtocol]

def plan_arm_state_callback(msg: planArmState):
    global plan_arm_state_progress, plan_arm_state_status
    plan_arm_state_progress = msg.progress
    plan_arm_state_status = msg.is_finished



def update_preview_progress(response: Response, stop_event: threading.Event):
    payload = response.payload
    update_interval = 0.001 
    global plan_arm_state_progress, plan_arm_state_status
    last_progress = None
    last_status = None
    
    while not stop_event.is_set():
        current_progress = plan_arm_state_progress
        current_status = plan_arm_state_status

        if current_progress != last_progress or current_status != last_status:
            last_progress = current_progress
            last_status = current_status
            
            payload.data["progress"] = current_progress
            payload.data["status"] = 0 if current_status else 1
            response_queue.put(response)
            
            if current_status:
                return
        
        time.sleep(update_interval)



async def websocket_message_handler(
    websocket: websockets.WebSocketServerProtocol, data: dict
):
    cmd = data.get("cmd", "")
    cmd_handler = f"{cmd}_handler"
    if cmd_handler in globals() and callable(globals()[cmd_handler]):
        await globals()[cmd_handler](websocket, data)
    else:
        print(f"Unknown command: {cmd}")
        error_payload = Payload(
            cmd="error", data={"message": f"Unknown command: {cmd}"}
        )
        error_response = Response(payload=error_payload, target=websocket)
        response_queue.put(error_response)


async def broadacast_handler(websocket: websockets.WebSocketServerProtocol, data: dict):
    payload = Payload(
        cmd="broadcast",
        data={"code": 0, "message": "Broadcast message"},
    )
    response = Response(
        payload=payload,
        target="all",
    )

    response_queue.put(response)


async def preview_action_handler(
    websocket: websockets.WebSocketServerProtocol, data: dict
):
    global g_robot_type
    print(g_robot_type)
    # Cancel existing thread for this client if it exists
    if websocket in active_threads:
        active_threads[websocket].set()
        del active_threads[websocket]

    payload = Payload(
        cmd="preview_action", data={"code": 0, "status": 0, "progress": 0}
    )

    data = data["data"]
    print(data)
    action_filename = data["action_filename"]
    action_file_path = os.path.expanduser(f"{ACTION_FILE_FOLDER}/{action_filename}")
    if not os.path.exists(action_file_path):
        print(f"Action file not found: {action_file_path}")
        payload.data["code"] = 1
        response = Response(payload=payload, target=websocket)
        response_queue.put(response)
        return

    if calculate_file_md5(action_file_path) != data["action_file_MD5"]:
        print(f"Action file MD5 mismatch: {action_file_path}")
        payload.data["code"] = 2
        response = Response(payload=payload, target=websocket)
        response_queue.put(response)
        return


    start_frame_time, end_frame_time = get_start_end_frame_time(action_file_path)
    if g_robot_type == "ocs2":
        action_frames = frames_to_custom_action_data_ocs2(action_file_path, start_frame_time, current_arm_joint_state)
        end_frame_time += 1
        call_change_arm_ctrl_mode_service(2)
    else:
        action_frames = frames_to_custom_action_data(action_file_path)

    req = create_bezier_request(action_frames, start_frame_time, end_frame_time)
    plan_arm_trajectory_bezier_curve_client(req)
    time.sleep(0.5)
    # If valid, create initial response
    response = Response(
        payload=payload,
        target=websocket,
    )

    # Create a new stop event for this thread
    stop_event = threading.Event()
    active_threads[websocket] = stop_event

    # Start a thread to update the progress
    thread = threading.Thread(
        target=update_preview_progress, args=(response, stop_event)
    )
    thread.start()


async def stop_preview_action_handler(websocket: websockets.WebSocketServerProtocol, data: dict):
    if websocket in active_threads:
        active_threads[websocket].set()
        del active_threads[websocket]
    
    payload = Payload(
        cmd="stop_preview_action", data={"code":0}
    )
    response = Response(
        payload=payload,
        target=websocket,
    )
    stop_plan_arm_trajectory_client()
    response_queue.put(response)

# Add a function to clean up when a websocket connection is closed
def cleanup_websocket(websocket: websockets.WebSocketServerProtocol):
    if websocket in active_threads:
        active_threads[websocket].set()
        del active_threads[websocket]

# Add a new function to set the robot type
def set_robot_type(robot_type: str):
    global g_robot_type
    g_robot_type = robot_type
