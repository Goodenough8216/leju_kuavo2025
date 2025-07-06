#!/usr/bin/env python3

import rospy
import numpy as np
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker
from std_msgs.msg import Float64MultiArray

from kuavo_msgs.msg import footPose, footPoseTargetTrajectories, headBodyPose  # 导入自定义消息类型
from motion_capture_ik.srv import changeArmCtrlMode, twoArmHandPoseCmdSrv

from grabBox.ikCmd import get_ik_cmd_msg
from grabBox.ikCmd import *
from grabBox.footTraj import generate_steps, get_foot_pose_traj_msg
from grabBox.footTraj import *

from apriltag_ros.msg import AprilTagDetectionArray

class AprilTagProcessor:
    def __init__(self, target_id):
        self.data_list = []
        self.max_count = 25
        self.target_id = target_id
        self.subscriber = rospy.Subscriber("/tag_detections", AprilTagDetectionArray, self.process_message)
        self.data_collected = False

    def quaternion_to_euler(self, x, y, z, w):
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)
        return math.degrees(pitch)

    def process_message(self, msg):
        num_detections = len(msg.detections)
        rospy.loginfo("Find %d AprilTag" % num_detections)

        for i in range(num_detections):
            detection = msg.detections[i]
            id = detection.id[0]

            if id == self.target_id and len(self.data_list) < self.max_count:
                quaternion = detection.pose.pose.pose.orientation
                pos = detection.pose.pose.pose.position

                # 将四元数和位置值存储到data_list中
                tag_data = {
                    'quaternion': quaternion,
                    'position': pos
                }
                self.data_list.append(tag_data)

        if len(self.data_list) >= self.max_count:
            self.data_collected = True
            self.subscriber.unregister()

    def get_average_data(self):
        # 计算四元数和位置的均值
        N = len(self.data_list)

        avg_quaternion = np.array([
        sum(item['quaternion'].x for item in self.data_list) / N,
        sum(item['quaternion'].y for item in self.data_list) / N,
        sum(item['quaternion'].z for item in self.data_list) / N,
        sum(item['quaternion'].w for item in self.data_list) / N
        ])

        avg_position = np.array([
            sum(item['position'].x for item in self.data_list) / N,
            sum(item['position'].y for item in self.data_list) / N,
            sum(item['position'].z for item in self.data_list) / N
        ])

        rospy.loginfo("Average Quaternion: %s" % avg_quaternion)
        rospy.loginfo("Average Position: %s" % avg_position)

        return avg_position, avg_quaternion

def get_apriltag_average_data(target_id):
    processor = AprilTagProcessor(target_id)

    # 等待数据收集完成
    while not rospy.is_shutdown() and not processor.data_collected:
        rospy.sleep(0.1)

    return processor.get_average_data()

def construct_box_marker(box_point, box_size, quat=[0, 0, 0, 1], alpha=0.6, color=[0, 1, 0]):
    if len(box_point) != 3:
        print("Invalid pos, cannot construct marker")
        return None    
    if len(box_size) != 3:
        print("Invalid box_size, cannot construct marker")
        return None
    marker = Marker()
    marker.header.frame_id = "odom"
    marker.type = Marker.CUBE
    marker.action = Marker.ADD
    marker.scale.x = box_size[0]
    marker.scale.y = box_size[1]
    marker.scale.z = box_size[2]
    marker.color.a = alpha
    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]
    marker.header.stamp = rospy.Time.now()
    marker.pose.position.x = box_point[0]
    marker.pose.position.y = box_point[1]
    marker.pose.position.z = box_point[2]
    marker.pose.orientation.x = quat[0]
    marker.pose.orientation.y = quat[1]
    marker.pose.orientation.z = quat[2]
    marker.pose.orientation.w = quat[3]
    return marker 

def change_arm_ctrl_mode(mode: int):
    service_name = "/change_arm_ctrl_mode"
    try:
        rospy.wait_for_service(service_name)
        changeHandTrackingMode_srv = rospy.ServiceProxy(service_name, changeArmCtrlMode)
        changeHandTrackingMode_srv(mode)
    except rospy.ROSException:
        rospy.logerr(f"Service {service_name} not available")

def send_ik_cmd_srv(msg: twoArmHandPoseCmd):
    service_name = "/ik/two_arm_hand_pose_cmd_srv"
    try:
        # print("Waiting for service")
        rospy.wait_for_service(service_name)
        # print("Service available")
        two_arm_hand_pose_cmd_srv = rospy.ServiceProxy(service_name, twoArmHandPoseCmdSrv)
        req = two_arm_hand_pose_cmd_srv(msg)
        # print(f"req: {req}")
        return req
    except rospy.ROSException:
        rospy.logerr(f"Service {service_name} not available")
        return None


def publish_multiple_steps(pub, body_poses):
    # height = 0.0

    num_steps = 2*len(body_poses)
    time_traj = []
    foot_idx_traj = []
    foot_traj = []
    torso_traj = []
    dt = 0.8
    for i in range(num_steps):
        time_traj.append(dt * (i+1))
        body_pose = body_poses[i//2]
        torso_pos = np.asarray(body_pose[:3])
        torso_yaw = np.radians(body_pose[3])
        # body_pose[3] = torso_yaw    
        l_foot, r_foot = generate_steps(torso_pos, torso_yaw, 0.1)
        l_foot = [*l_foot[:3], torso_yaw]
        r_foot = [*r_foot[:3], torso_yaw]
        if(i%2 == 0):
            foot_idx_traj.append(0)
            foot_traj.append(l_foot)
        else:
            foot_idx_traj.append(1)
            foot_traj.append(r_foot)
        torso_traj.append([*body_pose[:3], torso_yaw])
    print("time_traj:", time_traj)
    print("foot_idx_traj:", foot_idx_traj)
    print("foot_traj:", foot_traj)
    print("torso_traj:", torso_traj)
    # publish_foot_pose_traj(time_traj, foot_idx_traj, foot_traj, torso_traj)
    msg = get_foot_pose_traj_msg(time_traj, foot_idx_traj, foot_traj, torso_traj)
    # print(f"msg: {msg}")
    pub.publish(msg)

def get_joint_states_msg(q_arm):
    # arm_agl_limited = self.limit_angle(q_arm[-14:])
    msg = JointState()
    msg.name = ["arm_joint_" + str(i) for i in range(1, 15)]
    msg.header.stamp = rospy.Time.now()
    msg.position = 180.0 / np.pi * np.array(q_arm)
    return msg

       
def get_head_body_pose_msg(q_torso):
    height, yaw, pitch, roll = q_torso
    msg = headBodyPose()
    msg.head_pitch = 0
    msg.head_yaw = 0
    msg.body_yaw = yaw
    msg.body_pitch = max(3*np.pi/180.0, min(pitch, 40*np.pi/180.0))

    msg.body_x = 0.0
    msg.body_y = 0.0
    msg.body_height = max(-0.4, min(height, 0.2))
    return msg

def arm_ready_to_move(eef_bias=0.15):
    quat = [0.0, -0.707, 0.0, 0.707]
    pos_left = [0.2+eef_bias, 0.35, 0.05]
    pos_right = [0.2+eef_bias, -0.35, 0.05]
    return (pos_left, quat), (pos_right, quat)

def arm_grab_back(eef_bias=0.15):
    quat = [0.0, -0.707, 0.0, 0.707]
    pos_left = [0.2+eef_bias, 0.20, 0.12]
    pos_right = [0.2+eef_bias, -0.20, 0.12]
    return (pos_left, quat), (pos_right, quat)

def arm_grab_down(eef_bias=0.15):
    quat = [0.0, -0.707, 0.0, 0.707]
    pos_left = [0.2+eef_bias, 0.20, -0.1]
    pos_right = [0.2+eef_bias, -0.20, -0.1]
    return (pos_left, quat), (pos_right, quat)

def arm_grab_down_loosen(eef_bias=0.15):
    quat = [0.0, -0.707, 0.0, 0.707]
    pos_left = [0.2+eef_bias, 0.35, -0.1]
    pos_right = [0.2+eef_bias, -0.35, -0.1]
    return (pos_left, quat), (pos_right, quat)

def arm_grab_down_up(eef_bias=0.15):
    quat = [0.0, -0.707, 0.0, 0.707]
    pos_left = [0.2+eef_bias, 0.35, 0.1]
    pos_right = [0.2+eef_bias, -0.35, 0.1]
    return (pos_left, quat), (pos_right, quat)

def control_arm(left_pose, right_pose):
    msg = get_ik_cmd_msg(left_pose, right_pose)
    req = send_ik_cmd_srv(msg)
    if(req is None):
        return
    if req.success:
        l_pos = np.array(req.hand_poses.left_pose.pos_xyz)
        r_pos = np.array(req.hand_poses.right_pose.pos_xyz)
        l_err_norm = np.linalg.norm(l_pos - np.array(left_pose[0]))
        r_err_norm = np.linalg.norm(r_pos - np.array(right_pose[0]))
        # print("left pos error: ", l_err_norm)
        # print("right pos error: ", r_err_norm)
        if(l_err_norm > 0.01 or r_err_norm > 0.01):
            print("[Error]: pos error too large!!!")
            return
        pub_arm_traj.publish(get_joint_states_msg(req.q_arm))
        if req.with_torso:
            # print("torso height:", req.q_torso[0])
            pub_head_body_pose.publish(get_head_body_pose_msg(req.q_torso))
    else:
        print("IK failed")

def generate_grab_box_trajectory(box_pos, box_size):
    quat = [0.0, -0.707, 0.0, 0.707]
    x, y, z = box_pos
    r = box_size/2.0
    pre_x = 0.1
    pre_y = 0.1
    bias_y = 0.01
    grab_up_height = 0.1
    # pre-grasp1
    pos_left =  [x-pre_x, y+(r+pre_y), z]
    pos_right = [x-pre_x, y-(r+pre_y), z]
    control_arm((pos_left, quat), (pos_right, quat))
    rospy.sleep(1)
    # pre-grasp2
    pos_left =  [x, y+(r+pre_y), z]
    pos_right = [x, y-(r+pre_y), z]
    control_arm((pos_left, quat), (pos_right, quat))
    rospy.sleep(1)
    # grasp
    pos_left =  [x, y+(r-bias_y), z]
    pos_right = [x, y-(r-bias_y), z]
    control_arm((pos_left, quat), (pos_right, quat))
    rospy.sleep(1)
    # grasp up
    pos_left =  [x, y+(r-bias_y), z+grab_up_height]
    pos_right = [x, y-(r-bias_y), z+grab_up_height]
    control_arm((pos_left, quat), (pos_right, quat))


base_pos = np.zeros(3)
base_euler_zyx = np.zeros(3)

def basePosCallback(msg):
    global base_pos
    base_pos = np.array(msg.data)
    # print("base_pos:", base_pos)

def baseEulerCallback(msg):
    global base_euler_zyx
    base_euler_zyx = np.array(msg.data)

def pos_robot_to_world(pos, com_height):
    global base_pos, base_euler_zyx
    p_ws = np.array(base_pos)
    p_ws[2] = 0.0
    p_rb = np.array(pos)
    p_sb = p_rb
    p_sb[2] = p_sb[2] + com_height
    print("p_sb:", p_sb)
    rotation = Rotation.from_euler('zyx', base_euler_zyx, degrees=False)
    R_ws = rotation.as_matrix()
    # print("R_ws:", R_ws)
    p_wb = p_ws + R_ws @ p_sb
    print("p_wb:", p_wb)
    return p_wb

if __name__ == '__main__':
    # 初始化 ROS 节点
    rospy.init_node('foot_pose_publisher', anonymous=True)
    # 创建发布者，话题为 /humanoid_mpc_foot_pose_target_trajectories
    sub_base_pos = rospy.Subscriber('/state_estimate/base/pos_xyz', Float64MultiArray, basePosCallback)
    sub_base_euler = rospy.Subscriber('/state_estimate/base/angular_zyx', Float64MultiArray, baseEulerCallback)
    pub = rospy.Publisher('/humanoid_mpc_foot_pose_target_trajectories', footPoseTargetTrajectories, queue_size=10)
    pub_ik_cmd = rospy.Publisher('/ik/two_arm_hand_pose_cmd', twoArmHandPoseCmd, queue_size=10)
    pub_arm_traj = rospy.Publisher("/kuavo_arm_traj", JointState, queue_size=10)
    pub_head_body_pose = rospy.Publisher('/kuavo_head_body_orientation', headBodyPose, queue_size=10)
    pub_box_marker = rospy.Publisher("visualization_box_marker", Marker, queue_size=10)

    com_height = rospy.get_param("com_height", 0.15)
    print("com_height:", com_height)
    # 等待一定时间以确保订阅者已经准备好
    rospy.sleep(1)

    change_arm_ctrl_mode(2)

    control_arm(*arm_ready_to_move())
    rospy.sleep(1)
    # move 
    body_poses = [
        [0.2, 0.0, 0, 0],
        # [0.4, 0.0, 0, 0],
    ]
    publish_multiple_steps(pub, body_poses)

    rospy.sleep(2)
    # in local frame
    box_pos = [0.45, 0.0, 0.1]
    box_size = 0.3
    box_pos_world = pos_robot_to_world(box_pos, com_height)
    pub_box_marker.publish(construct_box_marker(box_pos_world, [box_size for i in range(3)]))
    generate_grab_box_trajectory(box_pos, box_size)

    # move 
    body_poses = [
        [-0.10, 0.0, 0, 0],
        [-0.20, 0.0, 0, 0],
    ]
    publish_multiple_steps(pub, body_poses)
    rospy.sleep(4)

    control_arm(*arm_grab_back())
    rospy.sleep(1)
    # move 
    body_poses = [
        [0.0, -0.05, 0, -30],
        [0.0, -0.10, 0, -60],
        [0.0, -0.15, 0, -90],
    ]
    publish_multiple_steps(pub, body_poses)

    rospy.sleep(6)
    control_arm(*arm_grab_down())
    rospy.sleep(1)
    control_arm(*arm_grab_down_loosen())
    rospy.sleep(1)
    control_arm(*arm_grab_down_up())
    rospy.sleep(1)
    