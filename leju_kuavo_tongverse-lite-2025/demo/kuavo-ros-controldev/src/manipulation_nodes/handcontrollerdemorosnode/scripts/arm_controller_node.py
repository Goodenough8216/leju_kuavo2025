#!/usr/bin/env python3
import sys
import signal
import argparse
import math
import os
import queue
import rospy
import tf
from geometry_msgs.msg import Quaternion
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker
from config import read_json_file
from handcontrollerdemorosnode.msg import armPoseWithTimeStamp
from handcontrollerdemorosnode.msg import handRotationEular
from handcontrollerdemorosnode.msg import robotArmPose
from handcontrollerdemorosnode.srv import changeHandTrackingMode
from handcontrollerdemorosnode.srv import srvArmIK
from utils import get_package_path, limit_values, rpy_to_quaternion


import tf.transformations

robot_left_arm_init_pose = [0, 0, 0, 1, 0.15, 0.3, -0.121]
robot_right_arm_init_pose = [0, 0, 0, 1, 0.15, -0.3, -0.121]

robot_arm_joint_state_min = [
    -90,
    0,
    -90,
    -90,
    -90,
    -10,
    -15,
    -90,
    -90,
    -90,
    -90,
    -90,
    -10,
    -15,
]

robot_arm_joint_state_max = [
    30,
    90,
    90,
    0,
    90,
    10,
    15,
    30,
    0,
    90,
    0,
    90,
    10,
    15,
]
joint_state_queue = queue.Queue()
reverse = 0


def signal_handler(sig, frame):
    rospy.signal_shutdown("Keyboard interrupt")
    sys.exit(0)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args():
    parser = argparse.ArgumentParser(description="Arm Controller Node")
    parser.add_argument(
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Enable debug mode",
    )
    parser.add_argument(
        "--side",
        type=str,
        default="all",
        help="Side of the hand controller",
    )
    parser.add_argument(
        "--pose",
        type=str,
        default="all",
        help="Pose of the hand controller",
    )
    parser.add_argument(
        "--hand_tracking",
        type=bool,
        default=False,
        help="Enable hand tracking",
    )
    parser.add_argument(
        "--match",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Match mode",
    )
    args, unknown = parser.parse_known_args()
    return args


def ask_reverse_input():
    while True:
        global reverse
        try:
            reverse_noitom_hi5_hand_ori = int(
                input(
                    "请问校正完手套设备后手掌模型是否朝向外（与我们的手的方向一致）一致请输入：1， 相反请输入：2 : "
                )
            )
            if reverse_noitom_hi5_hand_ori == 1:
                reverse = 0
                break
            elif reverse_noitom_hi5_hand_ori == 2:
                reverse = 180
                break
            else:
                print("Invalid input, please input 1 or 2.")
                continue
        except ValueError:
            print("Invalid input, please input 1 or 2.")
            continue

    return reverse


class ArmControllerNode:
    def __init__(self, config_file_path, reverse, debug=False):

        self.marker_pub = rospy.Publisher("visualization_marker", Marker, queue_size=10)

        self.kuavo_arm_traj_pub = rospy.Publisher(
            "/kuavo_arm_traj", JointState, queue_size=10
        )

        self.arm_pose_pub = rospy.Publisher(
            "/kuavo_hand_pose", armPoseWithTimeStamp, queue_size=10
        )
        self.left_hand_sub = rospy.Subscriber(
            "kuavo_lefthand_eular_array",
            handRotationEular,
            self.left_arm_callback,
        )
        self.right_hand_sub = rospy.Subscriber(
            "kuavo_righthand_eular_array",
            handRotationEular,
            self.right_arm_callback,
        )
        self.arm_pose_sub = rospy.Subscriber(
            "/robot_arm_pose", robotArmPose, self.arm_pose_callback
        )
        self.pub_joint_state_timer = rospy.Timer(
            rospy.Duration(0.002), self.pub_joint_state
        )
        self._current_arm_pose = []
        self._left_arm_target_pose = []
        self._right_arm_target_pose = []
        self.debug = debug
        self.br = tf.TransformBroadcaster()
        self.robot_arm_config = read_json_file(config_file_path)
        self.reverse = reverse
        self.kuavo_pkg_path = get_package_path("dynamic_biped")
        self.current_pkg_path = get_package_path("handcontrollerdemorosnode")

    @property
    def current_arm_pose(self):
        return self._current_arm_pose

    @property
    def target_arm_pose(self):
        return self._left_arm_target_pose + self._right_arm_target_pose

    @property
    def left_arm_target_pose(self):
        return self._left_arm_target_pose

    @property
    def right_arm_target_pose(self):
        return self._right_arm_target_pose

    @property
    def left_arm_current_pose(self):
        return self._left_arm_current_pose

    @property
    def right_arm_current_pose(self):
        return self._right_arm_current_pose

    def limit_arm_pose_position(self, arm_pose_pos, side):
        limit = self.robot_arm_config[side]["limit"]
        for i, pos in enumerate(arm_pose_pos):
            arm_pose_pos[i] = min(max(pos, limit[i][0]), limit[i][1])
        return arm_pose_pos

    def add_offset_to_arm_pose_position(self, arm_pose_pos, side):
        offset = self.robot_arm_config[side]["offset"]
        arm_pose_pos = [pos + offset[i] for i, pos in enumerate(arm_pose_pos)]
        return arm_pose_pos

    def left_arm_callback(self, msg):
        eulerAngles = msg.eulerAngles
        self._left_arm_target_pose.clear()
        # IMPORTANT: Do not change the order of the elements in the list
        if eulerAngles[21] and eulerAngles[22]:
            orientation = [
                -math.radians(eulerAngles[21].z - 90),
                math.radians(eulerAngles[21].x),
                -math.radians(eulerAngles[21].y + self.reverse),
            ]
            orientation = rpy_to_quaternion(*orientation)
            position = [
                eulerAngles[22].z - eulerAngles[24].z,
                -eulerAngles[22].x + eulerAngles[24].x,
                eulerAngles[22].y - eulerAngles[24].y,
            ]
            position = self.add_offset_to_arm_pose_position(position, "left")
            trans = position
            rot = orientation

            self.br.sendTransform(
                trans,
                rot,
                rospy.Time.now(),
                "l_a_t_pose",  # l_a_t_pose: left arm target pose
                "torso",
            )

            position = self.limit_arm_pose_position(position, "left")
        else:
            rospy.logerr("No left arm pose data")
            return

        self._left_arm_target_pose = orientation + position

        if self.debug:
            trans = self._left_arm_target_pose[4:]
            rot = orientation

            self.br.sendTransform(
                trans,
                rot,
                rospy.Time.now(),
                "l_a_t_with_limit_pose",
                "torso",
            )

    def right_arm_callback(self, msg):
        eulerAngles = msg.eulerAngles
        self._right_arm_target_pose.clear()
        # IMPORTANT: Do not change the order of the elements in the list
        if eulerAngles[21] and eulerAngles[22]:
            orientation = [
                -math.radians(eulerAngles[21].z + 90),
                math.radians(eulerAngles[21].x),
                -math.radians(eulerAngles[21].y + self.reverse),
            ]
            orientation = rpy_to_quaternion(*orientation)
            position = [
                eulerAngles[22].z - eulerAngles[24].z,
                -eulerAngles[22].x + eulerAngles[24].x,
                eulerAngles[22].y - eulerAngles[24].y,
            ]
            position = self.add_offset_to_arm_pose_position(position, "right")
            trans = position
            rot = orientation
            self.br.sendTransform(
                trans,
                rot,
                rospy.Time.now(),
                "r_a_t_pose",  # r_a_t_pose: right arm target pose
                "torso",
            )

            position = self.limit_arm_pose_position(position, "right")
        else:
            rospy.logerr("No right arm pose data")
            return

        self._right_arm_target_pose = orientation + position

        if self.debug:
            trans = self._right_arm_target_pose[4:]
            rot = orientation

            self.br.sendTransform(
                trans,
                rot,
                rospy.Time.now(),
                "r_a_t_with_limit_pose",
                "torso",
            )

    def arm_pose_callback(self, msg):
        self._current_arm_pose.clear()
        self._current_arm_pose += msg.left_arm_pose
        self._current_arm_pose += msg.right_arm_pose
        self._left_arm_current_pose = msg.left_arm_pose
        self._right_arm_current_pose = msg.right_arm_pose

        if self.debug:
            right_trans = self._right_arm_current_pose[4:]
            right_rot = self._right_arm_current_pose[:4]
            left_trans = self._left_arm_current_pose[4:]
            left_rot = self._left_arm_current_pose[:4]
            self.br.sendTransform(
                right_trans,
                right_rot,
                rospy.Time.now(),
                "r_a_current_pose",
                "torso",
            )

            self.br.sendTransform(
                left_trans,
                left_rot,
                rospy.Time.now(),
                "l_a_current_pose",
                "torso",
            )

    def enable_hand_tracking(self):
        service_name = "/change_hand_tracking_mode"
        try:
            rospy.wait_for_service(service_name)
            changeHandTrackingMode_srv = rospy.ServiceProxy(
                service_name, changeHandTrackingMode
            )
            changeHandTrackingMode_srv(True)
        except rospy.ROSException:
            rospy.logerr(f"Service {service_name} not available")

    def armIK(self, arm_pose):
        service_name = "/arm_IK"
        if len(arm_pose) != 14:
            rospy.logerr("Invalid arm pose, cannot call service")
            return
        left_arm_pose = arm_pose[:7]
        right_arm_pose = arm_pose[7:]
        try:
            rospy.wait_for_service(service_name, timeout=0.5)
            armIK_srv = rospy.ServiceProxy(service_name, srvArmIK)
            reponse = armIK_srv(left_arm_pose, right_arm_pose)
            return reponse.joint_state
        except rospy.ROSException:
            rospy.logerr(f"Service {service_name} not available")
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")

    def visualize_marker(self):
        left_marker = self._construct_marker(self._left_arm_target_pose, 0.0, 0.0, 1, 1)
        right_marker = self._construct_marker(
            self._right_arm_target_pose, 1.0, 0.0, 0, 2
        )
        if left_marker:
            self.marker_pub.publish(left_marker)
        if right_marker:
            self.marker_pub.publish(right_marker)

    def _construct_marker(self, arm_pose, r, g, b, id):
        if len(arm_pose) != 7:
            rospy.logerr("Invalid arm pose, cannot construct marker")
            return None
        marker = Marker()
        marker.id = id
        marker.header.frame_id = (
            "torso"  # set frame_id according to the actual situation
        )
        marker.type = Marker.MESH_RESOURCE
        marker.action = Marker.ADD
        if id == 1:
            marker.mesh_resource = (
                "file://"
                + self.current_pkg_path
                + "/models/biped_gen4.0/meshes/l_hand_roll.obj"
            )
        else:
            marker.mesh_resource = (
                "file://"
                + self.current_pkg_path
                + "/models/biped_gen4.0/meshes/r_hand_roll.obj"
            )
        marker.scale.x = 1
        marker.scale.y = 1
        marker.scale.z = 1
        marker.color.a = 1.0
        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        marker.header.stamp = rospy.Time.now()
        marker.pose.position.x = arm_pose[4]
        marker.pose.position.y = arm_pose[5]
        marker.pose.position.z = arm_pose[6]
        marker.pose.orientation = Quaternion(*arm_pose[:4])
        return marker

    def visualize_tf(self, trans, rot, frame_id, child_frame_id):
        self.br.sendTransform
        (
            trans,
            rot,
            rospy.Time.now(),
            child_frame_id,
            frame_id,
        )

    def pub_joint_state(self, event):
        global joint_state_queue
        if not joint_state_queue.empty():
            joint_state = joint_state_queue.get()
            position = joint_state.position
            position = limit_values(
                position, robot_arm_joint_state_min, robot_arm_joint_state_max
            )
            joint_state.position = position
            rospy.loginfo(f"Publishing joint state: {joint_state}")
            self.kuavo_arm_traj_pub.publish(joint_state)

    def pub_arm_pose(self, poses):
        arm_pose_msg = armPoseWithTimeStamp()
        arm_pose_msg.left_hand_pose = poses[:7]
        arm_pose_msg.right_hand_pose = poses[-7:]
        self.arm_pose_pub.publish(arm_pose_msg)


def handle_arm_pose_in_test_mode(side, pose, target_arm_pose):
    if side == "left":
        target_arm_pose[7:] = robot_right_arm_init_pose
    elif side == "right":
        target_arm_pose[:7] = robot_left_arm_init_pose

    if pose == "position":
        target_arm_pose[:4] = [0, 0, 0, 1]
        target_arm_pose[7:11] = [0, 0, 0, 1]
    elif pose == "orientation":
        target_arm_pose[4:7] = robot_left_arm_init_pose[4:]
        target_arm_pose[11:] = robot_right_arm_init_pose[4:]
    return target_arm_pose


def run_node(*args, **kwargs):
    try:

        config_file_path = kwargs.get("config_file_path")
        reverse = kwargs.get("reverse")
        debug = kwargs.get("debug")
        side = kwargs.get("side")
        pose = kwargs.get("pose")
        hand_tracking = kwargs.get("hand_tracking")

        rospy.init_node("arm_controller_node")
        rospy.loginfo("kwarg: %s", kwargs)
        arm_controller_node = ArmControllerNode(config_file_path, reverse, debug)
        if hand_tracking:
            arm_controller_node.enable_hand_tracking()
        rate = rospy.Rate(50)
        i = 0
        while not rospy.is_shutdown():
            target_arm_pose = arm_controller_node.target_arm_pose
            if debug:
                arm_controller_node.visualize_marker()
            if side and pose:
                target_arm_pose = handle_arm_pose_in_test_mode(
                    side, pose, target_arm_pose
                )
            if i % 50 == 0:
                i = 0
                rospy.loginfo(f"Target arm pose: {target_arm_pose}")
            arm_controller_node.pub_arm_pose(target_arm_pose)
            # joint_state = arm_controller_node.armIK(target_arm_pose)
            # if joint_state:
            #     joint_state_queue.put(joint_state)
            i += 1
            rate.sleep()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    args = parse_args()
    if args.match is not None:
        if args.match:
            args.reverse = 0
        else:
            args.reverse = 180
    else:
        args.reverse = ask_reverse_input()

    current_path = os.path.dirname(os.path.abspath(__file__))
    config_file_path = os.path.dirname(current_path) + "/config/robot_arm_config.json"
    run_node(config_file_path=config_file_path, **vars(args))
