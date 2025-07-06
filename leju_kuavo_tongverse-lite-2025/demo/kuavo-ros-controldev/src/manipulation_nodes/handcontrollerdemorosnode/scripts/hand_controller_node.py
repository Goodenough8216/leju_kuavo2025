#!/usr/bin/env python3
import os
import sys
import rospy
import signal
import argparse
import numpy as np
from config import read_json_file
from handcontrollerdemorosnode.msg import handRotationEular, robotHandPosition
from handcontrollerdemorosnode.srv import controlEndHand


bone_names = {
    "ForeArm": 0,
    "Hand": 1,
    "HandThumb1": 2,
    "HandThumb2": 3,
    "HandThumb3": 4,
    "InHandIndex": 5,
    "HandIndex1": 6,
    "HandIndex2": 7,
    "HandIndex3": 8,
    "InHandMiddle": 9,
    "HandMiddle1": 10,
    "HandMiddle2": 11,
    "HandMiddle3": 12,
    "InHandRing": 13,
    "HandRing1": 14,
    "HandRing2": 15,
    "HandRing3": 16,
    "InHandPinky": 17,
    "HandPinky1": 18,
    "HandPinky2": 19,
    "HandPinky3": 20,
    "HandRot": 21,
    "HandPos": 22,
    "HandPosT": 23,
    "HeadPos": 24,
    "HeadRot": 25,
}

needed_bone = [
    {
        "name": "HandThumb2",
        "index": bone_names["HandThumb2"],
    },
    {
        "name": "HandThumb1",
        "index": bone_names["HandThumb1"],
    },
    {
        "name": "HandIndex3",
        "index": bone_names["HandIndex3"],
    },
    {
        "name": "HandMiddle3",
        "index": bone_names["HandMiddle3"],
    },
    {
        "name": "HandRing3",
        "index": bone_names["HandRing3"],
    },
    {
        "name": "HandPinky3",
        "index": bone_names["HandPinky3"],
    },
]

KUAVO_HAND_POSITION_RANGE = 100


def map_to_range(x, y_min, y_max):
    return y_min + (y_max - y_min) * x


def signal_handler(sig, frame):
    rospy.signal_shutdown("Keyboard interrupt")
    sys.exit(0)


def parse_args():
    parser = argparse.ArgumentParser(description="Hand Controller Node")
    parser.add_argument(
        "--debug",
        type=bool,
        default=False,
        help="Enable debug mode",
    )
    args, unknown = parser.parse_known_args()
    return args


def clamp_value(val, min_val, max_val):
    if val < min_val:
        return min_val - min_val
    if val > max_val:
        return max_val - min_val
    return val - min_val


def normalize_value(val, min_val, max_val):
    val = np.array(val)
    min_val = np.array(min_val)
    max_val = np.array(max_val)

    adjusted_val = np.where(
        val > max_val, max_val, np.where(val > min_val, val, min_val)
    )

    normalized_val = (adjusted_val - min_val) / (max_val - min_val)

    return normalized_val


class HandControllerNode:
    def __init__(self, config_file_path, debug=False):

        self.left_hand_sub = rospy.Subscriber(
            "kuavo_lefthand_eular_array",
            handRotationEular,
            self.left_hand_callback,
        )
        self.right_hand_sub = rospy.Subscriber(
            "kuavo_righthand_eular_array",
            handRotationEular,
            self.right_hand_callback,
        )
        self.control_robot_hand_position_pub = rospy.Publisher(
            "control_robot_hand_position", robotHandPosition, queue_size=10
        )
        self.pub_robot_end_hand_timer = rospy.Timer(
            rospy.Duration(0.01), self.pub_robot_end_hand
        )
        self.left_hand_position = [0, 0, 0, 0, 0, 0]
        self.right_hand_position = [0, 0, 0, 0, 0, 0]

        self.debug = debug
        self.left_right_z_axis = read_json_file(config_file_path)

    def left_hand_callback(self, msg):
        eulerAngles = msg.eulerAngles

        normalize_z_list = self.get_normalized_z_list(eulerAngles, "left")
        self.left_hand_position = [
            int(normalize_z * KUAVO_HAND_POSITION_RANGE)
            for normalize_z in normalize_z_list
        ]
        self.left_hand_position[1] = int(
            map_to_range(normalize_z_list[1], 80, KUAVO_HAND_POSITION_RANGE)
        )
        if self.debug:
            rospy.loginfo(f"Left hand position: {self.left_hand_position}")

        # self.control_end_hand()

    def right_hand_callback(self, msg):
        eulerAngles = msg.eulerAngles

        normalize_z_list = self.get_normalized_z_list(eulerAngles, "right")
        self.right_hand_position = [
            int(normalize_z * KUAVO_HAND_POSITION_RANGE)
            for normalize_z in normalize_z_list
        ]

        self.right_hand_position[1] = int(
            map_to_range(normalize_z_list[1], 80, KUAVO_HAND_POSITION_RANGE)
        )

        if self.debug:
            rospy.loginfo(f"Right hand position: {self.right_hand_position}")

        # self.control_end_hand()

    def get_normalized_z_list(self, eulerAngles, side):
        z_list = []
        for bone in needed_bone:
            z = abs(eulerAngles[bone["index"]].z)
            min_val, max_val = self.left_right_z_axis[side][bone["name"]]
            normalize_z = normalize_value(z, min_val, max_val)
            z_list.append(normalize_z)
        return z_list

    def control_end_hand(self):
        service_name = "control_end_hand"
        try:
            rospy.wait_for_service(service_name, timeout=0.5)
            control_end_hand = rospy.ServiceProxy(service_name, controlEndHand)
            control_end_hand(self.left_hand_position, self.right_hand_position)
        except rospy.ROSException:
            rospy.logerr(f"Service {service_name} not available")
        except rospy.ServiceException as e:
            rospy.logerr("Failed to call service control_end_hand")
            rospy.error(e)

    def pub_robot_end_hand(self, event):
        robot_hand_position = robotHandPosition()
        robot_hand_position.header.stamp = rospy.Time.now()
        robot_hand_position.left_hand_position = self.left_hand_position
        robot_hand_position.right_hand_position = self.right_hand_position
        self.control_robot_hand_position_pub.publish(robot_hand_position)


def run_node(*args, **kwargs):
    try:
        config_file_path = kwargs.get("config_file_path")
        debug = kwargs.get("debug")
        rospy.init_node("hand_controller_node")
        hand_controller_node = HandControllerNode(config_file_path, debug)
        while not rospy.is_shutdown():
            rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    args = parse_args()
    current_file_path = os.path.dirname(os.path.abspath(__file__))
    config_file_path = (
        os.path.dirname(current_file_path) + "/config/left_right_z_axis.json"
    )
    run_node(config_file_path=config_file_path, debug=args.debug)
