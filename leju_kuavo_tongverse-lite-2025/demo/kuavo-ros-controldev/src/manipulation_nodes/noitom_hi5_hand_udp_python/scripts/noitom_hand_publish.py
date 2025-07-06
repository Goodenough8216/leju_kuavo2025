#!/usr/bin/env python3
import socket
import sys
import struct
import time
import datetime
import math
import rospy
from geometry_msgs.msg import Quaternion
from noitom_hi5_hand_udp_python.msg import QuaternionArray
from noitom_hi5_hand_udp_python.msg import handRotationEular

server_ip = "10.10.20.41"

local_port_left = 10050
local_port_right = 10060

pose_timestamp_name = "HandPosT"

bone_names = [
    "ForeArm",
    "Hand",
    "HandThumb1",
    "HandThumb2",
    "HandThumb3",
    "InHandIndex",
    "HandIndex1",
    "HandIndex2",
    "HandIndex3",
    "InHandMiddle",
    "HandMiddle1",
    "HandMiddle2",
    "HandMiddle3",
    "InHandRing",
    "HandRing1",
    "HandRing2",
    "HandRing3",
    "InHandPinky",
    "HandPinky1",
    "HandPinky2",
    "HandPinky3",
    "HandRot",
    "HandPos",
    pose_timestamp_name,
    "HeadPos",
    "HeadRot"
]

# Function to get the index of a bone name
def get_bone_index(bone_name):
    try:
        return bone_names.index(bone_name)
    except ValueError:
        return -1  # Returns -1 if the bone name is not found

class Point3D:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return f"Point3D(x={self.x:+09.4f}, y={self.y:+09.4f}, z={self.z:+09.4f})"

    def __eq__(self, other):
        if not isinstance(other, Point3D):
            return NotImplemented
        return self.x == other.x and self.y == other.y and self.z == other.z

    def diff(self, other):
        if not isinstance(other, Point3D):
            raise ValueError("The 'other' parameter must be an instance of Point3D")
        return Point3D(self.x - other.x, self.y - other.y, self.z - other.z)

class Quaternion:
    def __init__(self, x, y, z, w):
        self.x = x
        self.y = y
        self.z = z
        self.w = w

    def __repr__(self):
        return f"Quat(x={self.x:+2.4f}, y={self.y:+2.4f}, z={self.z:+2.4f}, w={self.w:+2.4f})"

    def __eq__(self, other):
        if not isinstance(other, Quaternion):
            return NotImplemented
        tolerance = 1e-5
        return (abs(self.x - other.x) < tolerance and
                abs(self.y - other.y) < tolerance and
                abs(self.z - other.z) < tolerance and
                abs(self.w - other.w) < tolerance)

    def diff(self, other):
        if not isinstance(other, Quaternion):
            raise ValueError("The 'other' parameter must be an instance of Quaternion")
        return Quaternion(self.x - other.x, self.y - other.y, self.z - other.z, self.w - other.w)

def euler_to_quaternion(euler_angles):
    # Convert Euler angles (in degrees) to radians
    roll = math.radians(euler_angles.x)
    pitch = math.radians(euler_angles.y)
    yaw = math.radians(euler_angles.z)

    # Abbreviations for the various angular functions
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    # Calculate the Quaternion components
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return Quaternion(round(x,3), round(y,3), round(z,3), round(w,3))

def vector3_to_milliseconds(x, y, z):
    # Reconstruct the bytes from the Vector3 components
    # x, y, z are assumed to be float but should contain integer values for this purpose
    byte0 = int(x) & 0xFF
    byte1 = int(y) >> 8 & 0xFF
    byte2 = int(y) & 0xFF
    byte3 = int(z) & 0xFF

    # Reverse the order of the bytes
    reversed_bytes = [byte3, byte2, byte1, byte0]

    # Pack the reversed bytes back into a single integer
    # '<' denotes little-endian byte order, 'I' denotes an unsigned int
    milliseconds_int = struct.unpack('<I', bytes(reversed_bytes))[0]
    return milliseconds_int


def send_data(sock, server_ip, server_port):
    message = "Hi!"
    try:
        sock.sendto(message.encode(), (server_ip, server_port))
    except socket.timeout:
        print(f"{datetime.datetime.now().strftime('%H:%M:%S.%f')} Timed out while trying to send data to server.")
        return []

    # Receive response
    try:
        data, server = sock.recvfrom(4096)
    except socket.timeout:
        print(f"socket timeout: {datetime.datetime.now().strftime('%H:%M:%S.%f')} Timed out waiting for data from server {server_ip}:{server_port}")
        return [], False

    float_array = []
    points = []

    if len(data) < len(bone_names) * 4:
        return [], False

    # Unpack the received data into float values
    for i in range(0, len(data), 4):
        float_value = struct.unpack('<f', data[i:i+4])[0]
        float_array.append(float_value)

    left_hand_mark = Quaternion(0xa5, 0x5a, 0xa5, 1)
    right_hand_mark = Quaternion(0x5a, 0xa5, 0x5a, 1)

    result = 0

    # Convert float_array to a structured array of Quaternion objects
    for i in range(0, len(float_array), 4):
        if i % 4 == 0 and bone_names[int(i/4)] == pose_timestamp_name:
            point = Quaternion(float_array[i], float_array[i + 1], float_array[i + 2], float_array[i + 3])
            points.append(point)
        else:
            if i + 3 < len(float_array):
                float_array[i] = float_array[i] if float_array[i] < 180 else float_array[i] - 360
                float_array[i + 1] = float_array[i + 1] if float_array[i + 1] < 180 else float_array[i + 1] - 360
                float_array[i + 2] = float_array[i + 2] if float_array[i + 2] < 180 else float_array[i + 2] - 360
                float_array[i + 3] = float_array[i + 3] if float_array[i + 3] < 180 else float_array[i + 3] - 360
                point = Quaternion(float_array[i], float_array[i + 1], float_array[i + 2], float_array[i + 3])
                points.append(point)

    # if points:
    #     print(f"Debug: First point - {points[0]}")
    #     print(f"Left Hand Mark: {left_hand_mark}")
    #     print(f"Right Hand Mark: {right_hand_mark}")


    if left_hand_mark == points[0]:
        result = 1
    elif right_hand_mark == points[0]:
        result = 2

    return points, result

if __name__ == "__main__":
    print("Server Starting...")
    server_left_hand_port = 10019
    server_right_hand_port = 10029

    # lefhand_pub = rospy.Publisher('kuavo_lefthand_quaternion_array', QuaternionArray, queue_size=10)
    lefhand_eular_pub = rospy.Publisher('kuavo_lefthand_eular_array', handRotationEular, queue_size=10)
    righthand_eular_pub = rospy.Publisher('kuavo_righthand_eular_array', handRotationEular, queue_size=10)

    left_hand_diff_head = rospy.Publisher('kuavo_lefthand_diff_head', handRotationEular, queue_size=10)
    right_hand_diff_head = rospy.Publisher('kuavo_righthand_diff_head', handRotationEular, queue_size=10)

    rospy.init_node('kuavo_hand_track_publisher', anonymous=True)

    if len(sys.argv) > 1:
        server_ip = sys.argv[1]

    print(f"server_ip: {server_ip}")

    timeout_seconds = 2
    # Create a UDP socket
    left_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    left_sock.bind(("", local_port_left))  # Empty string means "all interfaces"
    left_sock.settimeout(timeout_seconds)

    right_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # Bind it to a different port
    right_sock.bind(("", local_port_right))  # Empty string means "all interfaces"
    right_sock.settimeout(timeout_seconds)
    # Set a timeout for the socket operations


    # Replace time.sleep(0.05) with ROS spin mechanism
    # time.sleep(0.05)  # Old code for reference, to be removed

    # Instead of using time.sleep, we'll use rospy.Rate to control the loop rate
    # and rospy.is_shutdown() for gracefully handling Ctrl-C
    rate = rospy.Rate(100)  # 20Hz corresponds to 0.05s sleep time
    try:
        while not rospy.is_shutdown():  # Use rospy.is_shutdown() to check for Ctrl-C
            # Send a test message
            sys.stdout.write("\033[2J\033[H")  # ANSI escape code to clear the screen and move cursor to top-left
            points, left_hand_is_valid = send_data(left_sock, server_ip, server_left_hand_port)

            right_points, right_hand_is_valid = send_data(right_sock, server_ip, server_right_hand_port)

            print(f"{datetime.datetime.now().strftime('%H:%M:%S.%f')}")

            print(f"Left is valid: {left_hand_is_valid} len is:{len(points)}  Right is valid: {right_hand_is_valid} len is:{len(right_points)}")

            if left_hand_is_valid == 2 and right_hand_is_valid == 1:
                points, right_points = right_points, points
            elif left_hand_is_valid == 1 and right_hand_is_valid == 2:
                pass
            else:
                points = []
                right_points = []
            # Print the structured array of points

            if len(points) < len(bone_names) or len(right_points) < len(bone_names) or (-1 == left_hand_is_valid) or (-1 == right_hand_is_valid):
                continue

            eular_array_msg = handRotationEular()
            eular_array_msg.eulerAngles = points

            right_eular_array_msg = handRotationEular()
            right_eular_array_msg.eulerAngles = right_points

            lefhand_eular_pub.publish(eular_array_msg)
            righthand_eular_pub.publish(right_eular_array_msg)


            left_hand_head_diff_points = []

            # Assuming get_bone_index and points are defined elsewhere in your code.
            # Added a print statement to dump the difference points for debugging purposes.
            diff_point = points[get_bone_index("HandPos")].diff(points[get_bone_index("HeadPos")])
            left_hand_head_diff_points.append(diff_point)
            left_hand_head_diff_points.append(points[get_bone_index("HandRot")].diff(points[get_bone_index("HeadRot")]))

            left_hand_head_diff_msg = handRotationEular()
            left_hand_head_diff_msg.eulerAngles = left_hand_head_diff_points
            left_hand_diff_head.publish(left_hand_head_diff_msg)


            right_hand_head_diff_points  = []

            right_hand_head_diff_points.append(right_points[get_bone_index("HandPos")].diff(right_points[get_bone_index("HeadPos")]))
            right_hand_head_diff_points.append(right_points[get_bone_index("HandRot")].diff(right_points[get_bone_index("HeadRot")]))

            right_hand_head_diff_msg = handRotationEular()
            right_hand_head_diff_msg.eulerAngles = right_hand_head_diff_points
            right_hand_diff_head.publish(right_hand_head_diff_msg)

            # Check if the length of bone_names matches the length of right_points
            if (len(bone_names) != len(right_points)) or (len(bone_names) != len(points)):
                print("Error: The length of bone_names does not match the length of right_points.")
            else:
                for index, bone_name in enumerate(bone_names):
                    if bone_name == pose_timestamp_name:
                        print(f"T: {vector3_to_milliseconds(points[index].x,points[index].y,points[index].z)} L-R T:{vector3_to_milliseconds(right_points[index].x,right_points[index].y,right_points[index].z)}")
                    else:
                        print(f"{bone_name:12} {points[index]} L-R {right_points[index]}")
                    # if index < len(bone_names) - 1:
                    #     print(f"{bone_name:12} {euler_to_quaternion(points[index])} L-R {euler_to_quaternion(right_points[index])}")
                    # else:
                    #     print(f"{bone_name:12} Position {points[index]} L-R {right_points[index]}")

            rate.sleep()  # Use rospy.Rate.sleep() instead of time.sleep


    except Exception as e:
        print(f"Except: {e}")
    finally:
        left_sock.close()
        right_sock.close()

