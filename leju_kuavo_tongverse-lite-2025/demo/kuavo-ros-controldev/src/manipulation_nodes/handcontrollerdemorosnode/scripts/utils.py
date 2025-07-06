import rospy
import tf
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from std_msgs.msg import Header
import numpy as np
from geometry_msgs.msg import Quaternion
import rospkg


def rpy_to_quaternion(roll, pitch, yaw):
    """
    Converts roll-pitch-yaw angles to a quaternion.

    Args:
    - roll: Roll angle in radians (float).
    - pitch: Pitch angle in radians (float).
    - yaw: Yaw angle in radians (float).

    Returns:
    - Quaternion as a list [x, y, z, w].
    """
    quaternion = tf.transformations.quaternion_from_euler(roll, pitch, yaw)
    quaternion = [quaternion[0], quaternion[1], quaternion[2], quaternion[3]]
    return quaternion


def quaternion_to_rpy(quat_msg):
    """
    Converts a quaternion to roll-pitch-yaw angles.

    Args:
    - quat_msg: Quaternion message (geometry_msgs.msg.Quaternion).

    Returns:
    - Roll, pitch, and yaw angles in radians (floats).
    """

    quat = [quat_msg.x, quat_msg.y, quat_msg.z, quat_msg.w]
    roll, pitch, yaw = euler_from_quaternion(quat)
    return roll, pitch, yaw


def toPoseStamped(arm_pose):
    pose_stamped = PoseStamped()
    pose_stamped.header.stamp = rospy.Time.now()

    pose = Pose()
    pose.orientation = Quaternion(arm_pose[0], arm_pose[1], arm_pose[2], arm_pose[3])
    pose.position = Point(arm_pose[4], arm_pose[5], arm_pose[6])
    pose_stamped.pose = pose

    return pose_stamped


def toPose(pose_stamped):
    pose = []
    pose.append(pose_stamped.pose.orientation.x)
    pose.append(pose_stamped.pose.orientation.y)
    pose.append(pose_stamped.pose.orientation.z)
    pose.append(pose_stamped.pose.orientation.w)
    pose.append(pose_stamped.pose.position.x)
    pose.append(pose_stamped.pose.position.y)
    pose.append(pose_stamped.pose.position.z)

    return pose


def interpolate_pose(pose1, pose2, duration, rate):
    """
    Interpolates between two poses over a specified duration at a given rate.

    Args:
    - pose1: First pose (geometry_msgs.msg.PoseStamped).
    - pose2: Second pose (geometry_msgs.msg.PoseStamped).
    - duration: Duration of interpolation in seconds (float).
    - rate: Rate of interpolation in Hz (int).

    Returns:
    - Interpolated poses as a list of geometry_msgs.msg.PoseStamped.
    """
    num_steps = int(duration * rate)
    interpolated_poses = []

    # Calculate step size for interpolation
    step_size = 1.0 / num_steps

    # Perform interpolation
    for i in range(num_steps + 1):
        fraction = i * step_size
        interpolated_pose = PoseStamped()

        # Interpolate position
        pos1 = np.array(
            [
                pose1.pose.position.x,
                pose1.pose.position.y,
                pose1.pose.position.z,
            ]
        )
        pos2 = np.array(
            [
                pose2.pose.position.x,
                pose2.pose.position.y,
                pose2.pose.position.z,
            ]
        )
        interp_pos = pos1 + fraction * (pos2 - pos1)
        interpolated_pose.pose.position.x = interp_pos[0]
        interpolated_pose.pose.position.y = interp_pos[1]
        interpolated_pose.pose.position.z = interp_pos[2]

        # Interpolate orientation using quaternion slerp
        quat1 = np.array(
            [
                pose1.pose.orientation.x,
                pose1.pose.orientation.y,
                pose1.pose.orientation.z,
                pose1.pose.orientation.w,
            ]
        )
        quat2 = np.array(
            [
                pose2.pose.orientation.x,
                pose2.pose.orientation.y,
                pose2.pose.orientation.z,
                pose2.pose.orientation.w,
            ]
        )
        interp_quat = tf.transformations.quaternion_slerp(quat1, quat2, fraction)
        interpolated_pose.pose.orientation.x = interp_quat[0]
        interpolated_pose.pose.orientation.y = interp_quat[1]
        interpolated_pose.pose.orientation.z = interp_quat[2]
        interpolated_pose.pose.orientation.w = interp_quat[3]

        # Set header timestamp
        interpolated_pose.header = Header(stamp=rospy.Time.now())

        interpolated_poses.append(interpolated_pose)

    return interpolated_poses


def limit_values(value, min_values, max_values):
    result = []
    for val, min_val, max_val in zip(value, min_values, max_values):
        if val < min_val:
            print(
                f"Warning: Value {val} is below minimum {min_val}. Clipping to {min_val}."
            )
            result.append(min_val)
        elif val > max_val:
            print(
                f"Warning: Value {val} is above maximum {max_val}. Clipping to {max_val}."
            )
            result.append(max_val)
        else:
            result.append(val)
    return result


def get_package_path(package_name):
    try:
        rospack = rospkg.RosPack()
        package_path = rospack.get_path(package_name)
        return package_path
    except rospkg.ResourceNotFound:
        return None
