#!/usr/bin/env python3

import rospy
import tf2_ros
import numpy as np
import math
import sys, termios, tty
import signal
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import JointState
from tf.transformations import quaternion_matrix, translation_matrix, concatenate_matrices, inverse_matrix, euler_from_matrix
import matplotlib.pyplot as plt
import pandas as pd
import time
import random
import argparse
from scipy.stats import norm
from kuavo_msgs.srv import changeArmCtrlMode
from kuavo_msgs.msg import sensorsData
from scipy.interpolate import CubicSpline
import os
from datetime import datetime


def cubic_spline_interpolation(trajectory, rate):
    """
    Performs cubic spline interpolation between waypoints in a trajectory.

    Args:
        trajectory: List of dictionaries, each containing:
            - 'time': timestamp for the waypoint
            - 'pos': joint positions at that timestamp (list of floats)
        rate: Interpolation rate in Hz

    Returns:
        A list of interpolated joint positions, where each element is list of floats.
    """
    if len(trajectory) < 2:
        return [trajectory[0]['pos']]

    # Extract times and positions
    times = [point['time'] for point in trajectory]
    positions = [point['pos'] for point in trajectory]
    
    # Calculate total number of steps
    total_time = times[-1] - times[0]
    num_steps = int(total_time * rate)
    interpolation_times = np.linspace(times[0], times[-1], num_steps + 1)
    
    interpolated_positions = []
    num_joints = len(positions[0])

    # Interpolate each joint separately
    for joint_idx in range(num_joints):
        joint_positions = []
        
        # Interpolate between each pair of waypoints
        for t in interpolation_times:
            # Find the segment this time belongs to
            if t >= times[-1]:
                segment_idx = len(times) - 2
            else:
                segment_idx = 0
                for i in range(len(times) - 1):
                    if times[i] <= t <= times[i + 1]:
                        segment_idx = i
                        break
            
            # Ensure segment_idx is valid
            segment_idx = min(segment_idx, len(positions) - 2)
            
            # Get segment start and end points
            x0 = times[segment_idx]
            x1 = times[segment_idx + 1]
            y0 = positions[segment_idx][joint_idx]
            y1 = positions[segment_idx + 1][joint_idx]
            
            # Calculate cubic spline coefficients
            a = y0
            b = 0
            c = (3 * (y1 - y0) / ((x1 - x0)**2))
            d = (-2 * (y1-y0) / ((x1-x0)**3))
            
            # Calculate interpolated position
            pos = a + b*(t - x0) + c*(t - x0)**2 + d*(t - x0)**3
            joint_positions.append(pos)
            
        interpolated_positions.append(joint_positions)
    
    return np.array(interpolated_positions).T.tolist()

def format_array(arr):
    """Format numpy array to have 3 decimal places"""
    if isinstance(arr, np.ndarray):
        return np.round(arr, 3)
    return round(arr, 3)

def matrix_from_transform(transform):
    """Convert a transform message to a 4x4 matrix."""
    trans = transform.transform.translation
    rot = transform.transform.rotation
    trans_matrix = translation_matrix([trans.x, trans.y, trans.z])
    rot_matrix = quaternion_matrix([rot.x, rot.y, rot.z, rot.w])
    return concatenate_matrices(trans_matrix, rot_matrix)

def get_transform(tf_buffer, target_frame, source_frame):
    """Get transform between two frames using tf2."""
    try:
        transform = tf_buffer.lookup_transform(
            target_frame, source_frame, rospy.Time(0), rospy.Duration(4.0)
        )
        return transform
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
        rospy.logerr(f"Failed to lookup transform from {source_frame} to {target_frame}: {e}")
        return None

def enable_control():
    """Call service to set control mode to 2 (enable control)"""
    try:
        rospy.wait_for_service('/arm_traj_change_mode', timeout=1.0)
        change_mode = rospy.ServiceProxy('/arm_traj_change_mode', changeArmCtrlMode)
        response = change_mode(2)
        rospy.loginfo("Enabled control, service response: %s", response)
        return True
    except rospy.ROSException as e:
        rospy.logerr("Service call failed: %s", e)
        return False

def disable_control():
    """Call service to set control mode to 1 (disable control)"""
    try:
        rospy.wait_for_service('/arm_traj_change_mode', timeout=1.0)
        change_mode = rospy.ServiceProxy('/arm_traj_change_mode', changeArmCtrlMode)
        response = change_mode(1)
        rospy.loginfo("Disabled control, service response: %s", response)
        return True
    except rospy.ROSException as e:
        rospy.logerr("Service call failed: %s", e)
        return False

def publish_arm_joints(publisher, joint_names, joint_positions, current_arm_positions):
    """Publish arm joint information to /kuavo_arm_traj topic
    
    Args:
        publisher: ROS publisher for arm trajectory
        joint_names: List of all joint names
        joint_positions: Target joint positions
        current_positions: Current joint positions (if provided) to use as start
    """
    # Extract arm joint names and positions
    arm_joint_names = []
    arm_joint_positions = []
    
    for i, name in enumerate(joint_names):
        if "arm" in name.lower():
            arm_joint_names.append(name)
            # Convert to degrees
            arm_joint_positions.append(math.degrees(joint_positions[i]))
    
    # If no arm joints found, return
    if len(arm_joint_names) == 0:
        rospy.logwarn("No arm joints found in joint_names")
        return

    current_arm_positions = [math.degrees(pos) for pos in current_arm_positions]
    traj = [
        {"time":0.0, "pos":current_arm_positions},
        {"time":2.0, "pos":arm_joint_positions},
    ]

    freq = 100
    traj_pos = cubic_spline_interpolation(traj, freq)
    joint_state_msg = JointState()
    joint_state_msg.header.stamp = rospy.Time.now()
    joint_state_msg.name = arm_joint_names
    for pos in traj_pos:
        joint_state_msg.position = pos
        publisher.publish(joint_state_msg)
        time.sleep(1/freq)

    rospy.sleep(1.0) # Wait 1 more second for the arm to finish moving
    rospy.loginfo("Trajectory execution completed!")

def set_joint_positions(arm_publisher, joint_names, positions):
    # Publish to arm topic using spline trajectory
    publish_arm_joints(arm_publisher, joint_names, positions)
    
    # Allow enough time for the robot to execute the trajectory (trajectory itself takes 2 seconds)
    rospy.sleep(3.0)

def calculate_relative_transform(transform_base_to_marker, transform_base_to_effector):
    """Calculate the transform from marker to effector."""
    base_to_marker_matrix = matrix_from_transform(transform_base_to_marker)
    base_to_effector_matrix = matrix_from_transform(transform_base_to_effector)
    
    # Calculate the transform from effector to marker: effector_to_marker = inv(base_to_effector) * base_to_marker
    effector_to_marker_matrix = np.dot(inverse_matrix(base_to_effector_matrix), base_to_marker_matrix)
    return effector_to_marker_matrix

def calculate_rmse(translations, mean_translation):
    """Calculate Root Mean Square Error."""
    squared_diff = np.sum((translations - mean_translation) ** 2, axis=1)
    return np.sqrt(np.mean(squared_diff))

def extract_rotation_euler(transform_matrix):
    """Extract rotation in Euler angles (XYZ) from transformation matrix."""
    return euler_from_matrix(transform_matrix[:3, :3], 'sxyz')

def angular_distance(rot1, rot2, in_degrees=True):
    """Calculate angular distance between two rotation matrices."""
    # Get trace of rotation difference
    R_diff = np.dot(rot1, rot2.T)
    trace = np.trace(R_diff)
    trace = min(3.0, max(-1.0, (trace - 1) / 2))
    
    angle_rad = np.arccos(trace)
    if in_degrees:
        return np.degrees(angle_rad)
    return angle_rad

def calculate_statistics(transforms):
    """Calculate statistics for a set of transforms."""
    # Extract translations from transforms
    translations = np.array([transform[:3, 3] for transform in transforms])
    
    # Extract rotations (Euler angles)
    rotations = np.array([extract_rotation_euler(transform) for transform in transforms])
    rotation_matrices = [transform[:3, :3] for transform in transforms]
    
    # Translation statistics
    mean_translation = np.mean(translations, axis=0)
    std_translation = np.std(translations, axis=0)
    var_translation = np.var(translations, axis=0)
    max_diff_translation = np.max(np.abs(translations - mean_translation), axis=0)
    rmse_translation = calculate_rmse(translations, mean_translation)
    
    # Rotation statistics (Euler angles)
    mean_rotation = np.mean(rotations, axis=0)
    std_rotation = np.std(rotations, axis=0)
    var_rotation = np.var(rotations, axis=0)
    max_diff_rotation = np.max(np.abs(rotations - mean_rotation), axis=0)
    rmse_rotation = calculate_rmse(rotations, mean_rotation)
    
    # Angular difference statistics
    mean_rotation_matrix = np.mean(rotation_matrices, axis=0)
    # Need to orthogonalize the mean rotation matrix
    U, _, Vt = np.linalg.svd(mean_rotation_matrix, full_matrices=False)
    mean_rotation_matrix = np.dot(U, Vt)
    
    angular_differences = [angular_distance(mean_rotation_matrix, rot) for rot in rotation_matrices]
    mean_angular_diff = np.mean(angular_differences)
    max_angular_diff = np.max(angular_differences)
    std_angular_diff = np.std(angular_differences)
    
    return {
        'mean_translation': mean_translation,
        'std_translation': std_translation,
        'var_translation': var_translation,
        'max_diff_translation': max_diff_translation,
        'rmse_translation': rmse_translation,
        'mean_rotation': mean_rotation,
        'std_rotation': std_rotation,
        'var_rotation': var_rotation,
        'max_diff_rotation': max_diff_rotation,
        'rmse_rotation': rmse_rotation,
        'mean_angular_diff': mean_angular_diff,
        'max_angular_diff': max_angular_diff,
        'std_angular_diff': std_angular_diff
    }

def plot_results(statistics, output_dir):
    """Plot the results."""
    # Create directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Plot translation statistics
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    axes = ['X', 'Y', 'Z']
    for i, ax in enumerate(axs):
        ax.bar(['Mean', 'Std Dev', 'Max Diff'], 
               [statistics['mean_translation'][i], 
                statistics['std_translation'][i], 
                statistics['max_diff_translation'][i]])
        ax.set_title(f'{axes[i]} Statistics')
        ax.set_ylabel('Meters')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/translation_statistics.png")
    plt.close()
    
    # Plot rotation statistics
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, ax in enumerate(axs):
        ax.bar(['Mean', 'Std Dev', 'Max Diff'], 
               [statistics['mean_rotation'][i], 
                statistics['std_rotation'][i], 
                statistics['max_diff_rotation'][i]])
        ax.set_title(f'Rotation {axes[i]} Statistics')
        ax.set_ylabel('Radians')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/rotation_statistics.png")
    plt.close()
    
    # Save statistics to CSV
    df = pd.DataFrame({
        'Statistic': ['Mean', 'Std Dev', 'Variance', 'Max Diff', 'RMSE'],
        'X': [statistics['mean_translation'][0], 
              statistics['std_translation'][0], 
              statistics['var_translation'][0], 
              statistics['max_diff_translation'][0],
              statistics['rmse_translation']],
        'Y': [statistics['mean_translation'][1], 
              statistics['std_translation'][1], 
              statistics['var_translation'][1], 
              statistics['max_diff_translation'][1],
              statistics['rmse_translation']],
        'Z': [statistics['mean_translation'][2], 
              statistics['std_translation'][2], 
              statistics['var_translation'][2], 
              statistics['max_diff_translation'][2],
              statistics['rmse_translation']]
    })
    
    df.to_csv(f"{output_dir}/translation_statistics.csv", index=False)
    
    # Save rotation statistics
    df_rotation = pd.DataFrame({
        'Statistic': ['Mean', 'Std Dev', 'Variance', 'Max Diff', 'RMSE', 'Mean Angular Diff', 'Max Angular Diff'],
        'X': [statistics['mean_rotation'][0], 
              statistics['std_rotation'][0], 
              statistics['var_rotation'][0], 
              statistics['max_diff_rotation'][0],
              statistics['rmse_rotation'],
              statistics['mean_angular_diff'],
              statistics['max_angular_diff']],
        'Y': [statistics['mean_rotation'][1], 
              statistics['std_rotation'][1], 
              statistics['var_rotation'][1], 
              statistics['max_diff_rotation'][1],
              statistics['rmse_rotation'],
              0, 0],
        'Z': [statistics['mean_rotation'][2], 
              statistics['std_rotation'][2], 
              statistics['var_rotation'][2], 
              statistics['max_diff_rotation'][2],
              statistics['rmse_rotation'],
              0, 0]
    })
    
    df_rotation.to_csv(f"{output_dir}/rotation_statistics.csv", index=False)
    
    print(f"Results saved to {output_dir}")

# Add keyboard input retrieval functionality
def get_key():
    """Get character input from keyboard"""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

# Add sensor data callback
class SensorDataHandler:
    def __init__(self):
        self.latest_sensor_data = None
        self.current_joint_positions = None
        self.joint_names = None
        
    def sensor_data_callback(self, data):
        """Sensor data callback function, save the latest sensor data"""
        self.latest_sensor_data = data
        
        # If joint names are set, update current joint positions
        if self.joint_names is not None:
            self.update_current_joint_positions()
    
    def update_current_joint_positions(self):
        """Update current joint positions from sensor data"""
        if self.latest_sensor_data is None:
            rospy.logwarn("No sensor data received yet")
            return False
            
        # Ensure joint_q array has enough elements
        if len(self.latest_sensor_data.joint_data.joint_q) < 26:
            rospy.logwarn("Sensor data does not have enough joint_q values")
            return False
            
        # Get joint_q values from index 13 to 26
        arm_joint_values = self.latest_sensor_data.joint_data.joint_q[12:26]
        
        # Get arm joint names
        arm_joint_names = [name for name in self.joint_names if "arm" in name.lower()]
        
        # Ensure the number of joints matches the number of values obtained
        if len(arm_joint_names) != len(arm_joint_values):
            rospy.logwarn(f"Mismatch between arm joints ({len(arm_joint_names)}) and sensor values ({len(arm_joint_values)})")
            # If the count does not match, try to use the available minimum count
            min_len = min(len(arm_joint_names), len(arm_joint_values))
            arm_joint_names = arm_joint_names[:min_len]
            arm_joint_values = arm_joint_values[:min_len]
        
        # Create a complete joint position list (including non-arm joints)
        if self.current_joint_positions is None:
            self.current_joint_positions = [0.0] * len(self.joint_names)
        
        # Update arm joint positions
        for name, value in zip(arm_joint_names, arm_joint_values):
            if name in self.joint_names:
                index = self.joint_names.index(name)
                self.current_joint_positions[index] = value
        
        return True

def main():
    output_dir = os.path.expanduser(f'~/.ros/easy_handeye/calibration_evaluation/{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    parser = argparse.ArgumentParser(description='Evaluate hand-eye calibration accuracy')
    parser.add_argument('--repeats_per_position', type=int, default=10, help='Number of repeat measurements for each joint position')
    parser.add_argument('--calibration_file', type=str, default=None, help='Path to the calibration file')
    parser.add_argument('--robot_base_frame', type=str, default='base_link', help='Robot base frame')
    parser.add_argument('--robot_effector_frame', type=str, default='zarm_r7_end_effector', help='Robot effector frame')
    parser.add_argument('--aruco_marker_frame', type=str, default='aruco_marker_frame', help='Aruco marker frame')
    
    # Use parse_known_args instead of parse_args to ignore unrecognized parameters
    args, unknown = parser.parse_known_args()

    rospy.init_node('hand_eye_calibration_evaluator')
    
    # Parameters (in a real application, these could be loaded from ROS parameters)
    robot_base_frame = args.robot_base_frame
    robot_effector_frame = args.robot_effector_frame
    aruco_marker_frame = args.aruco_marker_frame
    
    # Joint parameters
    joint_names = ['zarm_l1_link', 'zarm_l2_link', 'zarm_l3_link', 'zarm_l4_link', 'zarm_l5_link', 'zarm_l6_link', 'zarm_l7_link', 'zarm_r1_link', 'zarm_r2_link', 'zarm_r3_link', 'zarm_r4_link', 'zarm_r5_link', 'zarm_r6_link', 'zarm_r7_link']
    
    # Setup TF listener
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)
    
    # Setup arm trajectory publisher
    arm_pub = rospy.Publisher('/kuavo_arm_traj', JointState, queue_size=10)
    
    # Create sensor data handler
    sensor_handler = SensorDataHandler()
    sensor_handler.joint_names = joint_names
    
    # Subscribe to sensor data
    sensor_subscriber = rospy.Subscriber('/sensors_data_raw', sensorsData, sensor_handler.sensor_data_callback)
    
    # Wait for the TF tree to be available
    rospy.loginfo("Waiting for TF tree...")
    rospy.sleep(2.0)
    
    # Enable arm control
    rospy.loginfo("Enabling arm control...")
    if not enable_control():
        rospy.logwarn("Failed to enable arm control, continuing anyway")
    
    # List to save joint positions
    saved_joint_positions = []
    
    # Load preset joint positions based on ROBOT_VERSION
    try:
        # Read ROBOT_VERSION from environment variable
        robot_version = os.environ.get('ROBOT_VERSION', None)
        if robot_version:
            rospy.loginfo(f"Detected ROBOT_VERSION: {robot_version}")
            
            # Construct path to preset file
            preset_file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                          "calibration_evaluation", "preset_arm_joint_pos.json")
            
            # Check if file exists
            if os.path.exists(preset_file_path):
                with open(preset_file_path, 'r') as f:
                    import json
                    preset_data = json.load(f)
                    
                    # Check if this robot version has preset positions
                    if robot_version in preset_data:
                        preset_positions = preset_data[robot_version]['joint_pos']
                        rospy.loginfo(f"Found {len(preset_positions)} preset positions for ROBOT_VERSION {robot_version}")
                        
                        # Append preset positions to saved_joint_positions
                        for pos in preset_positions:
                            saved_joint_positions.append(pos)
                        
                        rospy.loginfo(f"Loaded {len(preset_positions)} preset joint positions")
                    else:
                        rospy.logwarn(f"No preset positions found for ROBOT_VERSION {robot_version}")
            else:
                rospy.logwarn(f"Preset file not found: {preset_file_path}")
        else:
            rospy.loginfo("ROBOT_VERSION environment variable not set, no preset positions loaded")
    except Exception as e:
        rospy.logerr(f"Error loading preset joint positions: {e}")
    
    # List to save transforms
    marker_to_effector_transforms = []
    
    # Capture CTRL+C signal to ensure proper exit
    def signal_handler(sig, frame):
        rospy.loginfo("Disabling arm control and exiting...")
        disable_control()
        rospy.signal_shutdown("User requested exit")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Start interactive mode
    evaluate_mode = False
    rospy.loginfo("=== Hand-Eye Calibration Evaluation Interactive Mode ===")
    rospy.loginfo("Press 'R' to record the current joint position")
    rospy.loginfo("Press 'S' to start the evaluation process")
    rospy.loginfo("Press 'P' to print the current arm joint positions")
    rospy.loginfo("Press 'Q' to exit")
    
    try:
        while not rospy.is_shutdown():
            key = get_key()
            
            if key == 'r' or key == 'R':
                # Update current joint position
                if sensor_handler.update_current_joint_positions():
                    # Save current joint position
                    if sensor_handler.current_joint_positions:
                        saved_joint_positions.append(sensor_handler.current_joint_positions.copy())
                        
                        if len(saved_joint_positions) >= 3:
                            rospy.loginfo("Sufficient joint positions collected, press 'S' to start evaluation")
                else:
                    rospy.logwarn("Unable to retrieve current joint position, please ensure sensor data is normal")
            
            elif key == 'p' or key == 'P':
                # Update current joint position
                if sensor_handler.update_current_joint_positions():
                    # Print current arm joint positions
                    arm_joint_names = [name for name in joint_names if "arm" in name.lower()]
                    arm_joint_positions = []
                    
                    for name in arm_joint_names:
                        if name in joint_names:
                            index = joint_names.index(name)
                            arm_joint_positions.append(sensor_handler.current_joint_positions[index])
                    
                    rospy.loginfo("Current Arm Joint Positions:")
                    rospy.loginfo(f"{arm_joint_positions}")
                else:
                    rospy.logwarn("Unable to retrieve current joint position, please ensure sensor data is normal")
            
            elif key == 's' or key == 'S':
                if len(saved_joint_positions) == 0:
                    rospy.logwarn("No saved joint positions, please record positions first by pressing 'R'")
                    continue
                
                evaluate_mode = True
                rospy.loginfo(f"Starting evaluation, using {len(saved_joint_positions)} saved joint positions")
                break
            
            elif key == 'q' or key == 'Q':
                rospy.loginfo("User requested exit")
                disable_control()
                return
            
            # Simple throttling to prevent high CPU usage
            rospy.sleep(0.1)
        
        # Execute evaluation process
        if evaluate_mode:
            # All transforms for all positions
            all_marker_to_effector_transforms = []
            # Transform list for each position
            position_transforms = []
            
            for i, joint_positions in enumerate(saved_joint_positions):
                rospy.loginfo(f"Processing saved position {i+1}/{len(saved_joint_positions)}")
                
                # Transform list for current position
                current_transforms = []
                
                # Repeat measurement N times for each position
                for repeat in range(args.repeats_per_position):
                    enable_control()

                    rospy.loginfo(f"  Repeating measurement {repeat+1}/{args.repeats_per_position}")
                    
                    # Set joint position
                    publish_arm_joints(arm_pub, joint_names, joint_positions, sensor_handler.current_joint_positions)
                    
                    # Wait for the robot to reach the position
                    rospy.sleep(1.0)
                    
                    # Calculate the relative transform from marker to end effector
                    marker_to_effector_transform = get_transform(tf_buffer, aruco_marker_frame, robot_effector_frame)

                    marker_to_effector_matrix = matrix_from_transform(marker_to_effector_transform)
                    
                    # Save transform to current position list
                    current_transforms.append(marker_to_effector_matrix)
                    
                    # Also save to global list
                    all_marker_to_effector_transforms.append(marker_to_effector_matrix)
                    
                    disable_control()

                    rospy.sleep(2.0)
                    
                
                # If there are valid transforms for the current position, calculate statistics
                if current_transforms:
                    position_transforms.append(current_transforms)
                    stats = calculate_statistics(current_transforms)
                    
                    rospy.loginfo(f"=== Evaluation Results for Position {i+1} ===")
                    rospy.loginfo("   Translation Statistics:")
                    rospy.loginfo(f"  Mean Translation: {format_array(stats['mean_translation'])}")
                    rospy.loginfo(f"  Translation Standard Deviation: {format_array(stats['std_translation'])}")
                    rospy.loginfo(f"  Translation Variance: {format_array(stats['var_translation'])}")
                    rospy.loginfo(f"  Maximum Translation Difference: {format_array(stats['max_diff_translation'])}")
                    rospy.loginfo(f"  Translation RMSE: {format_array(stats['rmse_translation'])}")
                    
                    rospy.loginfo("   Rotation Statistics:")
                    rospy.loginfo(f"  Mean Rotation: {format_array(stats['mean_rotation'])}")
                    rospy.loginfo(f"  Rotation Standard Deviation: {format_array(stats['std_rotation'])}")
                    rospy.loginfo(f"  Rotation Variance: {format_array(stats['var_rotation'])}")
                    rospy.loginfo(f"  Maximum Rotation Difference: {format_array(stats['max_diff_rotation'])}")
                    rospy.loginfo(f"  Rotation RMSE: {format_array(stats['rmse_rotation'])}")
                    
                else:
                    rospy.logwarn(f"No valid transforms collected for position {i+1}")
            
            # Calculate overall statistics for all positions
            if not all_marker_to_effector_transforms:
                rospy.logerr("No valid transforms collected. Check TF frames and robot movement.")
                return
            
            overall_statistics = calculate_statistics(all_marker_to_effector_transforms)
            
            # Print overall statistics
            rospy.loginfo("")
            rospy.loginfo("")
            rospy.loginfo("")
            rospy.loginfo("#######################################################")
            rospy.loginfo("### Overall Hand-Eye Calibration Evaluation Results ###")
            rospy.loginfo("#######################################################")
            rospy.loginfo(f"Mean Translation: {format_array(overall_statistics['mean_translation'])}")
            rospy.loginfo(f"Translation Standard Deviation: {format_array(overall_statistics['std_translation'])}")
            rospy.loginfo(f"Translation Variance: {format_array(overall_statistics['var_translation'])}")
            rospy.loginfo(f"Maximum Translation Difference: {format_array(overall_statistics['max_diff_translation'])}")
            rospy.loginfo(f"Translation RMSE: {format_array(overall_statistics['rmse_translation'])}")
            
            # Add rotation statistics output
            rospy.loginfo(f"Mean Rotation (Euler Angles): {format_array(overall_statistics['mean_rotation'])}")
            rospy.loginfo(f"Rotation Standard Deviation: {format_array(overall_statistics['std_rotation'])}")
            rospy.loginfo(f"Mean Angular Difference: {format_array(overall_statistics['mean_angular_diff'])} degrees")
            rospy.loginfo(f"Maximum Angular Difference: {format_array(overall_statistics['max_angular_diff'])} degrees")
            rospy.loginfo(f"Rotation RMSE: {format_array(overall_statistics['rmse_rotation'])}")
            
            
            # Save results for each position
            for i, pos_transforms in enumerate(position_transforms):
                pos_stats = calculate_statistics(pos_transforms)
                pos_dir = os.path.join(output_dir, f"position_{i+1}")

                if not os.path.exists(pos_dir):
                    os.makedirs(pos_dir)
                
                # Save statistics for this position to CSV
                df = pd.DataFrame({
                    'Statistic': ['Mean', 'Std Dev', 'Variance', 'Max Diff', 'RMSE'],
                    'X': [pos_stats['mean_translation'][0], 
                        pos_stats['std_translation'][0], 
                        pos_stats['var_translation'][0], 
                        pos_stats['max_diff_translation'][0],
                        pos_stats['rmse_translation']],
                    'Y': [pos_stats['mean_translation'][1], 
                        pos_stats['std_translation'][1], 
                        pos_stats['var_translation'][1], 
                        pos_stats['max_diff_translation'][1],
                        pos_stats['rmse_translation']],
                    'Z': [pos_stats['mean_translation'][2], 
                        pos_stats['std_translation'][2], 
                        pos_stats['var_translation'][2], 
                        pos_stats['max_diff_translation'][2],
                        pos_stats['rmse_translation']]
                })
                
                df.to_csv(f"{pos_dir}/position_{i+1}_statistics.csv", index=False)
            
            # Plot and save overall results
            plot_results(overall_statistics, output_dir)
            
            os.system(f"cp {args.calibration_file} {output_dir}/calibration.yaml")
    
    finally:
        disable_control()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        # Ensure control is disabled even on interruption
        disable_control()
        rospy.sleep(1.0)
        rospy.signal_shutdown("User requested exit")
        pass
