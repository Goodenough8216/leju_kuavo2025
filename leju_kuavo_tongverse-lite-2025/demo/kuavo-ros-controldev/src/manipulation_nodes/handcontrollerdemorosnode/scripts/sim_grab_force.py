#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from geometry_msgs.msg import Wrench
from nav_msgs.msg import Odometry
import threading
import sys
import select
import tty
import termios
import numpy as np
from tf.transformations import euler_from_quaternion

# Function to capture single keypress
def get_key():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        if select.select([sys.stdin], [], [], 0.1)[0]:
            key = sys.stdin.read(1)
        else:
            key = ''
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return key

# Publisher class
class WrenchPublisher:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('wrench_publisher_node', anonymous=True)
        
        # Create publishers for left and right hand
        self.left_pub = rospy.Publisher('/external_wrench/left_hand', Wrench, queue_size=10)
        self.right_pub = rospy.Publisher('/external_wrench/right_hand', Wrench, queue_size=10)
        
        # Initialize Wrench messages
        self.left_wrench = Wrench()
        self.left_wrench.force.x = 0.0
        self.left_wrench.force.y = -10.0
        self.left_wrench.force.z = 0.0
        self.left_wrench.torque.x = 0.0
        self.left_wrench.torque.y = 0.0
        self.left_wrench.torque.z = 0.0
        
        self.right_wrench = Wrench()
        self.right_wrench.force.x = 0.0
        self.right_wrench.force.y = 10.0
        self.right_wrench.force.z = 0.0
        self.right_wrench.torque.x = 0.0
        self.right_wrench.torque.y = 0.0
        self.right_wrench.torque.z = 0.0
        
        self.robot_pose = None  # To store the robot's current pose
        
        # Control flags
        self.publish_flag = False
        self.shutdown_flag = False
        
        # Start threads
        self.publisher_thread = threading.Thread(target=self.publish_loop)
        self.publisher_thread.start()
        
        self.listener_thread = threading.Thread(target=self.key_listener)
        self.listener_thread.start()
        
        # Subscribe to /odom to get robot's pose
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)

    def odom_callback(self, msg):
        self.robot_pose = msg.pose.pose

    def publish_loop(self):
        rate = rospy.Rate(1000)  # 1000 Hz
        while not rospy.is_shutdown() and not self.shutdown_flag:
            if self.publish_flag and self.robot_pose is not None:
                # Extract orientation quaternion
                orientation_q = self.robot_pose.orientation
                quaternion = [
                    orientation_q.x,
                    orientation_q.y,
                    orientation_q.z,
                    orientation_q.w
                ]
                
                # Convert quaternion to Euler angles
                roll, pitch, yaw = euler_from_quaternion(quaternion)
                
                # Create rotation matrix around Z-axis (yaw)
                cos_yaw = np.cos(yaw)
                sin_yaw = np.sin(yaw)
                rotation_matrix = np.array([
                    [cos_yaw, -sin_yaw, 0],
                    [sin_yaw,  cos_yaw, 0],
                    [0,        0,       1]
                ])
                
                # Original force vectors in local frame
                left_force_local = np.array([
                    self.left_wrench.force.x,
                    self.left_wrench.force.y,
                    self.left_wrench.force.z
                ])
                
                right_force_local = np.array([
                    self.right_wrench.force.x,
                    self.right_wrench.force.y,
                    self.right_wrench.force.z
                ])
                
                # Transform forces to global frame
                left_force_global = rotation_matrix.dot(left_force_local)
                right_force_global = rotation_matrix.dot(right_force_local)
                
                # Update wrench messages with transformed forces
                transformed_left_wrench = Wrench()
                transformed_left_wrench.force.x = left_force_global[0]
                transformed_left_wrench.force.y = left_force_global[1]
                transformed_left_wrench.force.z = left_force_global[2]
                transformed_left_wrench.torque = self.left_wrench.torque  # Assuming torque remains the same
                
                transformed_right_wrench = Wrench()
                transformed_right_wrench.force.x = right_force_global[0]
                transformed_right_wrench.force.y = right_force_global[1]
                transformed_right_wrench.force.z = right_force_global[2]
                transformed_right_wrench.torque = self.right_wrench.torque  # Assuming torque remains the same
                
                # Publish transformed wrench messages
                self.left_pub.publish(transformed_left_wrench)
                self.right_pub.publish(transformed_right_wrench)
            elif self.publish_flag and self.robot_pose is None:
                rospy.logwarn("Robot pose not received yet. Waiting for /odom messages.")
            rate.sleep()
        rospy.loginfo("Publishing loop has been shut down.")
    
    def key_listener(self):
        rospy.loginfo("Press 's' to start publishing, 'x' to stop, 'q' to quit.")
        while not rospy.is_shutdown() and not self.shutdown_flag:
            key = get_key()
            if key:
                if key == 's':
                    self.publish_flag = True
                    rospy.loginfo("Started publishing.")
                elif key == 'x':
                    self.publish_flag = False
                    rospy.loginfo("Stopped publishing.")
                elif key == 'q':
                    self.publish_flag = False
                    self.shutdown_flag = True
                    rospy.loginfo("Shutting down.")
            rospy.sleep(0.1)
    
    def shutdown(self):
        self.shutdown_flag = True
        self.publisher_thread.join()
        self.listener_thread.join()
        rospy.loginfo("WrenchPublisher node has been shut down.")

if __name__ == '__main__':
    try:
        wp = WrenchPublisher()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        wp.shutdown()
