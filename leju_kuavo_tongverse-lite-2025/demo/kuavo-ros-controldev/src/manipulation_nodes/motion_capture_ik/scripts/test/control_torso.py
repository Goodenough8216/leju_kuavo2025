#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from geometry_msgs.msg import Twist
import sys
import time
import termios
import tty
import select
import argparse

class KeyListener:
    def __init__(self):
        self.exit_program = False
        self.key_callbacks = {}
        self.old_settings = termios.tcgetattr(sys.stdin)

    def register_callback(self, key, callback):
        """ 注册按键和对应的回调函数 """
        self.key_callbacks[key] = callback

    def unregister_callback(self, key):
        """ 注销按键的回调函数 """
        if key in self.key_callbacks:
            del self.key_callbacks[key]

    def getKey(self):
        tty.setraw(sys.stdin.fileno())
        rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
        if rlist:
            key = sys.stdin.read(1)
        else:
            key = ''
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
        return key
    
    def on_press(self, key):
        try:
            if key in self.key_callbacks and callable(self.key_callbacks[key]):
                self.key_callbacks[key](key)
        except AttributeError:
            # 某些特殊键（如功能键）可能没有字符属性
            pass
        except Exception as e:
            print(f"Error processing key: {e}")
        print("pressed key: '", key, "'",end='\r')
    def stop(self):
        self.exit_program = True
    def loop_control(self):
        try:
            while not self.exit_program:
                key = self.getKey()
                if key:
                    self.on_press(key)
                if (key == '\x03'):  # Ctrl-C
                    break 
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)    

def round_value(value):
    """
    Round the value to one decimal place
    """
    return round(value, 2)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Control torso')
    parser.add_argument('--pitch_step', type=float, default=0.5, help='PITCH angle step')
    parser.add_argument('--height_step', type=float, default=0.5, help='height step')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    # Initialize ROS node
    rospy.init_node('cmd_pose_publisher')

    # Create publisher
    cmd_pose_pub = rospy.Publisher('/cmd_pose', Twist, queue_size=10)

    # Initialize Twist message
    cmd_pose_msg = Twist()
    cmd_pose_msg.linear.x = 0.0  # X position offset in meters
    cmd_pose_msg.linear.y = 0.0  # Y position offset in meters
    cmd_pose_msg.linear.z = 0.0  # Z position offset in meters (height)
    cmd_pose_msg.angular.x = 0.0
    cmd_pose_msg.angular.y = 0.0  # Pitch angle in radians
    cmd_pose_msg.angular.z = 0.0  # Yaw angle in radians

    # Set control step size
    pitch_step = args.pitch_step
    height_step = args.height_step

    # Initialize keyboard listener
    kl = KeyListener()

    # Define key mapping for controls
    key_to_control = {
        'w': ('angular_y', pitch_step),     # Increase pitch angle
        's': ('angular_y', -pitch_step),    # Decrease pitch angle
        'i': ('linear_z', height_step),      # Increase height
        'k': ('linear_z', -height_step)      # Decrease height
    }

    def cmd_pose_callback(key):
        """
        Callback function for keyboard control
        Updates pose based on key press and publishes command
        """
        if key in key_to_control:
            control_type, delta = key_to_control[key]
            
            if control_type == 'angular_y':
                cmd_pose_msg.angular.y += delta
                cmd_pose_msg.angular.y = round_value(cmd_pose_msg.angular.y)
                print(f"Updated angular.y (pitch): {cmd_pose_msg.angular.y}")
            elif control_type == 'linear_z':
                cmd_pose_msg.linear.z += delta
                cmd_pose_msg.linear.z = round_value(cmd_pose_msg.linear.z)
                print(f"Updated linear.z (height): {cmd_pose_msg.linear.z}")
            
            cmd_pose_pub.publish(cmd_pose_msg)

    # Register keyboard callbacks
    for key in key_to_control.keys():
        kl.register_callback(key, cmd_pose_callback)

    try:
        print("\033[96m---------------------------------------------\n"
              "Pose Control:\n"
              "  Pitch: [Key <w>, Key <s>] (angular.y)\n"
              "  Height: [Key <i>, Key <k>] (linear.z)\n"
              "----------------------------------------------\033[0m")
        # Start keyboard control loop
        kl.loop_control()
    except KeyboardInterrupt:
        kl.stop()