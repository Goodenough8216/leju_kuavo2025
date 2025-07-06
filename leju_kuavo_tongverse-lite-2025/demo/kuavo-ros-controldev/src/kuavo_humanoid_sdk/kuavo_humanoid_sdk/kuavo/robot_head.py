#!/usr/bin/env python3
# coding: utf-8
import math
from kuavo_humanoid_sdk.kuavo.core.core import KuavoRobotCore
from kuavo_humanoid_sdk.common.logger import SDKLogger

class KuavoRobotHead:
    def __init__(self):
        self._kuavo_core = KuavoRobotCore()
    
    def control_head(self, yaw: float, pitch: float)->bool:
        """
            Control the head of the robot.
            Args:
                yaw (float): The yaw angle of the head in radians, range [-1.396, 1.396] (-80 to 80 degrees).
                pitch (float): The pitch angle of the head in radians, range [-0.436, 0.436] (-25 to 25 degrees).
            Returns:
                bool: True if the head is controlled successfully, False otherwise.
        """
        # Check yaw limits (-80 to 80 degrees)
        if yaw < -math.pi*4/9 or yaw > math.pi*4/9:  # -80 to 80 degrees in radians
            SDKLogger.warn(f"[Robot] yaw {yaw} exceeds limit [-{math.pi*4/9:.3f}, {math.pi*4/9:.3f}] radians (-80 to 80 degrees), will be limited")
        limited_yaw = min(math.pi*4/9, max(-math.pi*4/9, yaw))

        # Check pitch limits (-25 to 25 degrees)
        if pitch < -math.pi/7.2 or pitch > math.pi/7.2:  # -25 to 25 degrees in radians
            SDKLogger.warn(f"[Robot] pitch {pitch} exceeds limit [-{math.pi/7.2:.3f}, {math.pi/7.2:.3f}] radians (-25 to 25 degrees), will be limited")
        limited_pitch = min(math.pi/7.2, max(-math.pi/7.2, pitch))
        return self._kuavo_core.control_robot_head(yaw=limited_yaw, pitch=limited_pitch)
