#!/usr/bin/env python3
# coding: utf-8
from kuavo_humanoid_sdk.kuavo.core.ros_env import KuavoROSEnv
kuavo_ros_env = KuavoROSEnv()
if not kuavo_ros_env.Init():
    raise RuntimeError("Failed to initialize ROS environment")
else:
    import time
    from typing import Tuple
    from kuavo_humanoid_sdk.kuavo.robot_info import KuavoRobotInfo
    from kuavo_humanoid_sdk.kuavo.robot_arm import KuavoRobotArm 
    from kuavo_humanoid_sdk.kuavo.robot_head import KuavoRobotHead
    from kuavo_humanoid_sdk.kuavo.core.core import KuavoRobotCore
    from kuavo_humanoid_sdk.interfaces.robot import RobotBase
    from kuavo_humanoid_sdk.interfaces.data_types import KuavoPose, KuavoIKParams
    from kuavo_humanoid_sdk.common.logger import SDKLogger, disable_sdk_logging
"""
Kuavo SDK - Python Interface for Kuavo Robot Control

This module provides the main interface for controlling Kuavo robots through Python.
It includes two main classes:

KuavoSDK:
    A static class providing SDK initialization and configuration functionality.
    Handles core setup, ROS environment initialization, and logging configuration.

KuavoRobot:
    The main robot control interface class that provides access to:
    - Robot information and status (via KuavoRobotInfo)
    - Arm control capabilities (via KuavoRobotArm)
    - Head control capabilities (via KuavoRobotHead)
    - Core robot functions (via KuavoRobotCore)
    
The module requires a properly configured ROS environment to function.
"""
__all__ = ["KuavoSDK", "KuavoRobot"]

class KuavoSDK:
    def __init__(self):
       pass

    @staticmethod   
    def Init(log_level: str = "INFO")-> bool:
        """Initialize the SDK.
        
        Args:
            log_level (str): The logging level to use. Can be "ERROR", "WARN", "INFO", "DEBUG".
                Defaults to "INFO".
                
        Returns:
            bool: True if initialization is successful, False otherwise.
            
        Raises:
            RuntimeError: If the initialization fails.
        """
        SDKLogger.setLevel(log_level.upper())
        # Initialize core components, connect ROS Topics...
        kuavo_core = KuavoRobotCore()
        if log_level.upper() == 'DEBUG':
            debug = True
        else:
            debug = False    
        if not kuavo_core.initialize(debug=debug):
            SDKLogger.error("[SDK] Failed to initialize core components.")
            return False
        
        return True
    
    @staticmethod
    def DisableLogging():
        """Disable all logging output from the SDK."""
        disable_sdk_logging()

class KuavoRobot(RobotBase):
    def __init__(self):
        super().__init__(robot_type="kuavo")
        self._robot_info = KuavoRobotInfo()
        self._robot_arm  = KuavoRobotArm()
        self._robot_head = KuavoRobotHead()
        self._kuavo_core = KuavoRobotCore()

    def stance(self)->bool:
        """Put the robot into 'stance' mode.
        
        Returns:
            bool: True if the robot is put into stance mode successfully, False otherwise.
            
        Note:
            You can call KuavoRobotState.wait_for_stance() to wait until the robot enters stance mode.
        """
        return self._kuavo_core.to_stance()
        
    def trot(self)->bool:
        """Put the robot into 'trot' mode.
        
        Returns:
            bool: True if the robot is put into trot mode successfully, False otherwise.
            
        Note:
            You can call KuavoRobotState.wait_for_walk() to wait until the robot enters trot mode.
        """
        return self._kuavo_core.to_trot()

    def walk(self, linear_x:float, linear_y:float, angular_z:float)->bool:
        """Control the robot's motion.
        
        Args:
            linear_x (float): The linear velocity along the x-axis in m/s, range [-0.4, 0.4].
            linear_y (float): The linear velocity along the y-axis in m/s, range [-0.2, 0.2].
            angular_z (float): The angular velocity around the z-axis in rad/s, range [-0.4, 0.4].
            
        Returns:
            bool: True if the motion is controlled successfully, False otherwise.
            
        Note:
            You can call KuavoRobotState.wait_for_walk() to wait until the robot enters walk mode.
        """
        # Limit velocity ranges
        limited_linear_x = min(0.4, max(-0.4, linear_x))
        limited_linear_y = min(0.2, max(-0.2, linear_y)) 
        limited_angular_z = min(0.4, max(-0.4, angular_z))
        
        # Check if any velocity exceeds limits.
        if abs(linear_x) > 0.4:
            SDKLogger.warn(f"[Robot] linear_x velocity {linear_x} exceeds limit [-0.4, 0.4], will be limited")
        if abs(linear_y) > 0.2:
            SDKLogger.warn(f"[Robot] linear_y velocity {linear_y} exceeds limit [-0.2, 0.2], will be limited")
        if abs(angular_z) > 0.4:
            SDKLogger.warn(f"[Robot] angular_z velocity {angular_z} exceeds limit [-0.4, 0.4], will be limited")
        return self._kuavo_core.walk(limited_linear_x, limited_linear_y, limited_angular_z)

    def jump(self):
        """Jump the robot."""
        pass

    def squat(self, height: float, pitch: float=0.0)->bool:
        """Control the robot's squat height and pitch.
        Args:
                height (float): The height offset from normal standing height in meters, range [-0.3, 0.0],Negative values indicate squatting down.
                pitch (float): The pitch angle of the robot's torso in radians, range [-0.4, 0.4].
            
        Returns:
            bool: True if the squat is controlled successfully, False otherwise.
        """
        # Limit height range
        limited_height = min(0.0, max(-0.3, height))
        
        # Check if height exceeds limits
        if height > 0.0 or height < -0.3:
            SDKLogger.warn(f"[Robot] height {height} exceeds limit [-0.3, 0.0], will be limited")
        # Limit pitch range
        limited_pitch = min(0.4, max(-0.4, pitch))
        
        # Check if pitch exceeds limits
        if abs(pitch) > 0.4:
            SDKLogger.warn(f"[Robot] pitch {pitch} exceeds limit [-0.4, 0.4], will be limited")
        
        return self._kuavo_core.squat(limited_height, limited_pitch)
     
    def step_by_step(self, target_pose:list, dt:float=0.4, is_left_first_default:bool=True, collision_check:bool=True)->bool:
        """Control the robot's motion by step.
        
        Args:
            target_pose (list): The target position of the robot in [x, y, z, yaw] m, rad.
            dt (float): The time interval between each step in seconds. Defaults to 0.4s.
            is_left_first_default (bool): Whether to start with the left foot. Defaults to True.
            collision_check (bool): Whether to check for collisions. Defaults to True.
            
        Returns:
            bool: True if motion is successful, False otherwise.
            
        Raises:
            RuntimeError: If the robot is not in stance state when trying to control step motion.
            ValueError: If target_pose length is not 4.
        """
        if len(target_pose) != 4:
            raise ValueError(f"[Robot] target_pose length must be 4 (x, y, z, yaw), but got {len(target_pose)}")

        return self._kuavo_core.step_control(target_pose, dt, is_left_first_default, collision_check)
    
    def control_head(self, yaw: float, pitch: float)->bool:
        """Control the head of the robot.
        
        Args:
            yaw (float): The yaw angle of the head in radians, range [-1.396, 1.396] (-80 to 80 degrees).
            pitch (float): The pitch angle of the head in radians, range [-0.436, 0.436] (-25 to 25 degrees).
            
        Returns:
            bool: True if the head is controlled successfully, False otherwise.
        """
        return self._robot_head.control_head(yaw=yaw, pitch=pitch)
    
    """ Robot Arm Control """
    def arm_reset(self)->bool:
        """Reset the robot arm.
        
        Returns:
            bool: True if the arm is reset successfully, False otherwise.
        """
        return self._robot_arm.arm_reset()
        
    def control_arm_position(self, joint_positions:list)->bool:
        """Control the position of the arm.
        
        Args:
            joint_positions (list): The target position of the arm in radians.
            
        Returns:
            bool: True if the arm is controlled successfully, False otherwise.
            
        Raises:
            ValueError: If the joint position list is not of the correct length.
            ValueError: If the joint position is outside the range of [-π, π].
            RuntimeError: If the robot is not in stance state when trying to control the arm.
        """
        if len(joint_positions) != self._robot_info.arm_joint_dof:
            print("The length of the position list must be equal to the number of DOFs of the arm.")
            return False
        
        return self._robot_arm.control_arm_position(joint_positions)
    
    def control_arm_target_poses(self, times:list, q_frames:list)->bool:
        """Control the target poses of the robot arm.
        
        Args:
            times (list): List of time intervals in seconds.
            q_frames (list): List of joint positions in radians.
            
        Returns:
            bool: True if the control was successful, False otherwise.
            
        Raises:
            ValueError: If the times list is not of the correct length.
            ValueError: If the joint position list is not of the correct length.
            ValueError: If the joint position is outside the range of [-π, π].
            RuntimeError: If the robot is not in stance state when trying to control the arm.
            
        Note:
            This is an asynchronous interface. The function returns immediately after sending the command.
            Users need to wait for the motion to complete on their own.
        """
        return self._robot_arm.control_arm_target_poses(times, q_frames)

    def set_fixed_arm_mode(self) -> bool:
        """Freezes the robot arm.
        
        Returns:
            bool: True if the arm is frozen successfully, False otherwise.
        """
        return self._robot_arm.set_fixed_arm_mode()

    def set_auto_swing_arm_mode(self) -> bool:
        """Swing the robot arm.
        
        Returns:
            bool: True if the arm is swinging successfully, False otherwise.
        """
        return self._robot_arm.set_auto_swing_arm_mode()
    
    def set_external_control_arm_mode(self) -> bool:
        """External control the robot arm.
        
        Returns:
            bool: True if the arm is external controlled successfully, False otherwise.
        """
        return self._robot_arm.set_external_control_arm_mode()
    
    """ Arm Forward kinematics && Arm Inverse kinematics """
    def arm_ik(self, 
               left_pose: KuavoPose, 
               right_pose: KuavoPose,
               left_elbow_pos_xyz: list = [0.0, 0.0, 0.0],
               right_elbow_pos_xyz: list = [0.0, 0.0, 0.0],
               arm_q0: list = None,
               params: KuavoIKParams=None) -> list:
        """Inverse kinematics for the robot arm.
        
        Args:
            left_pose (KuavoPose): Pose of the robot left arm, xyz and quat.
            right_pose (KuavoPose): Pose of the robot right arm, xyz and quat.
            left_elbow_pos_xyz (list): Position of the robot left elbow. If [0.0, 0.0, 0.0], will be ignored.
            right_elbow_pos_xyz (list): Position of the robot right elbow. If [0.0, 0.0, 0.0], will be ignored.
            arm_q0 (list, optional): Initial joint positions in radians. If None, will be ignored.
            params (KuavoIKParams, optional): Parameters for the inverse kinematics. If None, will be ignored.
                Contains:
                - major_optimality_tol: Major optimality tolerance
                - major_feasibility_tol: Major feasibility tolerance
                - minor_feasibility_tol: Minor feasibility tolerance
                - major_iterations_limit: Major iterations limit
                - oritation_constraint_tol: Orientation constraint tolerance
                - pos_constraint_tol: Position constraint tolerance, works when pos_cost_weight==0.0
                - pos_cost_weight: Position cost weight. Set to 0.0 for high accuracy
                
        Returns:
            list: List of joint positions in radians, or None if inverse kinematics failed.
        """
        return self._robot_arm.arm_ik(left_pose, right_pose, left_elbow_pos_xyz, right_elbow_pos_xyz, arm_q0, params)

    def arm_fk(self, q: list) -> Tuple[KuavoPose, KuavoPose]:
        """Forward kinematics for the robot arm.
        
        Args:
            q (list): List of joint positions in radians.
            
        Returns:
            Tuple[KuavoPose, KuavoPose]: Tuple of poses for the robot left arm and right arm,
                or (None, None) if forward kinematics failed.
        """
        return self._robot_arm.arm_fk(q)