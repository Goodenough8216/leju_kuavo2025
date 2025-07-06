#!/usr/bin/env python3
# coding: utf-8

import math
from typing import Tuple
from kuavo_humanoid_sdk.interfaces.data_types import KuavoArmCtrlMode, KuavoIKParams, KuavoPose
from kuavo_humanoid_sdk.kuavo.core.core import KuavoRobotCore
from kuavo_humanoid_sdk.kuavo.robot_info import KuavoRobotInfo

class KuavoRobotArm:
    def __init__(self):
        self._kuavo_core = KuavoRobotCore()
        self._robot_info = KuavoRobotInfo(robot_type="kuavo")
    
    def arm_reset(self)-> bool:
        return self._kuavo_core.robot_arm_reset()
        
    def control_arm_position(self, joint_position:list)->bool:
        """
            Control the position of the robot arm joint.
            Args:
                joint_position (list): List of joint positions in radians
            Raises:
                ValueError: If the joint position list is not of the correct length.
                ValueError: If the joint position is outside the range of [-π, π].
                RuntimeError: If the robot is not in stance state when trying to control the arm.
            Returns:
                True if the control was successful, False otherwise.
        """
        if len(joint_position) != self._robot_info.arm_joint_dof:
            raise ValueError("Invalid position length. Expected {}, got {}".format(self._robot_info.arm_joint_dof, len(joint_position)))
        
        # Check if joint positions are within ±180 degrees (±π radians)
        for pos in joint_position:
            if abs(pos) > math.pi:
                raise ValueError(f"Joint position {pos} rad exceeds ±π rad (±180 deg) limit")

        return self._kuavo_core.control_robot_arm_traj(joint_data=joint_position)

    def control_arm_target_poses(self, times:list, q_frames:list)->bool:
        """
            Control the target poses of the robot arm.
            Args:
                times (list): List of time intervals in seconds
                joint_q (list): List of joint positions in radians
            Raises:
                ValueError: If the times list is not of the correct length.
                ValueError: If the joint position list is not of the correct length.
                ValueError: If the joint position is outside the range of [-π, π].
                RuntimeError: If the robot is not in stance state when trying to control the arm.
            Returns:
                bool: True if the control was successful, False otherwise.    
        """
        if len(times) != len(q_frames):
            raise ValueError("Invalid input. times and joint_q must have thesame length.")
        
        # Check if joint positions are within ±180 degrees (±π radians)
        q_degs = []
        for q in q_frames:
            if any(abs(pos) > math.pi for pos in q):
                raise ValueError("Joint positions must be within ±π rad (±180 deg)")
            if len(q) != self._robot_info.arm_joint_dof:
                raise ValueError("Invalid position length. Expected {}, got {}".format(self._robot_info.arm_joint_dof, len(q)))
            # Convert joint positions from radians to degrees
            q_degs.append([(p * 180.0 / math.pi) for p in q])

        return self._kuavo_core.control_robot_arm_target_poses(times=times, joint_q=q_degs)

    def set_fixed_arm_mode(self) -> bool:
        """
        Freezes the robot arm.
        Returns:
            bool: True if the arm is frozen successfully, False otherwise.
        """
        return self._kuavo_core.change_robot_arm_ctrl_mode(KuavoArmCtrlMode.ArmFixed)

    def set_auto_swing_arm_mode(self) -> bool:
        """
        Swing the robot arm.
        Returns:
            bool: True if the arm is swinging successfully, False otherwise.
        """
        return self._kuavo_core.change_robot_arm_ctrl_mode(KuavoArmCtrlMode.AutoSwing)
    
    def set_external_control_arm_mode(self) -> bool:
        """
        External control the robot arm.
        Returns:
            bool: True if the arm is external controlled successfully, False otherwise.
        """
        return self._kuavo_core.change_robot_arm_ctrl_mode(KuavoArmCtrlMode.ExternalControl)

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
        return self._kuavo_core.arm_ik(left_pose, right_pose, left_elbow_pos_xyz, right_elbow_pos_xyz, arm_q0, params)

    def arm_fk(self, q: list) -> Tuple[KuavoPose, KuavoPose]:
        """Forward kinematics for the robot arm.
        
        Args:
            q (list): List of joint positions in radians.
            
        Returns:
            Tuple[KuavoPose, KuavoPose]: Tuple of poses for the robot left arm and right arm,
                or (None, None) if forward kinematics failed.
        """
        if len(q) != self._robot_info.arm_joint_dof:
            raise ValueError("Invalid position length. Expected {}, got {}".format(self._robot_info.arm_joint_dof, len(q)))
        
        result = self._kuavo_core.arm_fk(q)
        if result is None:
            return None, None
        return result
