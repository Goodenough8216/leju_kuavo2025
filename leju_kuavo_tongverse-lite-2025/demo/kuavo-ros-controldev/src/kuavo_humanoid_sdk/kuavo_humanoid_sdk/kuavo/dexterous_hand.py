#!/usr/bin/env python3
# coding: utf-8
from typing import Tuple
from kuavo_humanoid_sdk.interfaces.end_effector import EndEffector
from kuavo_humanoid_sdk.interfaces.data_types import EndEffectorSide, EndEffectorState
from kuavo_humanoid_sdk.kuavo.core.dex_hand_control import DexHandControl
from kuavo_humanoid_sdk.kuavo.core.ros.state import KuavoRobotStateCore

class DexterousHand(EndEffector):
    def __init__(self):
        joint_names = ['l_thumb', 'l_thumb_aux', 'l_index', 'l_middle', 'l_ring', 'l_pinky',
                       'r_thumb', 'r_thumb_aux', 'r_index', 'r_middle', 'r_ring', 'r_pinky',]
        super().__init__(joint_names=joint_names)
        self.dex_hand_control = DexHandControl()
        self._rb_state = KuavoRobotStateCore()

    def control(self, target_positions:list, target_velocities:list=None, target_torques:list=None)->bool:
        """Set the position of the hand.

        Args:
            target_positions (list): List of target positions for all joints, length must be 12 (6 joints for each hand),
                range => [0.0 ~ 100.0]
            target_velocities (list, optional): Not supported. Defaults to None.
            target_torques (list, optional): Not supported. Defaults to None.

        Returns:
            bool: True if control successful, False otherwise.

        Note:
            target_velocities and target_torques are not supported.
        """
        if len(target_positions) != len(self.joint_names):
            raise ValueError(f"Target positions must have the same length as joint names {len(target_positions)} != {len(self.joint_names)}")
        
        q = [max(0, min(100, pos if isinstance(pos, int) else int(pos))) for pos in target_positions]

        # control the hand
        return self.dex_hand_control.control(target_positions=q, side=EndEffectorSide.BOTH)

    def control_right(self, target_positions:list, target_velocities:list=None, target_torques:list=None)->bool:
        """Control the right dexterous hand.

        Args:
            target_positions (list): Target positions for right hand joints [0 ~ 100], length must be 6
            target_velocities (list, optional): Not supported. Defaults to None.
            target_torques (list, optional): Not supported. Defaults to None.

        Returns:
            bool: True if control successful, False otherwise.

        Raises:
            ValueError: If target positions length doesn't match joint count or values outside [0,100] range

        Note:
            target_velocities and target_torques are not supported.
        """
        if len(target_positions) != (len(self.joint_names)/2):
                raise ValueError(f"Target positions must have the same length as joint names {len(target_positions)} != {len(self.joint_names)/2}.")
        
        q = [max(0, min(100, pos if isinstance(pos, int) else int(pos))) for pos in target_positions]

        return self.dex_hand_control.control(target_positions=q, side=EndEffectorSide.RIGHT)

    def control_left(self, target_positions:list, target_velocities:list=None, target_torques:list=None)->bool:
        """Control the left dexterous hand.

        Args:
            target_positions (list): Target positions for left hand joints [0 ~ 100], length must be 6
            target_velocities (list, optional): Not supported. Defaults to None.
            target_torques (list, optional): Not supported. Defaults to None.

        Returns:
            bool: True if control successful, False otherwise.

        Raises:
            ValueError: If target positions length doesn't match joint count or values outside [0,100] range

        Note:
            target_velocities and target_torques are not supported.
        """
        if len(target_positions) != (len(self.joint_names)/2):
            raise ValueError(f"Target positions must have the same length as joint names {len(target_positions)} != {len(self.joint_names)/2}.")
        
        q = [max(0, min(100, pos if isinstance(pos, int) else int(pos))) for pos in target_positions]

        return self.dex_hand_control.control(target_positions=q, side=EndEffectorSide.LEFT)

    def open(self, side: EndEffectorSide=EndEffectorSide.BOTH)->bool:
        """Open the dexterous hand(s) by setting all joint positions to 0.

        Args:
            side (EndEffectorSide, optional): Which hand(s) to open. Defaults to EndEffectorSide.BOTH.
                Can be LEFT, RIGHT, or BOTH.

        Returns:
            bool: True if open command sent successfully, False otherwise.
        """
        zero_pos = [0]*len(self.joint_names)
        if side == EndEffectorSide.LEFT:
            return self.dex_hand_control.control(target_positions=zero_pos[:len(zero_pos)//2], side=EndEffectorSide.LEFT)
        elif side == EndEffectorSide.RIGHT:
            return self.dex_hand_control.control(target_positions=zero_pos[len(zero_pos)//2:], side=EndEffectorSide.RIGHT)
        else:
            return self.dex_hand_control.control(target_positions=zero_pos, side=EndEffectorSide.BOTH)     

    def make_gesture(self, l_gesture_name: str, r_gesture_name: str)->bool:
        """Make predefined gestures for both hands.

        Args:
            l_gesture_name (str): Name of gesture for left hand. None to skip left hand.
            r_gesture_name (str): Name of gesture for right hand. None to skip right hand.

        Returns:
            bool: True if gesture command sent successfully, False otherwise.

        Note:
            gestures e.g.: 'fist', 'ok', 'thumbs_up', '666'...
        """
        gesture = []
        if l_gesture_name is not None:
            gesture.append({'gesture_name':l_gesture_name, 'hand_side':EndEffectorSide.LEFT})
        if r_gesture_name is not None:
            gesture.append({'gesture_name':r_gesture_name, 'hand_side':EndEffectorSide.RIGHT})    
        self.dex_hand_control.make_gestures(gesture)

    def get_gesture_names(self)->list:
        """Get the names of all gestures.

        Returns:
            list: List of gesture names.
                e.g.: ['fist', 'ok', 'thumbs_up', '666', 'number_1', 'number_2', 'number_3', ... ],
                None if no gestures.
        """
        return self.dex_hand_control.get_gesture_names()
    
    def get_state(self)->Tuple[EndEffectorState, EndEffectorState]:
        # TODO(kuavo): Not implemented yet
        """Get the state of the dexterous hand.

        Returns:
            Tuple[EndEffectorState, EndEffectorState]: The state of the dexterous hand.
        """
        return self._rb_state.eef_state

    def get_position(self)->Tuple[float, float]:
        # TODO(kuavo): Not implemented yet
        """Get the position of the dexterous hand.

        Returns:
            Tuple[float, float]: The position of the dexterous hand.
        """
        return self._rb_state.eef_state.position
    
    def get_velocity(self)->Tuple[float, float]:
        # TODO(kuavo): Not implemented yet
        """Get the velocity of the dexterous hand.

        Returns:
            Tuple[float, float]: The velocity of the dexterous hand.
        """
        return self._rb_state.eef_state.velocity

    def get_effort(self)->Tuple[float, float]:
        # TODO(kuavo): Not implemented yet
        """Get the effort of the dexterous hand.

        Returns:
            Tuple[float, float]: The effort of the dexterous hand.

        Note:
            0 ~ 100 for each finger. Fraction of max motor current, absolute number.
            The max motor current is 600mA, in a word, 100.
        """
        return self._rb_state.eef_state.effort

    def get_grasping_state(self)->Tuple[EndEffectorState.GraspingState, EndEffectorState.GraspingState]:
        # TODO(kuavo): Not implemented yet
        """Get the grasping state of the dexterous hand.

        Returns:
            Tuple[EndEffectorState.GraspingState, EndEffectorState.GraspingState]: The grasping state of the dexterous hand.
        """
        return self._rb_state.eef_state.state
