#!/usr/bin/env python3
# coding: utf-8
import time
from typing import Tuple
from kuavo_humanoid_sdk.interfaces.end_effector import EndEffector
from kuavo_humanoid_sdk.interfaces.data_types import EndEffectorSide, EndEffectorState
from kuavo_humanoid_sdk.kuavo.core.leju_claw_control import LejuClawControl
from kuavo_humanoid_sdk.kuavo.core.ros.state import KuavoRobotStateCore

class LejuClaw(EndEffector):
    def __init__(self):
        super().__init__(joint_names=['left_claw', 'right_claw'])
        self.leju_claw_control = LejuClawControl()
        self._rb_state = KuavoRobotStateCore()

    def control(self, target_positions: list, target_velocities:list=None, target_torques: list=None)->bool:
        """Control the claws to grip.

        Args:
            target_positions (list): The target positions of the claws.
            target_velocities (list, optional): The target velocities of the claws. If None, default value [90, 90] will be used.
            target_torques (list, optional): The target torques of the claws. If None, default value [1.0, 1.0] will be used.

        Note:
            The target_positions, target_velocities,  target_torques must be a list of length 2.
            After calling this function, you can call wait_for_finish() to wait until the claws reach the target position.

        Warning:
            If the claws are still in motion from a previous command, this request may be dropped.

        Returns:
            bool: True if the claws are successfully gripped, False otherwise
        """
        if target_positions is None:
            raise ValueError("Target positions must be provided.")
        
        target_positions = [max(0.0, min(100.0, pos)) for pos in target_positions]
        if target_velocities is None:
            target_velocities = [90, 90]
        else:
           target_velocities = [max(0.0, min(100.0, vel)) for vel in target_velocities]
        
        if target_torques is None:
            target_torques = [1.0, 1.0]
        else:
            target_torques = [max(0.0, min(10.0, torque)) for torque in target_torques]
            
        return self.leju_claw_control.control(target_positions, target_velocities, target_torques, EndEffectorSide.BOTH)

    def control_left(self, target_positions:list, target_velocities:list=None, target_torques:list=None)->bool:
        """Control the left claw to grip.

        Args:
            target_positions (list): The target position of the left claw.
            target_velocities (list, optional): The target velocity of the left claw. If None, default value 90 will be used.
            target_torques (list, optional): The target torque of the left claw. If None, default value 1.0 will be used.

        Note:
            The target_positions, target_velocities, target_torques must be a list of length 1
            After calling this function, you can call wait_for_finish() to wait until the claws reach the target position.

        Warning:
            If the claws are still in motion from a previous command, this request may be dropped.

        Returns:
            bool: True if the claw is successfully gripped, False otherwise.
        """
        if target_positions is None:
            raise ValueError("Target positions must be provided.")
        
        for data in [target_positions, target_velocities, target_torques]:
            if data is not None and len(data) != 1:
                raise ValueError("Target positions must be a list of length 1.")
        
        q = max(0.0, min(100.0, target_positions[0]))

        if target_velocities is not None:
            v = max(0.0, min(100.0, target_velocities[0]))
        else:
            v = 90

        if target_torques is not None:
            tau = max(0.0, min(10.0, target_torques[0]))
        else:
            tau = 1.0

        return self.leju_claw_control.control([q], [v], [tau], EndEffectorSide.LEFT)

    def control_right(self, target_positions:list, target_velocities:list=None, target_torques:list=None)->bool:
        """Control the right claw to grip.

        Args:
            target_positions (list): The target position of the right claw.
            target_velocities (list, optional): The target velocity of the right claw. If None, default value 90 will be used.
            target_torques (list, optional): The target torque of the right claw. If None, default value 1.0 will be used.

        Returns:
            bool: True if the claw is successfully gripped, False otherwise.
        
        Note:
            The target_positions, target_velocities, target_torques must be a list of length 1
            After calling this function, you can call wait_for_finish() to wait until the claws reach the target position.

        Warning:
            If the claws are still in motion from a previous command, this request may be dropped.    
        """
        if target_positions is None:
            raise ValueError("Target positions must be provided.")
        
        for data in [target_positions, target_velocities, target_torques]:
            if data is not None and len(data) != 1:
                raise ValueError("Target positions must be a list of length 1.")
        
        q = max(0.0, min(100.0, target_positions[0]))

        if target_velocities is not None:
            v = max(0.0, min(100.0, target_velocities[0]))
        else:
            v = 90

        if target_torques is not None:
            tau = max(0.0, min(10.0, target_torques[0]))
        else:
            tau = 1.0

        return self.leju_claw_control.control([q], [v], [tau], EndEffectorSide.RIGHT)
    
    def open(self, side:EndEffectorSide=EndEffectorSide.BOTH)->bool:
        """Control the claws to release/open.

        Note:
            Control the claws to open.
            After calling this function, you can call wait_for_finish() to wait until the claws reach the target position.

        Args:
            side (EndEffectorSide, optional): The side to control. Defaults to EndEffectorSide.BOTH.

        Returns:
            bool: True if the claw is successfully released, False otherwise.
        """
        return self.leju_claw_control.release(side)

    def close(self, side:EndEffectorSide=EndEffectorSide.BOTH)->bool:
        """Control the claws to close/grip.

        Note:
            Control the claws to close.
            After calling this function, you can call wait_for_finish() to wait until the claws reach the target position.

        Args:
            side (EndEffectorSide, optional): The side to control. Defaults to EndEffectorSide.BOTH.

        Returns:
            bool: True if the claw is successfully gripped, False otherwise.
        """
        return self.leju_claw_control.control(position=[100, 100], velocity=[90, 90], torque=[1.0, 1.0], side=side)

    def get_grasping_state(self)->Tuple[EndEffectorState.GraspingState, EndEffectorState.GraspingState]:
        """Get the grasping state of the claws.

        Returns:
            Tuple[EndEffectorState.GraspingState, EndEffectorState.GraspingState]: The grasping state of the claws.
        """
        return self._rb_state.eef_state.state
    
    def get_position(self)->Tuple[float, float]:
        """Get the position of the claws.

        Returns:
            Tuple[float, float]: The position of the claws, range [0.0, 100.0].
        """
        return self._rb_state.eef_state.position
    
    def get_velocity(self)->Tuple[float, float]:
        """Get the velocity of the claws.

        Returns:
            Tuple[float, float]: The velocity of the claws.
        """
        return self._rb_state.eef_state.velocity
    
    def get_effort(self)->Tuple[float, float]:
        """Get the effort of the claws.

        Returns:
            Tuple[float, float]: The effort of the claws.
        """
        return self._rb_state.eef_state.effort

    def get_state(self)->Tuple[EndEffectorState, EndEffectorState]:
        """Get the state of the claws.

        Returns:
            Tuple[EndEffectorState, EndEffectorState]: The state of the claws.
                - position: The position of the claws, range [0.0, 100.0].
                - velocity: The velocity of the claws.
                - effort: The effort of the claws.
                - state: The grasping state of the claws.
        """
        return self._rb_state.eef_state

    def wait_for_finish(self, side: EndEffectorSide=EndEffectorSide.BOTH, timeout:float=2.5):
        """Wait for the claw motion to finish.

        Args:
            side (EndEffectorSide, optional): The side of the claw to wait for. Defaults to EndEffectorSide.BOTH.
            timeout (float, optional): The timeout duration in seconds. Defaults to 2.5.

        Returns:
            bool: True if motion completed before timeout, False otherwise.
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            if side == EndEffectorSide.BOTH:
                if self._rb_state.eef_state[0].state >= EndEffectorState.GraspingState.REACHED and \
                   self._rb_state.eef_state[1].state >= EndEffectorState.GraspingState.REACHED:
                    return True
            elif side == EndEffectorSide.LEFT:
                if self._rb_state.eef_state[0].state >= EndEffectorState.GraspingState.REACHED:
                    return True
            elif side == EndEffectorSide.RIGHT:
                if self._rb_state.eef_state[1].state >= EndEffectorState.GraspingState.REACHED:
                    return True
            time.sleep(0.1)
        return False