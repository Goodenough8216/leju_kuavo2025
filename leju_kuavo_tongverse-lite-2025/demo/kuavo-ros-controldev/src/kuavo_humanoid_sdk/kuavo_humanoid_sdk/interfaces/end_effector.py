from abc import ABC, abstractmethod
from typing import Tuple
from kuavo_humanoid_sdk.interfaces.data_types import EndEffectorSide, EndEffectorState
    
class EndEffector(ABC):
    def __init__(self, joint_names: list):
        self.joint_names = joint_names
        
    @abstractmethod
    def control(self, target_positions:list, target_velocities:list, target_torques:list)->bool:
        pass

    @abstractmethod
    def control_right(self, target_positions:list, target_velocities:list, target_torques:list)->bool:
        pass

    @abstractmethod
    def control_left(self, target_positions:list, target_velocities:list, target_torques:list)->bool:
        pass    

    @abstractmethod
    def open(self, side:EndEffectorSide)->bool:
        pass

    @abstractmethod
    def get_state(self)->Tuple[EndEffectorState, EndEffectorState]:
        pass

    @abstractmethod
    def get_position(self)->Tuple[float, float]:
        pass

    @abstractmethod
    def get_velocity(self)->Tuple[float, float]:
        pass

    @abstractmethod
    def get_effort(self)->Tuple[float, float]:
        pass

    @abstractmethod
    def get_grasping_state(self)->Tuple[EndEffectorState.GraspingState, EndEffectorState.GraspingState]:
        pass