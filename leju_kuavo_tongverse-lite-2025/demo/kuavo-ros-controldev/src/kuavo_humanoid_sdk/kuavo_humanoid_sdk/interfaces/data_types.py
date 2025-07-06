from typing import Tuple
from enum import Enum
from dataclasses import dataclass

@dataclass
class KuavoJointData:
    """Data class representing joint states of the robot.

    Args:
        position (list): List of joint positions (angles) in radians
        velocity (list): List of joint velocities in radians/second
        torque (list): List of joint torques/efforts in Newton-meters or Amperes
        acceleration (list): List of joint accelerations in radians/second^2
    """
    position: list
    velocity: list
    torque: list
    acceleration:list

@dataclass
class KuavoImuData:
    """Data class representing IMU (Inertial Measurement Unit) data from the robot.

    Args:
        gyro (Tuple[float, float, float]): Angular velocity around x, y, z axes in rad/s
        acc (Tuple[float, float, float]): Linear acceleration in x, y, z axes in m/s^2
        free_acc (Tuple[float, float, float]): Free acceleration (gravity compensated) in x, y, z axes in m/s^2
        quat (Tuple[float, float, float, float]): Orientation quaternion (x, y, z, w)
    """
    gyro : Tuple[float, float, float]
    acc : Tuple[float, float, float]
    free_acc: Tuple[float, float, float]
    quat: Tuple[float, float, float, float]

@dataclass
class KuavoOdometry:
    """Data class representing odometry data from the robot.

    Args:
        position (Tuple[float, float, float]): Robot position (x, y, z) in world coordinates in meters
        orientation (Tuple[float, float, float, float]): Robot orientation as quaternion (x, y, z, w)
        linear (Tuple[float, float, float]): Linear velocity (x, y, z) in world coordinates in m/s
        angular (Tuple[float, float, float]): Angular velocity (x, y, z) in world coordinates in rad/s
    """
    position: Tuple[float, float, float]
    orientation: Tuple[float, float, float, float]
    linear: Tuple[float, float, float]
    angular: Tuple[float, float, float]

class KuavoArmCtrlMode(Enum):
    """Enum class representing the control modes for the Kuavo robot arm.

    Attributes:
        ArmFixed: The robot arm is fixed in position (value: 0)
        AutoSwing: The robot arm is in automatic swinging mode (value: 1)
        ExternalControl: The robot arm is controlled by external commands (value: 2)
    """
    ArmFixed = 0
    AutoSwing = 1 
    ExternalControl = 2


@dataclass
class EndEffectorState:
    """Data class representing the state of the end effector.

    Args:
        position (float): Position of the end effector, [0, 100]
        velocity (float): ...
        effort (float): ...
    """
    position: float
    velocity: float
    effort: float
    class GraspingState(Enum):
        """Enum class representing the grasping states of the end effector.

        Attributes:
            ERROR: Error state (value: -1)
            UNKNOWN: Unknown state (value: 0)
            REACHED: Target position reached (value: 1)
            MOVING: Moving to target position (value: 2)
            GRABBED: Object successfully grasped (value: 3)
        """
        ERROR = -1
        UNKNOWN = 0
        MOVING = 1
        REACHED = 2
        GRABBED = 3

    state: GraspingState # gripper grasping states

class EndEffectorSide(Enum):
    """Enum class representing the sides of the end effector.

    Attributes:
        LEFT: The left side of the end effector (value: 'left')
        RIGHT: The right side of the end effector (value: 'right')
        BOTH: Both sides of the end effector (value: 'both')
    """
    LEFT = 'left'
    RIGHT = 'right'
    BOTH = 'both'

@dataclass
class KuavoPose:
    """Data class representing the pose of the robot."""
    position: Tuple[float, float, float] # x, y, z
    orientation: Tuple[float, float, float, float] # x, y, z, w

@dataclass
class KuavoIKParams:
    """Data class representing the parameters for the IK node."""
    # snopt params
    major_optimality_tol: float = 1e-3
    major_feasibility_tol: float = 1e-3  
    minor_feasibility_tol: float = 1e-3
    major_iterations_limit: float = 100
    # constraint and cost params
    oritation_constraint_tol: float = 1e-3
    pos_constraint_tol: float = 1e-3 # 0.001m, work when pos_cost_weight==0.0
    pos_cost_weight: float = 0.0 # If U need high accuracy, set this to 0.0 !!!