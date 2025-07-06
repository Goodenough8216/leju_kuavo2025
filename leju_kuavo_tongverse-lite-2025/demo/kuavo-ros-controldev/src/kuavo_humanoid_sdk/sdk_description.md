# Kuavo Humanoid SDK

A comprehensive Python SDK for controlling Kuavo humanoid robots. This SDK provides interfaces for robot state management, arm and head control, and end-effector operations. It is designed to work with ROS (Robot Operating System) environments.

**Warning**: This SDK currently only supports **ROS1**. ROS2 support is not available.

**Warning**: **This SDK can only be used on the onboard NUC computer located in the robot's torso.** 


![Kuavo 4Pro Robot](https://kuavo.lejurobot.com/manual/assets/images/kuavo_4pro-cf84d43f1c370666c6e810d2807ae3e4.png)

## Features

- Robot State Management
  - IMU data (acceleration, angular velocity, euler angles)
  - Joint/motor states (position, velocity, torque)
  - Torso state (position, orientation, velocity)
  - Odometry information
  - End-effector states:
    - Gripper: position, velocity, torque, grasp status
    - Dexterous hand: position, velocity, torque
    - End-effector position and orientation
  - Motion states: stand, walk, step_control, trot

- Motion Control
  - Arm Control
    - Joint position control
    - End-effector 6D control via inverse kinematics
    - Forward kinematics (FK) for computing end-effector pose
    - Keyframe sequence control for complex motions
  - End-effector Control
    - Gripper control (position control with configurable velocity and torque)
    - Dexterous hand control
      - Position control
      - Pre-defined hand gestures (OK, 666, fist, etc.)
  - Head Control
    - Position control
  - Torso Control
    - Height control (squatting)
    - Forward/backward tilt control
  - Dynamic Motion Control
    - Stance
    - Trot
    - Walking (xy and yaw velocity control)
    - Stepping (gait switching)

- Robot Basic Information
  - Robot type (kuavo)
  - Robot version
  - End-effector type
  - Joint names
  - Total degrees of freedom (28)
  - Arm degrees of freedom (7 per arm)
  - Head degrees of freedom (2)
  - Leg degrees of freedom (12)

## Installation

To install Kuavo Humanoid SDK, you can use pip:
```bash
 pip install kuavo-humanoid-sdk
```

For development installation (editable mode), use:
```bash
 pip install -e .
```

## Package Information

You can check the package information using pip:
```bash
pip show kuavo-humanoid-sdk
```

## Quick Start

Here's a simple example to get started with Kuavo Humanoid SDK:

> **Warning**: Before running any code, make sure to start the robot first by executing either:
> - For simulation: `roslaunch humanoid_controllers load_kuavo_mujoco_sim.launch` (Example command)
> - For real robot: `roslaunch humanoid_controllers load_kuavo_real.launch` (Example command)
```python3
# Copyright (c) 2025 Leju Robotics. Licensed under the MIT License.
import time
from kuavo_humanoid_sdk import KuavoSDK, KuavoRobot

def main():
    if not KuavoSDK().Init():  # Init! !!! IMPORTANT !!!
        print("Init KuavoSDK failed, exit!")
        exit(1)
    robot = KuavoRobot()    
    
    """ arm reset """
    print("Switching to arm reset mode...")
    robot.arm_reset()
    
    """ stance """
    print("Switching to stance mode...")
    robot.stance()

    """ trot """
    print("Switching to trot mode...")
    robot.trot()
    
    """ walk forward """
    print("Starting forward walk...")
    duration = 4.0  # seconds
    speed = 0.3     # m/s
    start_time = time.time()
    while (time.time() - start_time < duration):
        robot.walk(linear_x=speed, linear_y=0.0, angular_z=0.0)
        time.sleep(0.1)  # Small sleep to prevent busy loop
    
if __name__ == "__main__":
    main()
```

## Examples

#### WARNING
Before running any code examples, make sure to start the robot first by executing either:

- For simulation: `roslaunch humanoid_controllers load_kuavo_mujoco_sim.launch` (Example command)
- For real robot: `roslaunch humanoid_controllers load_kuavo_real.launch` (Example command)

### Robot Info

Examples showing how to get basic robot information.

[https://gitee.com/leju-robot/kuavo-ros-opensource/tree/master/src/kuavo_humanoid_sdk/examples/robot_info_example.py](https://gitee.com/leju-robot/kuavo-ros-opensource/tree/master/src/kuavo_humanoid_sdk/examples/robot_info_example.py)

### Basic Robot Control

A basic example showing how to initialize the SDK and control the robot’s movement.

[https://gitee.com/leju-robot/kuavo-ros-opensource/tree/master/src/kuavo_humanoid_sdk/examples/motion_example.py](https://gitee.com/leju-robot/kuavo-ros-opensource/tree/master/src/kuavo_humanoid_sdk/examples/motion_example.py)

### End Effector Control

#### LejuClaw Gripper

Examples demonstrating how to control the LejuClaw gripper end effector, including position, velocity and torque control.

[https://gitee.com/leju-robot/kuavo-ros-opensource/tree/master/src/kuavo_humanoid_sdk/examples/lejuclaw_example.py](https://gitee.com/leju-robot/kuavo-ros-opensource/tree/master/src/kuavo_humanoid_sdk/examples/lejuclaw_example.py)

#### QiangNao DexHand

Examples showing how to control the QiangNao DexHand, a dexterous robotic hand with multiple degrees of freedom for complex manipulation tasks.

[https://gitee.com/leju-robot/kuavo-ros-opensource/tree/master/src/kuavo_humanoid_sdk/examples/dexhand_example.py](https://gitee.com/leju-robot/kuavo-ros-opensource/tree/master/src/kuavo_humanoid_sdk/examples/dexhand_example.py)

### Arm Control

Examples showing arm trajectory control and target pose control.

[https://gitee.com/leju-robot/kuavo-ros-opensource/tree/master/src/kuavo_humanoid_sdk/examples/ctrl_arm_example.py](https://gitee.com/leju-robot/kuavo-ros-opensource/tree/master/src/kuavo_humanoid_sdk/examples/ctrl_arm_example.py)

### Forward and Inverse Kinematics

Examples demonstrating how to use forward kinematics (FK) to compute end-effector positions from joint angles, and inverse kinematics (IK) to calculate joint angles needed to achieve desired end-effector poses.

[https://gitee.com/leju-robot/kuavo-ros-opensource/tree/master/src/kuavo_humanoid_sdk/examples/arm_ik_example.py](https://gitee.com/leju-robot/kuavo-ros-opensource/tree/master/src/kuavo_humanoid_sdk/examples/arm_ik_example.py)

### Head Control

Examples showing how to control the robot’s head movements, including nodding (pitch) and shaking (yaw) motions.

[https://gitee.com/leju-robot/kuavo-ros-opensource/tree/master/src/kuavo_humanoid_sdk/examples/ctrl_head_example.py](https://gitee.com/leju-robot/kuavo-ros-opensource/tree/master/src/kuavo_humanoid_sdk/examples/ctrl_head_example.py)

### Step-by-Step Control

Examples showing how to control the robot’s movements step by step, including individual foot placement and trajectory control.

[https://gitee.com/leju-robot/kuavo-ros-opensource/tree/master/src/kuavo_humanoid_sdk/examples/step_control_example.py](https://gitee.com/leju-robot/kuavo-ros-opensource/tree/master/src/kuavo_humanoid_sdk/examples/step_control_example.py)


## Docs

https://gitee.com/leju-robot/kuavo-ros-opensource/tree/master/src/kuavo_humanoid_sdk/docs/markdown/index.md

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact & Support

For any questions, support, or bug reports, please contact:
- Email: edu@lejurobot.com
- Website: https://gitee.com/leju-robot/kuavo-ros-opensource/
- Source Code: https://gitee.com/leju-robot/kuavo-ros-opensource/
- Issue Tracker: https://gitee.com/leju-robot/kuavo-ros-opensource/issues
