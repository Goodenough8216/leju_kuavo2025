<a id="examples"></a>

# Examples

#### WARNING
Before running any code examples, make sure to start the robot first by executing either:

- For simulation: `roslaunch humanoid_controllers load_kuavo_mujoco_sim.launch` (Example command)
- For real robot: `roslaunch humanoid_controllers load_kuavo_real.launch` (Example command)

## Robot Info

Examples showing how to get basic robot information.

[https://gitee.com/leju-robot/kuavo-ros-opensource/tree/master/src/kuavo_humanoid_sdk/examples/robot_info_example.py](https://gitee.com/leju-robot/kuavo-ros-opensource/tree/master/src/kuavo_humanoid_sdk/examples/robot_info_example.py)

## Basic Robot Control

A basic example showing how to initialize the SDK and control the robot’s movement.

[https://gitee.com/leju-robot/kuavo-ros-opensource/tree/master/src/kuavo_humanoid_sdk/examples/motion_example.py](https://gitee.com/leju-robot/kuavo-ros-opensource/tree/master/src/kuavo_humanoid_sdk/examples/motion_example.py)

## End Effector Control

### LejuClaw Gripper

Examples demonstrating how to control the LejuClaw gripper end effector, including position, velocity and torque control.

[https://gitee.com/leju-robot/kuavo-ros-opensource/tree/master/src/kuavo_humanoid_sdk/examples/lejuclaw_example.py](https://gitee.com/leju-robot/kuavo-ros-opensource/tree/master/src/kuavo_humanoid_sdk/examples/lejuclaw_example.py)

### QiangNao DexHand

Examples showing how to control the QiangNao DexHand, a dexterous robotic hand with multiple degrees of freedom for complex manipulation tasks.

[https://gitee.com/leju-robot/kuavo-ros-opensource/tree/master/src/kuavo_humanoid_sdk/examples/dexhand_example.py](https://gitee.com/leju-robot/kuavo-ros-opensource/tree/master/src/kuavo_humanoid_sdk/examples/dexhand_example.py)

## Arm Control

Examples showing arm trajectory control and target pose control.

[https://gitee.com/leju-robot/kuavo-ros-opensource/tree/master/src/kuavo_humanoid_sdk/examples/ctrl_arm_example.py](https://gitee.com/leju-robot/kuavo-ros-opensource/tree/master/src/kuavo_humanoid_sdk/examples/ctrl_arm_example.py)

## Forward and Inverse Kinematics

Examples demonstrating how to use forward kinematics (FK) to compute end-effector positions from joint angles, and inverse kinematics (IK) to calculate joint angles needed to achieve desired end-effector poses.

[https://gitee.com/leju-robot/kuavo-ros-opensource/tree/master/src/kuavo_humanoid_sdk/examples/arm_ik_example.py](https://gitee.com/leju-robot/kuavo-ros-opensource/tree/master/src/kuavo_humanoid_sdk/examples/arm_ik_example.py)

## Head Control

Examples showing how to control the robot’s head movements, including nodding (pitch) and shaking (yaw) motions.

[https://gitee.com/leju-robot/kuavo-ros-opensource/tree/master/src/kuavo_humanoid_sdk/examples/ctrl_head_example.py](https://gitee.com/leju-robot/kuavo-ros-opensource/tree/master/src/kuavo_humanoid_sdk/examples/ctrl_head_example.py)

## Step-by-Step Control

Examples showing how to control the robot’s movements step by step, including individual foot placement and trajectory control.

[https://gitee.com/leju-robot/kuavo-ros-opensource/tree/master/src/kuavo_humanoid_sdk/examples/step_control_example.py](https://gitee.com/leju-robot/kuavo-ros-opensource/tree/master/src/kuavo_humanoid_sdk/examples/step_control_example.py)
