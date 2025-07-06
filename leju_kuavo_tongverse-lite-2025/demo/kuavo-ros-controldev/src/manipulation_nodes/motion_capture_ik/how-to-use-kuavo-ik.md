# 如何使用 kuavo IK 逆解模块

## 编译

```bash
catkin build motion_capture_ik
```

## 启动

### 启动参数

以下是 IK 模块启动时可选的参数:

- `visualize` : 是否在 Rviz 中可视化, 默认值为 `false`
- `robot_version` : 机器人版本号, 默认值为 `ROBOT_VERSION` 环境变量
- `control_hand_side` : 用于设置控制对应的左右手臂, 0 为只控制左手, 1 为只控制右手, 2 为控制双手, 默认值为 2
- `eef_z_bias` : 末端执行器 z 坐标的偏置, 默认值为 `-0.17`, 单位 m
- `model_path` : IK 求解时加载的 URDF 文件路径, 一般无需手动设置, 默认会从 kuavo_assets 包下根据 `robot_version` 搜索设置
- `model_path_vis` : 在 RVIZ 中可视化时加载的 URDF 文件, 一般无需手动设置, 默认会从 kuavo_assets 包下根据 `robot_version` 搜索设置
- `print_ik_info` : 是否在终端打印求解的调试信息, 默认值为 `false`

### 启动示例

版本号为 `43` 的机器人, 且开启可视化:

```bash
source devel/setup.bash
roslaunch motion_capture_ik ik_node.launch robot_version:=43 visualize:=1 print_ik_info:=false
```

只求解右臂且不可视化 (robot_version 为环境变量中 `ROBOT_VERSION`):

```bash
source devel/setup.bash
roslaunch motion_capture_ik ik_node.launch control_hand_side:=1
```

## ROS 接口

### 话题

#### 订阅的话题

##### `/ik/two_arm_hand_pose_cmd`

话题描述: 手臂 IK 求解话题

消息类型: `motion_capture_ik/twoArmHandPoseCmd`

| 字段                  | 类型             | 描述                                                |
| ------------------- | -------------- | ------------------------------------------------- |
| hand_poses          | twoArmHandPose | 双手信息: 末端位置, 末端四元数, 手肘位置等                           |
| use_custom_ik_param | bool           | 是否使用自定义的 IK 参数, 设置为 true 时 会使用消息中的 ik_param 值用于求解 |
| joint_angles_as_q0  | bool           | 是否使用 hand_poses 中的 joint_angles 作为求解时的 q0         |
| ik_param            | ikSolveParam   | 自定义的 IK 求解参数                                       |

对于 `ik_param` 字段详细描述如下:

| 字段                       | 类型      | 描述                                           |
| ------------------------ | ------- | -------------------------------------------- |
| major_optimality_tol     | float64 | snopt 参数, 即主要迭代中优化性的容差, 该参数决定最优性条件的满足程度      |
| major_feasibility_tol    | float64 | snopt 参数, 即主要迭代中的可行性容差, 用于控制非线性约束            |
| minor_feasibility_tol    | float64 | snopt 参数, 次要迭代中的可行性容差，主要用于线性化后的模型            |
| major_iterations_limit   | float64 | snopt 参数, 主要迭代的最大次数                          |
| oritation_constraint_tol | float64 | 姿态约束参数                                      |
| pos_constraint_tol       | float64 | 位置约束参数, **该参数只会在 pos_cost_weight 大于 0.0 时生效** |
| pos_cost_weight          | float64 | 位置成本参数, **当设置成 0.0 时求解精度要求最高**               |

如果您期望更高精度的 IK 求解, 可将 `pos_cost_weight` 设置成 `0.0`, 但与此同时也会降低求解成功的概率.

对于 `hand_poses` 字段详细描述如下:

| 字段            | 类型         | 描述                                               |
| ------------- | ---------- | ------------------------------------------------ |
| pos_xyz       | float64[3] | 末端期望的位置, 单位 m                                     |
| quat_xyzw     | float64[4] | 末端期望的姿态                                          |
| elbow_pos_xyz | float64[3] | 手肘期望的位置, 全设置为 0.0 时忽略该参数                         |
| joint_angles  | float64[7] | 如果 joint_angles_as_q0 为 true, 则使用该值作为求解时的 q0, 单位弧度 |

#### 发布的话题

##### `/ik/result`

话题描述: 发布 IK 求解的结果

消息类型: `motion_capture_ik/twoArmHandPose`

发布 IK 结果中左右手的结果,

| 字段            | 类型         | 描述          |
| ------------- | ---------- | ----------- |
| pos_xyz       | float64[3] | 末端位置, 单位 m   |
| quat_xyzw     | float64[4] | 末端姿态        |
| elbow_pos_xyz | float64[3] | 手肘位置, 单位 m      |
| joint_angles  | float64[7] | 手臂关节值, 单位弧度 |

##### `/ik/debug/time_cost`

话题描述: 可忽略, 主要调试输出求解耗时信息, 单位毫秒

消息类型: `std_msgs/Float32MultiArray`

- `data[0]` : 循环耗时, 单位毫秒
- `data[1]` : 解算耗时, 单位毫秒

##### `/kuavo_arm_traj`

话题描述: 当通过 `/ik/two_arm_hand_pose_cmd` 话题调用时, 有结果便会输出, 单位弧度

消息类型: `sensor_msgs/JointState`

### 服务

##### `/ik/two_arm_hand_pose_cmd_srv`

话题描述: IK 逆解服务

消息类型: `motion_capture_ik/twoArmHandPoseCmdSrv`

请求参数:

| 字段                       | 类型                | 描述                                    |
| ------------------------ | ----------------- | ------------------------------------- |
| twoArmHandPoseCmdRequest | twoArmHandPoseCmd | 详情请参考 `/ik/two_arm_hand_pose_cmd` 话题的内容 |

返回结果:

| 字段         | 类型             | 描述                    |
| ---------- | -------------- | --------------------- |
| success    | bool           | 是否成功                  |
| with_torso | bool           | 是否包含躯干                |
| q_arm      | float64[]      | 手臂关节值, 单位弧度           |
| q_torso    | float64[]      | 躯干的关节值                |
| time_cost  | float64        | 求解耗时, 单位 ms           |
| hand_poses | twoArmHandPose | IK 求解结果, 具体内容见上述同类型消息 |

##### `/ik/fk_srv`

话题描述: FK 正解服务

消息类型: `motion_capture_ik/fkSrv`

| 字段         | 类型             | 描述                                                         |
| ---------- | -------------- | ---------------------------------------------------------- |
| q          | float64[]      | 长度为 14 ，内容为手臂关节的角度, 单位弧度 |
| success    | bool           | 返回参数, 是否成功                                                 |
| hand_poses | twoArmHandPose | 返回参数, 正解结果, 具体内容见上述同类型消息                                   |

## 使用示例

### 使用 IK 求解服务

以下代码展示了如何调用 IK 服务, 并打印返回的结果, 可通过运行如下命令调用:

```bash
source devel/setup.bash
rosrun motion_capture_ik example_ik_srv.py
```

代码:

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import time
from motion_capture_ik.msg import twoArmHandPoseCmd, ikSolveParam
from motion_capture_ik.srv import twoArmHandPoseCmdSrv
from sensor_msgs.msg import JointState

# decide use custom ik param or not
use_custom_ik_param = True
# joint angles as initial guess for ik
joint_angles_as_q0 = False # True for custom initial guess, False for default initial guess
# ik solver param
ik_solve_param = ikSolveParam()
# snopt params
ik_solve_param.major_optimality_tol = 1e-3
ik_solve_param.major_feasibility_tol = 1e-3
ik_solve_param.minor_feasibility_tol = 1e-3
ik_solve_param.major_iterations_limit = 100
# constraint and cost params
ik_solve_param.oritation_constraint_tol= 1e-3
ik_solve_param.pos_constraint_tol = 1e-3 # 0.001m, work when pos_cost_weight==0.0
ik_solve_param.pos_cost_weight = 0.0 # If U need high accuracy, set this to 0.0 !!!

def call_ik_srv(eef_pose_msg):
    rospy.wait_for_service('/ik/two_arm_hand_pose_cmd_srv')
    try:
        ik_srv = rospy.ServiceProxy('/ik/two_arm_hand_pose_cmd_srv', twoArmHandPoseCmdSrv)
        res = ik_srv(eef_pose_msg)
        return res
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)
        return False, []

if __name__ == "__main__":
    rospy.init_node("example_ik_srv_node", anonymous=True)
    time.sleep(1.5)

    eef_pose_msg = twoArmHandPoseCmd()
    eef_pose_msg.ik_param = ik_solve_param
    eef_pose_msg.use_custom_ik_param = use_custom_ik_param        # # True for custom ik param, False for default ik param  
    eef_pose_msg.joint_angles_as_q0 = joint_angles_as_q0

    # joint_angles_as_q0 为 False 时，这两个参数不会被使用
    eef_pose_msg.hand_poses.left_pose.joint_angles = np.zeros(7)    # rads
    eef_pose_msg.hand_poses.right_pose.joint_angles = np.zeros(7)   # rads

    # 设置左手末端执行器的位置和姿态
    eef_pose_msg.hand_poses.left_pose.pos_xyz =  np.array([0.45,0.25,0.11988012])
    eef_pose_msg.hand_poses.left_pose.quat_xyzw = [0.0,-0.70682518,0.0,0.70738827] # 四元数
    eef_pose_msg.hand_poses.left_pose.elbow_pos_xyz = np.zeros(3) # 设置成 0.0 时,不会被使用

    # 设置右手末端执行器的位置和姿态
    eef_pose_msg.hand_poses.right_pose.pos_xyz =  np.array([0.45,-0.25,0.11988012])
    eef_pose_msg.hand_poses.right_pose.quat_xyzw = [0.0,-0.70682518,0.0,0.70738827] # 四元数
    eef_pose_msg.hand_poses.right_pose.elbow_pos_xyz = np.zeros(3)  # 设置成 0.0 时,不会被使用

    # 调用 IK 服务
    res = call_ik_srv(eef_pose_msg) # 服务
    if(res.success):
        l_pos = res.hand_poses.left_pose.pos_xyz
        l_pos_error = np.linalg.norm(l_pos - eef_pose_msg.hand_poses.left_pose.pos_xyz)
        r_pos = res.hand_poses.right_pose.pos_xyz
        r_pos_error = np.linalg.norm(r_pos - eef_pose_msg.hand_poses.right_pose.pos_xyz)
        
        # 打印结果
        print(f"time_cost: {res.time_cost:.2f} ms. left_pos_error: {1e3*l_pos_error:.2f} mm, right_pos_error: {1e3*r_pos_error:.2f} mm")
        print(f"left_joint_angles: {res.hand_poses.left_pose.joint_angles}")
        print(f"right_joint_angles: {res.hand_poses.right_pose.joint_angles}")
    else:
        print(f"ik success: {res.success}")
```

### 使用 IK 求解话题

以下代码展示了如何通过 IK 话题使用 IK 功能, 并打印返回的结果, 可通过运行如下命令调用:

```bash
source devel/setup.bash
rosrun motion_capture_ik example_ik.py
```

代码:

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy np
import time
from motion_capture_ik.msg import twoArmHandPoseCmd, ikSolveParam
from motion_capture_ik.msg import twoArmHandPose

# decide use custom ik param or not
use_custom_ik_param = True
# joint angles as initial guess for ik
joint_angles_as_q0 = False # True for custom initial guess, False for default initial guess
# ik solver param
ik_solve_param = ikSolveParam()
# snopt params
ik_solve_param.major_optimality_tol = 1e-3
ik_solve_param.major_feasibility_tol = 1e-3
ik_solve_param.minor_feasibility_tol = 1e-3
ik_solve_param.major_iterations_limit = 100
# constraint and cost params
ik_solve_param.oritation_constraint_tol= 1e-3
ik_solve_param.pos_constraint_tol = 1e-3 # 0.001m, work when pos_cost_weight==0.0
ik_solve_param.pos_cost_weight = 0.0 # If U need high accuracy, set this to 0.0 !!!

def kuavo_ik_result_callback(msg):
    print("************ik result***********")
    print(f"left_joint_angles: {msg.left_pose.joint_angles}")
    print(f"right_joint_angles: {msg.right_pose.joint_angles}")

if __name__ == "__main__":
    rospy.init_node("example_ik_node", anonymous=True)
    pub = rospy.Publisher('/ik/two_arm_hand_pose_cmd', twoArmHandPoseCmd, queue_size=10)
    ik_result_sub = rospy.Subscriber("/ik/result", twoArmHandPose, kuavo_ik_result_callback)
    time.sleep(1.0)

    eef_pose_msg = twoArmHandPoseCmd()
    eef_pose_msg.ik_param = ik_solve_param
    eef_pose_msg.use_custom_ik_param = use_custom_ik_param        # # True for custom ik param, False for default ik param  
    eef_pose_msg.joint_angles_as_q0 = joint_angles_as_q0

    # joint_angles_as_q0 为 False 时，这两个参数不会被使用
    eef_pose_msg.hand_poses.left_pose.joint_angles = np.zeros(7)    # rads
    eef_pose_msg.hand_poses.right_pose.joint_angles = np.zeros(7)   # rads

    # 设置左手末端执行器的位置和姿态
    eef_pose_msg.hand_poses.left_pose.pos_xyz =  np.array([0.45,0.25,0.11988012])
    eef_pose_msg.hand_poses.left_pose.quat_xyzw = [0.0,-0.70682518,0.0,0.70738827] # 四元数
    eef_pose_msg.hand_poses.left_pose.elbow_pos_xyz = np.zeros(3) # 设置成 0.0 时,不会被使用

    # 设置右手末端执行器的位置和姿态
    eef_pose_msg.hand_poses.right_pose.pos_xyz =  np.array([0.45,-0.25,0.11988012])
    eef_pose_msg.hand_poses.right_pose.quat_xyzw = [0.0,-0.70682518,0.0,0.70738827] # 四元数
    eef_pose_msg.hand_poses.right_pose.elbow_pos_xyz = np.zeros(3)  # 设置成 0.0 时,不会被使用

    print("publish ik cmd")
    pub.publish(eef_pose_msg)

    rospy.spin()
```

### 使用 FK 求解服务

以下代码展示了如何调用 FK 服务, 并打印返回的结果, 可通过运行如下命令调用:

```bash
source devel/setup.bash
rosrun motion_capture_ik example_fk_srv.py
```

代码:

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from motion_capture_ik.srv import fkSrv

import numpy as np

def fk_srv_client(joint_angles):
    rospy.wait_for_service('/ik/fk_srv')
    try:
        fk_srv = rospy.ServiceProxy('/ik/fk_srv', fkSrv)
        fk_result = fk_srv(joint_angles)
        print("FK result:", fk_result.success)
        return fk_result.hand_poses
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)

if __name__ == "__main__":
    rospy.init_node("example_fk_srv_node", anonymous=True)
    # 单位：弧度
    joint_angles = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.38, -1.39, -0.29, -0.43, 0.0, -0.17, 0.0]

    # 调用 FK 正解服务
    hand_poses = fk_srv_client(joint_angles)
    if hand_poses is not None:
        print("left eef position:", hand_poses.left_pose.pos_xyz)
        print("\nright eef position: ", hand_poses.right_pose.pos_xyz)
    else:
        print("No hand poses returned")
```
