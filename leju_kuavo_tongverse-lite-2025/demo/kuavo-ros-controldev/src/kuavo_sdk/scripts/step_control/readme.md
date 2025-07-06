# simStepControl.py
### 代码说明

#### 功能

该代码实现了一个 ROS 节点 `foot_pose_publisher` ，用于发布 `/humanoid_mpc_foot_pose_target_trajectories` 话题的指令。该指令使用 `footPoseTargetTrajectories` 消息类型来输出根据给定的身体姿态序列生成的机器人步态信息。

#### 程序输入参数

1. **pose_id**
  - 程序参数 `pose_id` 可输入1，2，3，4，5，6分别对应向前直行0.9米，向后直行0.9米，向左横移0.9米，向右横移0.42米，向左旋转90度，向右旋转90度。

#### 程序中函数的功能及参数说明

1. **get_foot_pose_traj_msg**
  - 描述: 该函数用于创建并返回一个 `footPoseTargetTrajectories` 消息对象，该对象包含了机器人的步态信息。
  - 输入参数：
    - `time_traj`: 时间轨迹列表，表示每一步的时间点。
    - `foot_idx_traj`: 脚索引轨迹列表，指示哪只脚在当前步态中移动。
    - `foot_traj`: 脚姿态轨迹列表，包含每一步的脚位置和姿态。
    - `torso_traj`: 躯干姿态轨迹列表，包含每一步的躯干位置和姿态。
  - 返回值： `footPoseTargetTrajectories` 消息对象。

2. **generate_steps**
  - 描述: 根据给定的躯干位置和偏航角，计算并返回左右脚的位置。
  - 输入参数：
    - `torso_pos`: 躯干位置的数组，包含x、y、z坐标。
    - `torso_yaw`: 躯干的偏航角，以弧度表示。
    - `foot_bias`: 脚的横向偏移量，用于计算左右脚的位置。
  - 返回值： 左脚和右脚的位置

3. **get_multiple_steps_msg**
  - 描述: 生成多步步态的消息对象，包含多个步态的时间、脚索引、脚姿态和躯干姿态。
  - 输入参数：
    - `body_poses`: 身体姿态的列表，每个姿态包含x、y、z坐标和偏航角。
    - `dt`: 每一步的时间间隔。
    - `is_left_first`: 布尔值，指示是否左脚先行。
    - `collision_check`: 布尔值，指示是否进行碰撞检测。
  - 返回值：  `footPoseTargetTrajectories` 消息对象。

#### 逻辑

1. **导入库**:

   - 导入 `rospy` 库以使用 ROS 的 Python 接口。
   - 导入 `numpy` 库以使用矩阵相关运算。
   - 导入 `say.py` 中用于碰撞检测的工具类 `RotatingRectangle`。
   - 导入 `footPose` `footPoseTargetTrajectories`话题相关的消息类型。

2. **定义发布函数**:

   - `humanoid_mpc_foot_pose_target_trajectories(footPoseTargetTrajectories)`:
     - 设置发布频率为 10 Hz。
     - 发布 `footPoseTargetTrajectories`，并打印打印时间轨迹，脚索引轨迹，脚姿态轨迹，躯干姿态轨迹。

3. **主程序**:

   - 初始化 ROS 节点，节点名称为 `robot_arm_ik_srv_node`。
   - 创建一个发布者，用于发布 `footPoseTargetTrajectories` 类型的消息到 `/humanoid_mpc_foot_pose_target_trajectories` 话题。
   - 设定参数及身体姿态序列
   - 调用 `get_multiple_steps_msg` 生成步态消息，并通过发布者发布该消息

### 总结

该代码实现了一个 ROS 节点，用于发布 `/humanoid_mpc_foot_pose_target_trajectories` 话题的控制指令。该指令使用 `footPoseTargetTrajectories` 消息类型来发布根据给定的身体姿态序列生成的机器人步态。
