# robot_arm_fk_ik.py
### 代码说明

#### 功能

该代码通过 ROS 服务和话题，实现了手臂控制模式的更改，机器人手臂的正逆运动求解，并控制手臂到达指定位置。

#### 代码参数

1. **--joint_angles_id**: 
   - 程序参数 `joint_angles_id` 可输入1，2，3分别对应三组不同的末端执行器姿态。单只手臂的七个电机本测试程序均有涉及。

#### 逻辑

1. **导入库**:

   - 导入 `rospy` 库以使用 ROS 的 Python 接口。
   - 导入 `fkSrv`   `twoArmHandPoseCmdSrv` `twoArmHandPoseCmd` `ikSolveParam` `armTargetPoses` `changeArmCtrlMode`服务话题相关的消息类型。
   - 导入 `math` `numpy` `time` `argparse` 库用来数学计算，矩阵运算，解析命令行输入参数等。  

2. **FK 正解服务 (fk_srv_client)**:

   - 功能：调用 ROS 服务 `/ik/fk_srv`，计算给定关节角度的正运动学解。
   - 输入：关节角度列表（弧度）。
   - 输出：末端执行器的位姿信息（位置和姿态）。

3. **IK 逆解服务 (CALL_Ik_srv)**:

   - 功能：调用 ROS 服务 `/ik/two_arm_hand_pose_cmd_srv`，计算给定关节角度的正运动学解。
   - 输入： 详见字段 `twoArmHandPoseCmdRequest`
   - 输出： q_arm: 手臂关节值, 单位弧度

4. **设置手臂运动模式 (set_arm_control_mode)**:

   - 功能：调用 ROS 服务 `/arm_traj_change_mode`，设置机械臂的控制模式。
   - 输入：控制模式（整数）。
  
5. **发布手臂目标姿态 (publish_arm_target_poses)**:

   - 功能：将目标关节角度发布到 `/kuavo_arm_target_poses` 话题。
   - 输入：
      - times：时间列表。
      - values：关节角度列表（角度）。

6. **主函数 (main)**
   - 功能：初始化 ROS 节点，设置机械臂控制模式，调用 FK 正解服务，并发布目标姿态，回到初始位置，调用 IK 正解服务，并发布目标姿态。

### 总结

该代码通过 ROS 服务和话题，实现了手臂控制模式的更改，机器人手臂的正逆运动求解，并控制手臂到达指定位置，用户可以通过该程序来对机器人手臂控制模式进行更改，进行正逆运动的求解运算并观察手臂位置。

