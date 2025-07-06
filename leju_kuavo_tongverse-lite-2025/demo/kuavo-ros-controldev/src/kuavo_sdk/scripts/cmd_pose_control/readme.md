# robot_cmd_pose_control.py

### 代码说明

#### 功能

该代码实现了一个 ROS 节点，用于发布 `/cmd_pose` 话题的控制指令。该指令使用 `geometry_msgs/Twist` 消息类型来控制机器人的位置和姿态。通过发送位置指令，机器人可以从当前位置移动到目标位置。

#### 代码参数

1. **--pose_id**: 
   - 程序参数 `pose_id` 可输入1，2，3, 4, 5, 6分别对应机器人基于当前位置下沿x轴正方向移动1米，沿x轴负方向移动1米，沿y轴正方向移动1米，沿y轴负方向移动1米，沿z轴向左旋转90度，沿z轴向右旋转90度。

- **话题名称**: `/cmd_pose`
- **消息类型**: `geometry_msgs/Twist`
  - **linear.x**: 基于当前位置的 x 方向值，单位为米 (m)
  - **linear.y**: 基于当前位置的 y 方向值，单位为米 (m)
  - **linear.z**: 增量高度，单位为米 (m)
  - **angular.z**: 基于当前位置旋转（偏航）的角度，单位为弧度 (radian)
  - **angular.x** 和 **angular.y**: 未使用，设置为 0

#### 逻辑

1. **导入库**:
   - 导入 `rospy` 库以使用 ROS 的 Python 接口。
   - 导入 `Twist` 消息类型用于发布位置指令。

2. **定义发布函数**:
   - `publish_cmd_pose()`:
     - 创建一个发布者 `cmd_pose_pub`，用于发布 `Twist` 类型的消息到 `/cmd_pose` 话题。
     - 创建一个 `Twist` 消息对象 `cmd_pose_msg`。
     - 设置位置指令，其中 `linear.x` 为 1.0 m，表示基于当前位置向前移动 1.0 米，其他位置移动设置为 0。
     - 使用 `rospy.sleep(1)` 确保发布者连接成功。
     - 发布 `cmd_pose_msg`，并打印当前发布的位置信息。

3. **主程序**:
   - 在 `__main__` 中初始化 ROS 节点，节点名称为 `cmd_pose_publisher`。
   - 调用 `publish_cmd_pose()` 函数进行消息发布。

4. **异常处理**:
   - 使用 `try-except` 块捕获 `rospy.ROSInterruptException` 异常，以便在程序终止时进行适当处理。

### 总结

该代码提供了一个简单的接口，用于通过 ROS 发布 `/cmd_pose` 话题的位置信息。用户可以通过调整 `Twist` 消息中的参数来控制机器人的移动和姿态。由于该指令只需发送一次，因此在发布指令后程序会退出。