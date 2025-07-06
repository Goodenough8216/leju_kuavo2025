# robot_vel_control.py

### 代码说明

#### 功能

该代码实现了一个 ROS 节点，用于发布 `/cmd_vel` 话题的控制指令。该指令使用 `geometry_msgs/Twist` 消息类型来控制机器人的速度和姿态。通过发布速度指令，机器人可以在不同状态之间切换，例如行走或站立。

#### 代码参数

- **话题名称**: `/cmd_vel`
- **消息类型**: `geometry_msgs/Twist`
  - **linear.x**: x 方向速度，单位为米每秒 (m/s)
  - **linear.y**: y 方向速度，单位为米每秒 (m/s)
  - **linear.z**: 增量高度，单位为米 (m)
  - **angular.z**: yaw 方向速度，单位为弧度每秒 (radian/s)
  - **angular.x** 和 **angular.y**: 未使用，设置为 0

#### 逻辑

1. **导入库**:
   - 导入 `rospy` 库以使用 ROS 的 Python 接口。
   - 导入 `Twist` 消息类型用于发布速度指令。

2. **定义发布函数**:
   - `publish_cmd_vel(cmd_vel_pub)`:
     - 设置发布频率为 10 Hz。
     - 创建一个 `Twist` 消息对象 `cmd_vel_msg`。
     - 设置速度指令，其中 `linear.x` 为 0.2 m/s，其他速度和角速度设置为 0。
     - 在循环中持续发布 `cmd_vel_msg`，并打印当前发布的速度指令。

3. **主程序**:
   - 初始化 ROS 节点，节点名称为 `cmd_vel_publisher`。
   - 创建一个发布者 `cmd_vel_pub`，用于发布 `Twist` 类型的消息到 `/cmd_vel` 话题。
   - 调用 `publish_cmd_vel(cmd_vel_pub)` 函数进行消息发布。

4. **异常处理**:
   - 使用 `try-except` 块捕获 `rospy.ROSInterruptException` 异常，以便在程序终止时进行适当处理。

### 总结

该代码提供了一个简单的接口，用于通过 ROS 发布 `/cmd_vel` 话题的速度指令。用户可以通过调整 `Twist` 消息中的参数来控制机器人的移动和姿态。通过发送非零速度指令，机器人可以切换到行走状态；通过发送全零速度指令，机器人可以切换到站立状态。

# robot_cmd_pose_control.py

### 代码说明

#### 功能

该代码实现了一个 ROS 节点，用于发布 `/cmd_pose` 话题的控制指令。该指令使用 `geometry_msgs/Twist` 消息类型来控制机器人的位置和姿态。通过发送位置指令，机器人可以从当前位置移动到目标位置。

#### 代码参数

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
     - 设置位置指令，其中 `linear.x` 为 0.5 m，表示基于当前位置向前移动 0.5 米，其他速度和角速度设置为 0。
     - 使用 `rospy.sleep(1)` 确保发布者连接成功。
     - 发布 `cmd_pose_msg`，并打印当前发布的位置信息。

3. **主程序**:
   - 在 `__main__` 中初始化 ROS 节点，节点名称为 `cmd_pose_publisher`。
   - 调用 `publish_cmd_pose()` 函数进行消息发布。

4. **异常处理**:
   - 使用 `try-except` 块捕获 `rospy.ROSInterruptException` 异常，以便在程序终止时进行适当处理。

### 总结

该代码提供了一个简单的接口，用于通过 ROS 发布 `/cmd_pose` 话题的位置信息。用户可以通过调整 `Twist` 消息中的参数来控制机器人的移动和姿态。由于该指令只需发送一次，因此在发布指令后程序会退出。