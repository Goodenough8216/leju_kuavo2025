### 代码说明

#### 功能

该代码实现了一个 ROS 节点，用于发布机器人头部的目标姿态到指定的话题 `/robot_head_motion_data`。用户可以通过设置偏航（yaw）和俯仰（pitch）角度来控制头部的运动。

#### 代码参数

- **yaw (float)**: 头部的偏航角，取值范围为[-30, 30]度。
- **pitch (float)**: 头部的俯仰角，取值范围为[-25, 25]度。

#### 逻辑

1. **导入库**:

   - 导入 `rospy` 库以使用 ROS 的 Python 接口。
   - 导入 `robotHeadMotionData` 消息类型，用于定义头部运动数据消息。

2. **定义设置头部目标位置的函数**:

   - `set_head_target(yaw, pitch)`:

     - 创建一个 Publisher，发布到 `/robot_head_motion_data` 话题，消息类型为 `robotHeadMotionData`。
     - 使用 `rospy.sleep(0.5)` 确保 Publisher 已注册。
     - 创建 `robotHeadMotionData` 消息对象，并设置 `joint_data` 为传入的 `yaw` 和 `pitch`。
     - 发布消息，并记录日志，显示已发布的头部目标位置。

3. **主程序**:

   - 在脚本入口处，初始化 ROS 节点，节点名称为 `robot_head_controller`。
   - 调用 `set_head_target(0, 0)` 发布一次头部目标位置为 `0, 0`。

4. **入口点**:

   - 如果此脚本是主程序，则调用 `set_head_target` 函数，并处理可能的 ROS 中断异常。

### 总结

该代码提供了一个简单的接口，用于通过 ROS 发布头部的目标姿态，用户可以方便地设置偏航和俯仰角以控制头部的运动。通过确保 Publisher 注册，保证消息能够被及时接收。