### 代码说明

#### 功能

该代码实现了一个 ROS 节点，用于控制机器人的手部动作。它通过发布 `robotHandPosition` 消息到 `control_robot_hand_position` 话题来设置左手和右手的目标位置。

#### 代码参数

- `hand_traj (list)`:  包含左手和右手位置的列表。列表长度为 12，前 6 个元素表示左手位置，后 6 个元素表示右手位置。每个位置参数的具体含义由 `kuavo_sdk.msg` 中的 `robotHandPosition` 消息类型定义。

#### 逻辑

1. **导入库**: 导入 `rospy` 用于 ROS 功能，以及 `kuavo_sdk.msg` 中的 `robotHandPosition` 消息类型。

2. **`publish_controlEndHand` 函数**:
    - 创建一个 ROS 发布器 `pub`，用于发布消息到 `control_robot_hand_position` 话题。消息类型为 `robotHandPosition`，队列大小为 10。
    - 使用 `rospy.sleep(0.5)` 等待一段时间，确保发布器已注册。**（注意：这行代码并非必要，可以移除）**
    - 创建 `robotHandPosition` 消息对象 `msg`。
    - 将 `hand_traj` 列表中的值赋给 `msg.left_hand_position` 和 `msg.right_hand_position`，分别设置左手和右手的目标位置。
    - 使用 `pub.publish(msg)` 发布消息。
    - 使用 `try...except` 块捕获 `rospy.ServiceException` 异常，如果发布消息失败，则记录错误日志并返回 `False`。

3. **`main` 函数**:
    - 使用 `rospy.init_node('robot_hand_controller')` 初始化 ROS 节点，节点名称为 `robot_hand_controller`。
    - 创建一个示例 `hand_traj` 列表，包含左手和右手的目标位置数据。
    - 调用 `publish_controlEndHand` 函数发布手部位置。

4. **入口点**: 使用 `if __name__ == "__main__":`  确保代码只在作为主程序运行时执行。