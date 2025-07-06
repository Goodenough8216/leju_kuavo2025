# robot_end_hand_gesture.py
### 代码说明

#### 功能

该代码通过 ROS 服务，实现了机器人灵巧手预设手势的展示。

#### 代码参数

1. **--gesture_name_id**: 
   - 程序参数 `gesture_name_id` 可输入1到24分别对应灵巧手预设的24个手势。

#### 逻辑

1. **导入库**:

   - 导入 `rospy` 库以使用 ROS 的 Python 接口。
   - 导入 `gestureTask`   `gestureExecute` `gestureExecuteState`服务话题相关的消息类型。
   - 导入 `time` `argparse` 库用来解析命令行输入参数等。  

2. **执行预设手势服务  (gesture_client)**:

   - 功能：调用 ROS 服务 `/gesture/execute`，执行预设手势
   - 输入：服务 `/gesture/execute`的请求值，`gesture_name`预设的手势名称和`hand_side`选择要使用的手。
   - 输出：服务 `/gesture/execute`的响应值，是否成功（bool类型）。

3. **查询是否有预设手势执行 (gesture_state_client)**:

   - 功能：调用 ROS 服务 `/gesture/execute_state`，查询是否有预设手势执行。
   - 输入： 无。
   - 输出： 是否有预设手势执行（bool类型）。

4. **主函数 (main)**
   - 功能：初始化 ROS 节点，获取手势名称，调用服务 `/gesture/execute` 执行预设手势，调用服务 `/gesture/execute` 查询是否有预设手势执行，手势执行完毕后，返回初始状态。


### 总结

该代码通过 ROS 服务，实现了机器人灵巧手预设手势的展示。


