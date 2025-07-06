# robot_end_claw.py

### 代码说明

#### 功能

该代码实现了一个 ROS 节点，订阅话题 `/leju_claw_state` 获得夹爪实时状态,调用服务 `/control_robot_leju_claw` 使机器人夹爪不断做开关动作。

#### 代码参数

1. **该程序无输入参数**  

2. **配置文件**：
   - 在运行本案例时，必须确保机器人手臂末端执行器为夹爪（二指爪），并且需要修改配置文件 `kuavo.json` 中 `EndEffectorType` 的值为 `lejuclaw` 。
   - 注意要选择机器人型号对应的配置文件，可以在终端中运行 `echo $ROBOT_VERSION` 来查看机器人型号，例如我的型号是42，所以修改配置文件 `kuavo.json` 的路径为 `kuavo-ros-control/src/kuavo_assets/config/kuavo_v42/kuavo.json` 。

3. **话题说明**：
   - **话题名称**: `/leju_claw_state`
   - **消息类型**: `kuavo_sdk.msg/lejuClawState` 字段具体含义详见接口文档

4. **服务说明**：
   - **话题名称**: `/control_robot_leju_claw`
   - **消息类型**: `kuavo_sdk.srv/controlLejuClaw` 字段具体含义详见接口文档

#### 逻辑

1. **导入库**:
   - 导入 `rospy` 库以使用 ROS 的 Python 接口。
   - 导入 `time` 库实现睡眠。
   - 导入 `controlLejuClaw`, `lejuClawState` 服务，话题相关消息格式。

2. **定义调用服务函数**:
   - `call_leju_claw_client()`:
     - 定义服务 `control_robot_leju_claw` 的请求数据，
     - 调用服务 `control_robot_leju_claw` 控制夹爪按照传入的参数运动。

3. **定义回调函数**：
   - `leju_calw_state_callback()`:
     - 从话题 `/leju_claw_state` 获得夹爪实时状态

4. **主程序**:
   - 在 `__main__` 中初始化 ROS 节点，节点名称为 `leju_claw_client_node`。
   - 创建话题订阅者 `claw_state_sub`，调用回调函数 `leju_calw_state_callback()`,从话题 `/leju_claw_state` 获得夹爪实时状态。
   - 调用服务 `control_robot_leju_claw` 控制夹爪按照传入的参数运动。
   - 节点不被停止时，在循环中使夹爪不断做开关动作。


### 总结

该代码提供了一个简单的接口，订阅话题 `/leju_claw_state` 获得夹爪实时状态,调用服务 `/control_robot_leju_claw` 使机器人夹爪不断做开关动作。
