# Bezier 手部规划轨迹程序

## 功能概述
本程序通过贝塞尔曲线实现机器人的手部运动轨迹规划，生成平滑的运动路径。

## `plan_arm_traj_bezier_no_tact_demo.py` 详细说明

`plan_arm_traj_bezier_no_tact_demo.py` 是一个用于通过贝塞尔曲线规划机器人手臂运动轨迹的脚本。该脚本的主要功能包括：

- **初始化和设置**：脚本首先初始化 ROS 节点，并设置必要的订阅者和发布者。它会订阅 `/bezier/arm_traj` 话题以接收轨迹数据，并发布到 `/kuavo_arm_traj` 话题以控制手臂运动。

- **服务调用**：脚本调用 `humanoid_change_arm_ctrl_mode` 服务来设置手臂的控制模式，并使用 `planArmTrajectoryBezierCurve` 服务来发送轨迹规划请求。

- **数据处理**：从 JSON 文件中加载动作数据，并通过 `add_init_frame` 和 `filter_data` 函数处理这些数据，以生成适合贝塞尔曲线插值的控制点。

- **轨迹规划**：通过 `create_bezier_request` 函数创建贝塞尔曲线请求，并调用 `plan_arm_trajectory_bezier_curve_client` 函数发送请求以规划手臂轨迹。

- **日志记录**：在 `trajectory_bezier.txt` 文件中记录轨迹信息，包括时间戳、关节位置和速度。


## 注意事项
1. 确保机器人手部运动范围内无障碍物。
2. 检查贝塞尔参数文件是否存在于指定路径。
3. 使用 Ctrl+C 可正常停止所有节点。
4. 确保手部运动路径在规划范围内。


## 补充说明
- 贝塞尔曲线插值法也可以读取 `welcome.tact` 文件，该文件中是一个预设动作，用户也可以自定义动作帧，具体可以参考 `plan_arm_traj_bezier_demo.py`。

## 代码说明

### `plan_arm_traj_bezier_no_tact_demo.py`
- 该脚本负责从JSON文件中加载贝塞尔曲线参数，并通过ROS服务调用来规划手臂的运动轨迹。
- 使用`/bezier/plan_arm_trajectory`服务来发送规划请求。
- 通过`/kuavo_arm_traj`话题发布关节状态。
- 记录轨迹信息到`trajectory_bezier.txt`文件中。