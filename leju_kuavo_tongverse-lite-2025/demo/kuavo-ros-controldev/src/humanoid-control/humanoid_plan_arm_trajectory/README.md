# 人形机器人手臂轨迹规划

## 1. 概述

本软件包用于规划人形机器人的手臂关节状态轨迹。目前提供两种插值方法：

1. 三次样条插值器
2. 贝塞尔曲线插值器

## 2. 安装和使用

### 2.1 构建

```bash
catkin build humanoid_plan_arm_trajectory
```

### 2.2 运行

作为 ROS 节点运行：
```bash
roslaunch humanoid_plan_arm_trajectory humanoid_plan_arm_trajectory.launch
```

作为 nodelet 运行：
```bash
roslaunch humanoid_plan_arm_trajectory humanoid_plan_arm_trajectory.launch use_nodelet:=true
```

### 2.3 集成

要将此软件包集成到您的 ROS launch 文件中，请添加：

```xml
<include file="$(find humanoid_plan_arm_trajectory)/launch/humanoid_plan_arm_trajectory.launch"/>
```

对于与现有 nodelet 管理器的 nodelet 集成，请在 launch 文件中注释掉这一行：

```xml
<!-- <node pkg="nodelet" type="nodelet" name="nodelet_manager" args="manager" output="screen" /> -->
```

## 3. 接口

每种插值方法提供以下接口：

- 服务：`/<interpolate_type>/plan_arm_trajectory`
- 发布的话题：
  - `/<interpolate_type>/arm_traj`：规划的手臂轨迹
  - `/<interpolate_type>/arm_traj_state`：手臂轨迹的当前状态

其中 `<interpolate_type>` 可以是 `cubic_spline` 或 `bezier`。

### 3.1 服务

**planArmTrajectoryBezierCurve 服务：**

名称：`/bezier/plan_arm_trajectory`

服务类型：`humanoid_plan_arm_trajectory/planArmTrajectoryBezierCurve`

请求参数：

| 参数 | 类型 | 描述 |
|-----------|------|-------------|
| multi_joint_bezier_trajectory | jointBezierTrajectory[] | 多个关节的贝塞尔轨迹数组 |
| start_frame_time | float64 | 轨迹的开始时间，单位为秒（默认为0） |
| end_frame_time | float64 | 轨迹的结束时间，单位为秒 |
| joint_names | string[] | 关节名称 |

jointBezierTrajectory 消息：

| 字段 | 类型 | 描述 |
|-------|------|-------------|
| bezier_curve_points | bezierCurveCubicPoint[] | 贝塞尔曲线控制点数组 |

bezierCurveCubicPoint 消息：

| 字段 | 类型 | 描述 |
|-------|------|-------------|
| end_point | float64[2] | 贝塞尔曲线段的终点 [时间, 值] |
| left_control_point | float64[2] | 贝塞尔曲线段的左控制点 [时间, 值] |
| right_control_point | float64[2] | 贝塞尔曲线段的右控制点 [时间, 值] |

**注意：** 一条贝塞尔曲线由两个 bezierCurveCubicPoint 组成，`p0` 是第一个 bezierCurveCubicPoint 的终点，`p1` 是第一个 bezierCurveCubicPoint 的右点，`p2` 是第二个 bezierCurveCubicPoint 的左点，`p3` 是第二个 bezierCurveCubicPoint 的终点。

响应：

| 字段 | 类型 | 描述 |
|-------|------|-------------|
| success | bool | 表示轨迹规划是否成功 |

**planArmTrajectoryCubicSpline 服务**

名称：`/cubic_spline/plan_arm_trajectory`

服务类型：`humanoid_plan_arm_trajectory/planArmTrajectoryCubicSpline`

请求参数：

| 参数 | 类型 | 描述 |
|-----------|------|-------------|
| joint_trajectory | trajectory_msgs/JointTrajectory | 关节轨迹规范 |

JointTrajectory 消息：

| 字段 | 类型 | 描述 |
|-------|------|-------------|
| joint_names | string[] | 关节名称 |
| points | JointTrajectoryPoint[] | 轨迹点数组 |

JointTrajectoryPoint 消息：

| 字段 | 类型 | 描述 |
|-------|------|-------------|
| positions | float64[] | 该点的关节位置 |
| velocities | float64[] | 该点的关节速度（未使用） |
| accelerations | float64[] | 该点的关节加速度（未使用） |
| time_from_start | duration | 从轨迹开始到该点的时间 |

响应：

| 字段 | 类型 | 描述 |
|-------|------|-------------|
| success | bool | 表示轨迹规划是否成功 |

### 3.2 发布的话题

**arm_traj 话题：**

名称：`<interpolate_type>/arm_traj`

消息类型：`trajectory_msgs/JointTrajectory`

消息字段：

| 字段 | 类型 | 描述 |
|-------|------|-------------|
| header | std_msgs/Header | 消息头 |
| joint_names | string[] | 关节名称 |
| points | JointTrajectoryPoint[] | 轨迹点数组 |

**注意：** 
只有 points[0] 包含更新后的关节值。

---

**arm_traj_state 话题：**

名称：`<interpolate_type>/arm_traj_state`

消息类型：`humanoid_plan_arm_trajectory/planArmState`

消息字段：

| 字段 | 类型 | 描述 |
|-------|------|-------------|
| progress | int32 | 轨迹的进度，单位为毫秒 |
| is_finished | bool | 表示轨迹是否完成 |

## 4. 示例

人形机器人仿真（假设您已经构建了工作空间）

```bash
source <kuavo-ros-control>/devel/setup.bash
roslaunch humanoid_controllers load_normal_controller_mujoco_nodelet.launch 
```

手臂轨迹规划服务

```bash
roslaunch humanoid_plan_arm_trajectory humanoid_plan_arm_trajectory.launch
```

手臂轨迹规划客户端

```bash
source <kuavo-ros-control>/devel/setup.bash
cd src/humanoid-control/humanoid_plan_arm_trajectory/scripts
python plan_arm_traj_bezier_demo.py # 贝塞尔曲线
python plan_arm_traj_cubic_spline_demo.py # 三次样条
```

**供参考：**

您可以通过模仿上述两个演示脚本来编写客户端调用服务。

## 5. 注意事项

为了提高代码的可重用性、可移植性和可维护性，用户应该：

- 自行订阅结果话题
- 根据具体需求实现最终的控制逻辑

