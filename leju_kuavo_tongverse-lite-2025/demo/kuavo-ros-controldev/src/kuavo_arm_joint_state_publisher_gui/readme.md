# Kuavo Arm Joint State Publisher GUI

这个软件包是一个基于Python Qt的GUI工具，用于设置和发布Kuavo机器人手臂的关节状态值。它允许用户通过滑块直观地控制机器人手臂的各个关节，并将这些状态值发布到ROS系统中。

## 功能特点

* 只显示名称包含"arm"的关节，专注于机器人手臂控制
* 以100Hz的频率将关节状态发布到`/kuavo_arm_traj`话题
* 支持启用和禁用关节控制模式
* 从传感器数据自动刷新关节位置
* 可调整界面布局(行数)

## 安装

### 依赖项

本软件包依赖于以下ROS软件包：

* joint_state_publisher
* python_qt_binding
* rospy
* kuavo_msgs (用于服务调用和消息)

安装过程会自动检查并安装`ros-$ROS_DISTRO-joint-state-publisher`包。

### 构建

```bash
cd <kuavo-ros-control>
catkin build kuavo_arm_joint_state_publisher_gui kuavo_msgs
```

## 使用方法

启动软件包：

```bash
rosrun kuavo_arm_joint_state_publisher_gui kuavo_arm_joint_state_publisher_gui
```

## 主要功能

* **启用控制**：调用`/arm_traj_change_mode`服务，将控制模式设置为2
* **禁用控制**：调用`/arm_traj_change_mode`服务，将控制模式设置为1
* **从传感器刷新**：从`/sensors_data_raw`话题获取最新的传感器数据，更新关节位置

## 发布的话题

* `/kuavo_arm_traj` (sensor_msgs/JointState): 包含所有手臂关节的位置，以角度表示

## 订阅的话题

* `/sensors_data_raw` (kuavo_msgs/sensorsData): 传感器数据，用于刷新关节位置

## 调用的服务

* `/arm_traj_change_mode` (kuavo_msgs/changeArmCtrlMode): 用于启用或禁用手臂控制模式
