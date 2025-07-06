# ARM ACTION SERVER

## 安装

1. 克隆仓库

```shell
# 此仓库需要依赖 ocs2_msgs 和 kuavo_ros_interfaces, 请此仓库克隆到在 kuavo-ros-control 或者 kuavo_ros1_workspace 工作空间中
git clone https://www.lejuhub.com/highlydynamic/planarmwebsocketservice.git
```

2. 编译依赖

```shell  
catkin_make ocs2_msgs kuavo_ros_interfaces planarmwebsocketservice
```

3. 安装依赖

```shell
cd planarmwebsocketservice
pip install -r requirements.txt
```


## 手臂动作服务器

运行命令

```shell
cd <catkin_workspace>
source devel/setup.bash
roslaunch planarmwebsocketservice plan_arm_action_websocket_server.launch robot_type:=kuavo # 如果是 ocs2 请使用 robot_type:=ocs2
```

这是一个 websocket 服务器，启动后会在所属网络中广播 `robot_info` 话题，内容如下：

```json
{
    "data": {
        "robot_name": ROBOT_NAME,
        "robot_ip": ROBOT_IP,
        "robot_connect_wifi": ROBOT_CONNECT_WIFI,
        "robot_ws_address": ROBOT_WS_ADDRESS,
        "robot_action_file_folder": ROBOT_ACTION_FILE_FOLDER,
        "robot_username": ROBOT_USERNAME,
        "robot_mac_address": ROBOT_MAC_ADDRESS,
    }
}
```

同时提供以下 websocket 接口(**请确保 kuavo 程序以 ros 节点运行，并且机器人已经在站立状态**):

#### 预览动作

request:

```json
{
    "cmd": "preview_action",
    "data": {
        "action_filename": "action_name",
        "action_file_MD5": "action_file_MD5",
    }
}
```

|名称|类型|描述|
|---|---|---|
|action_filename|string|动作文件名|
|action_file_MD5|string|动作文件MD5|

response:

```json
{
    "cmd": "preview_action",
    "data": {
        "code": 0,
        "status": 0,
        "progress": 0
    }
}
```

|名称|类型|描述|
|---|---|---|
|code|int|错误码 0: 成功 1: 动作文件不存在 2: 请求动作文件 MD5 与本地动作文件的 MD5 不一致|
|status|int|状态 0: 完成 1: 执行中|
|progress|int|动作执行进度, 单位为毫秒|

#### 停止预览

request:

```json
{
    "cmd": "stop_preview_action",
}
```

response:

```json
{
    "cmd": "stop_preview_action",
    "data": {
        "code": 0,
    }
}
```

|名称|类型|描述|
|---|---|---|
|code|int|错误码 0: 成功|



## Rosbag 到 Tact 文件转换工具

这是一个用于将 ROS bag 文件转换为 Tact 文件格式的工具。它主要用于处理机器人手臂、头部和手指的运动数据，并生成可用于动画或其他目的的 Tact 文件。

## 功能

- 录制手臂、头部和手指的 rosbag 数据
- 将 rosbag 数据转换为 tact 文件
- 数据平滑处理和控制点生成


## 使用方法

1. 运行主程序：
   ```
   cd <catkin_workspace>
   source devel/setup.bash
   cd planarmwebsocketservice
   python3 rosbag_to_act_frames.py
   ```

2. 在主菜单中选择所需的操作：
   - 录制手臂头部手指 rosbag 数据
   - 将 rosbag 数据转成 tact 文件

3. 按照屏幕上的提示进行操作。

main menu:

![main_menu](./imgs/main_menu.png)

record arm, head and hand rosbag:

![record_arm_head_hand_rosbag](./imgs/record_rosbag.png)

rosbag to tact:

![rosbag_to_tact](./imgs/rosbag_to_tact.png)


## 注意事项

- 确保您有足够的磁盘空间来存储生成的 tact 文件。
- 处理大型 rosbag 文件可能需要较长时间，请耐心等待。

## 故障排除

如果遇到问题，请检查以下几点：
- 确保所有依赖都已正确安装
- 检查 rosbag 文件是否完整且未损坏

## 贡献

欢迎提交 issues 和 pull requests 来帮助改进这个工具。
