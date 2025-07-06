# HandControllerDemoROSNode

## 概述

本项目为 Kuavo 机器人手掌控制程序，通过接收来自 Noitom Hi5 手套手掌的数据，控制 Kuavo 机器人手掌的运动。手指关节信息用于控制机器人手掌的开合，而机器人的手臂此程序不做控制，仅做末端 POSE 的坐标转换成适配机器人世界坐标系，然后发布到指定 ROS TOPIC，手掌的朝向信息用于代表机器人手臂末端的朝向，而 VR 手柄则用于代表机器人手臂末端的位置。

## 事前准备

### 连接接收器

* 将手套的接收器连接至 VR 设备的 Type-C 接口。

### 安装传感器

* 将所有传感器安装至手套内。每个传感器侧面标有对应的关节名称，安装时需确保位置正确，且区分左右手。

### 传感器上电

* 将所有传感器放入充电底座。
* 连接充电底座至充电器，并等待 3 秒。
* 断开充电底座与充电器的连接，完成上电。

### 获取 VR 设备 IP 的方法

* 在 VR 设备设置菜单中，选择“常规” -> “关于” -> “头戴式设备状态”查看 IP 地址。

### 穿戴设备

* 穿戴手套，确保左右手套被正确区分。
* 绑定 VR 手柄在手臂上。

### 校正过程

* 穿戴手套并启动 VR 设备。
* 使用 VR 手柄启动传感器读取程序。
* 确认所有传感器均被正确识别，包括型号和电量。
* 在 VR 界面中，使用头戴设备选择菜单项。
* 若有问题，请回到第一步重新检查。
* 使用头戴设备选择“校准”选项，并按照指示调整手部位置完成校正。校正后，可在虚拟空间中看到手指和手掌的虚拟姿态。
* 校准完成后将 VR 设备挂在胸前。
* 请记住校正后手掌3D模型朝向，若与我们手部方向一致朝外说明正常，若相反则需要在启动控制机器人手臂程序是做出相应调整。

## 使用说明

### 编译代码

```bash
cd ~/kuavo_ros1_workspace
catkin build handcontrollerdemorosnode
```

### 赋予执行权限

```bash
sudo chmod +x ~/kuavo_ros1_workspace/src/handcontrollerdemorosnode/scripts/hand_controller_node.py
sudo chmod +x ~/kuavo_ros1_workspace/src/handcontrollerdemorosnode/scripts/arm_controller_node.py
```

### 运行代码

#### 启动 ROS Master
  
```bash
roscore
```

#### 运行 Noitom Hi5 手套数据接收节点

* 请参考文档 [Hi5手套数据接收节点](../noitom_hi5_hand_udp_python/readme.md) 运行 Hi5 手套数据接收节点。

#### 启动控制机器人手掌节点

```bash
source ~/kuavo_ros1_workspace/devel/setup.bash
rosrun handcontrollerdemorosnode hand_controller_node.py 
```

#### 启动处理末端 POSE 转换节点

* 启动程序后，程序会询问校正后手部3D模型是否与实际手部姿态一致（手部模型方向应该与我们的手方向一致朝外），若一致则输入 `1`，否则输入 `2`。
* 调试模式，在 `rosrun` 命令后加上 `--debug` 参数，可通过 `rviz` 查看手部3D模型和机器人手臂末端 POSE 的坐标转换情况(在 `rviz` 中加载 `TF` 和 `Marker` display)。
  
```bash
source ~/kuavo_ros1_workspace/devel/setup.bash
rosrun handcontrollerdemorosnode arm_controller_node.py 
```

#### 启动 IK 节点

* 请参考文档 [Kuavo 机器人手臂逆运动学节点](../motion_capture_ik/README.md) 运行 Kuavo 机器人手臂逆运动学节点。

### 本项目的 ROS Topic

1. KUAVO 机器人手臂末端的位姿信息
    * 话题名称: `/kuavo_hand_pose`
    * 消息类型: `handcontrollerdemorosnode/armPoseWithTimeStamp`
    * 消息定义: 
    ```shell
    int32 offset #与第一次publish的时间戳的差值，单位为ms，第一次publish的时间戳为0
    float64[] left_hand_pose #左手末端的位姿信息，依次为四元数的x,y,z,w和位置的x,y,z
    float64[] right_hand_pose #右手末端的位姿信息，依次为四元数的x,y,z,w和位置的x,y,z
    ```

2. 可视化 Marker
    * 话题名称: `/visualization_marker`
    * 消息类型: `visualization_msgs/Marker`
    * 消息定义: 
    ```shell
     std_msgs/Header header
        uint32 seq
        time stamp
        string frame_id
      string ns
      int32 id
      int32 type
      int32 action
      geometry_msgs/Pose pose
        geometry_msgs/Point position
          float64 x
          float64 y
          float64 z
        geometry_msgs/Quaternion orientation
          float64 x
          float64 y
          float64 z
          float64 w
      geometry_msgs/Vector3 scale
        float64 x
        float64 y
        float64 z
      std_msgs/ColorRGBA color
        float32 r
        float32 g
        float32 b
        float32 a
      duration lifetime
      bool frame_locked
      geometry_msgs/Point[] points
        float64 x
        float64 y
        float64 z
      std_msgs/ColorRGBA[] colors
        float32 r
        float32 g
        float32 b
        float32 a
      string text
      string mesh_resource
      bool mesh_use_embedded_materials
    ```
    
    * 注意：此话题只有在 `arm_controller_node` 运行时带上 `--debug` 参数才会发布。
## FAQ

### 1. 为什么我的手掌运动了，但是机器人手掌没有运动？

首先启动 `kuavo` 程序进入站立状态后，按下 `h` 和 `k` 键，查看机器人手掌是否能够正常掌合运动。如果不可以:

- 请检查编译时 `ROBOT_VERSION` 对应的 `kuavo.json` 中的 `EndEffectorType` 是否为 `qiangnao`。
- 编译完 `kuavo` 项目后，在 `build` 目录下执行 `sudo ./lib/hand_sdk/hand_test` 测试脚本，检查机器人手掌是否能够正常掌合运动, 若不能则大概率是硬件问题。

如果测试脚本正常，但是无法通过动捕手套控制机器人手掌，请检查：

- 请检查 `noitom_hi5_hand_udp_python` 项目中的 `noitom_hand_publish.py` 中的 `server_ip` 是否正确。
- 请检查 `control_end_hand` ROS 服务是否正常启动，可以通过 `rosservice list` 查看是否有 `control_end_hand` 服务。
- 请检查开启 `kuavo` 程序和启动控制机器人手臂手掌节点后，`kuavo` 运行时的终端输出，是否有输出 `setEndhand to left: [...], right: [...]`。

### 2.为什么机器人手掌运动与我的手掌运动速度不太匹配？

由于网络原因，手套数据发布节点发布的数据间隔比较大，波动也比较大，导致机器人手掌运动速度不太匹配。可以将 `VR` 设备连接一个比较优质的网络中。

### 3.为什么我的手掌关节运动的幅度跟机器人手掌关节运动的幅度不太一致？

首先仓库中 `src/handcontrollerdemorosnode/config/left_right_z_axis.json` 配置文件会记录左右手的手指关节 `z` 轴的最大最小值（值越小，手掌越张开，值越大，手掌越闭合）。如下：

```json
{
  "left": {
    "ForeArm": [0,0],
    "Hand": [0,0],
    "HandThumb1": [15,60],
    "HandThumb2": [45,80],
    "HandThumb3": [1,50],
    ...
  },
  "right": {
    "ForeArm": [0,0],
    "Hand": [0,0],
    "HandThumb1": [15,60],
    "HandThumb2": [34,75],
    "HandThumb3": [4,46],
    ...
  }
}
```

### 4. 我知道如何调整手掌关节的最大最小值，但是具体应该调哪个？

配置文件中：

```shell
HandThumb1 对应 大拇指外展肌
HandThumb2 对应 大拇关节
HandIndex1 对应 食指关节
HandMiddle1 对应 中指关节
HandRing1 对应 无名指关节
HandPinky1 对应 小指关节
```

- 假如你的手指关节**自然张开**，但是机器人对应的关节**不能完全张开**，那么你需要将对应关节的**最小值调大**。 
- 如果你的手指关节**自然闭合**，但是机器人对应的关节**不能完全闭合**，那么你需要将对应关节的**最大值调小**。

### 5. 为什么我的中指动了，但是机器人中指没有动而是食指动了？

请检查手套传感器是否正确安装到正确的手指上，以及手套是否正确穿戴。然后重新在 VR 设备中校准手部，观察手部 3D 模型的手指运动是否与实际手部运动一致。

### 6. 为什么我的左手动了，但是机器人右手动了？

- 情况1: 在文件 `src/handcontrollerdemorosnode/src/hand_controller_node.py` 中调用 `control_end_hand` 服务时，先传入 `right` 手掌的数据，再传入 `left` 手掌的数据。（由于当时开发时，镜像看起来比较方便，所以先传入右手数据，再传入左手数据）。
- 情况2: 发布手套数据的节点中，左右手的数据错位了，导致机器人手掌运动与实际手掌运动不一致。（现**应该**不会出现这种情况，因为 `noitom_hi5_hand_udp_python` 在 ([1c7bb2b9aa24e87e6bee92839c78bc0bf491a5c4](https://www.lejuhub.com/highlydynamic/kuavo_ros1_workspace/-/commit/1c7bb2b9aa24e87e6bee92839c78bc0bf491a5c4))后，已经解决这个问题）

## 配置文件说明

`src/handcontrollerdemorosnode/config/robot_arm_config.json`

- **offset**: 用于记录手部坐标系和机器人坐标系之间的偏移量。
  - **如何更新**: 可以通过 `rviz` 查看手部 3D 模型坐标和机器人手臂末端 POSE 的坐标情况(在 `rviz` 中加载 `TF` 和 `Marker` display), 然后用户的手臂自然垂直放下，`rviz` 中的 `tf` 可以获取机器人手臂末端和手部 3D 模型的位置坐标，此时记录下两者之间的偏移量，然后更新到配置文件中。
- **limit**: 用于记录左右手末端在 `x`，`y`，`z` 轴的最大最小值。
  - **如何更新**: 在 `rviz` 中通过 **joint_state_publisher_gui** 将机器人手臂分别调整到 `x`，`y`，`z` 轴的最大最小值，然后记录下对应的值，更新到配置文件中。(还需考虑实际情况下，机器人能否达到这个位置)


## 注意事项

* 请确保所有传感器已经上电, 并且位置均正确。
* 若出现传感器数据不稳定左右手数据错位的情况，请及时关掉程序，避免机器人运动过程中出现意外。
* 请在启动控制机器人手臂节点前，输入 `1` 或者 `2` 确认手部3D模型方向是否与实际手部方向一致，否则可能导致机器人手臂运动方向错误。
* 请在控制机器人期间，缓慢移动手臂，避免机器人手臂移动过快导致机器人倒地或者电机跟踪错误等问题。
