# beta 分支

## Breaking Changes
- 无

## 文档相关
- 新增 kuavo-humanoid-sdk 文档， [文档链接](src/kuavo_humanoid_sdk/docs/markdown/index.md)
- Kuavo 文档中心新增轮臂机器人介绍, 导航案例, 正逆解案例使用说明
- 更新 README 中全身控制器参数与 kuavo 配置参数的说明, [文档链接](./docs/info文件说明.md), [文档链接](./docs/kuavo_json文档说明.md)
- 运动控制接口新增乐聚自研夹爪控制接口`/control_robot_leju_claw`, 使用方法见 [文档链接](./docs/运动控制API.md)
- 新增 kuavo IK 正逆解模块使用说明, [文档链接](./src/manipulation_nodes/motion_capture_ik/how-to-use-kuavo-ik.md)
- 补充`/joint_cmd`控制话题中 control_mode 参数详细描述 [文档链接](./docs/运动控制API.md)
- 更新如何实时查看到 Quest3 投屏的屏幕使用文档, [文档链接](./docs/Quest3_VR_basic.md)
- 🎉🎉🎉 : 新增 Kuavo 产品介绍, 快速开始, 开发接口,功能案例等文档, [内测版文档网站链接](https://kuavo.lejurobot.com/beta_manual/basic_usage/kuavo-ros-control/docs/1%e4%ba%a7%e5%93%81%e4%bb%8b%e7%bb%8d/%e4%ba%a7%e5%93%81%e4%bb%8b%e7%bb%8d/index.html), [正式版文档网站链接](https://kuavo.lejurobot.com/manual/basic_usage/kuavo-ros-control/docs/1%e4%ba%a7%e5%93%81%e4%bb%8b%e7%bb%8d/%e4%ba%a7%e5%93%81%e4%bb%8b%e7%bb%8d/index.html)
- 补充运动控制接口文档中`/sensor_data_raw`话题的详细说明和数据示例 [文档链接](./docs/运动控制API.md)

## 新增功能
- 新增 kuavo-humanoid-sdk 应用层 SDK 用于控制机器人，pypi 项目：https://pypi.org/project/kuavo-humanoid-sdk
- 新增接收 Pico 发布脚部位置的节点，待算法部接入遥操作
- 新增 ROS话题`leju_claw_command`用于控制自研二指夹爪
- 新增夹爪手臂机器人模型，版本为47
- 新增一键安装脚本,可在刷完镜像的机器人上通过 wget 下载并执行完成自动安装环境等操作
- 新增实现通过按键控制手臂末端逆解和躯干逆解的案例，[文档链接](./docs/5功能案例/通用案例/按键控制躯干逆解.md)
- 完善URDF模型中的camera_base说明并添加全局静态TF转换以连接odom与像素坐标系，并新增元数据虚拟相机帧数据发布
- 新增机器人启动时无需人工搀扶辅助，可从双脚悬空或贴地状态切到站立状态的功能
- 新增kuavo_assets 各个版本机器人可视化的 launch 文件和 rviz 配置
- Gazebo 仿真器支持 Realsense 实感摄像头
- 新增支持 Gazebo 仿真器和基于共享内存的中间件读写 ROS 控制接口以减低通信延迟
- 新增机器人搬箱子应用案例
- 新增足部末端高度可调节功能
- 新增单步大转向+正常走，不平地面和斜坡行走功能
- 话题`/cmd_pose`和`/cmd_pose_world` 支持控制躯干 pitch
- 新增话题`/humanoid/single_step_mode` 用于发布机器人当前是否为单步控制状态
- VR 功能支持在结束录制和启动的时候可以指定手臂的关节位置
- 更新 kuavo_assets 包中机器人模型文件, 统一命名关节名称并添加关节限位, 扭矩限制, 速度限制等约束
- kuavo_sdk 新增正逆解使用说明, 单步控制使用说明, 位姿控制使用说明
- VR: Quest3 末端执行器支持乐聚自研夹爪, 可通过手柄上扳机或食指捏合控制夹爪
- 末端执行器支持乐聚自研夹爪, 可通过修改 kuavo.json 配置生效, 控制接口`/control_robot_leju_claw`
- 新增一些机器人校准动作方便在校准模式下检查电机是否正常
- VR: 新增 Quest3/Vision Pro 视频流功能
- 支持手柄控制头部运动, RT+左摇杆控制头部
- 支持通过修改 kuavo.json 配置文件`only_half_up_body`为 true 只使能上半身电机, 并适配了半身轮臂机器人, 使用见 [REAME文档](./readme.md)
- 工具: 添加支持同时开启热点和连接WIFI工具, WIFI名称`$ROBOT_NAME的热点`, 密码`kuavo123456`[使用文档链接](./tools/linux_wifi_hotspot/readme.md)
- IK 服务增加工作空间检查和可选打印求解信息提示

## 修复问题
- 修复手臂控制硬件层存在问题，降低控制频率防止手臂指令阻塞
- 修复路径跟踪示例到达终点后会超出一些距离问题，已通过到达终点前缓慢降低速度解决
- 修复`leju_claw_state` 夹爪状态话题消息无消息头时间戳问题
- 修复非 v42 版本机器人站立时存在抖动问题
- 新增长手臂电机配置文件 long_arm_config.yaml
- 修复地面高度估计对单步控制的影响导致摔倒问题
- 修复`/cmd_pose`控制指令的自动启停触发问题，现在可以正确触发启停并支持被`/cmd_vel`指令覆盖
- 修复多次单步控制后容易摔倒的问题, 已通过使用规划值作为 init_target_state 的起点来解决
- 修复在一些电机出问题的机器上启动时存在电机异响的问题, 通过增大站立控制器的关节跟踪权重来避免异响
- 修复 42 版本行走时躯干 pitch limit 晃动幅度变大问题
- 修复单步控制模式下未对高度进行补偿导致的高度变化问题
- 修复 Gazebo 仿真器中相机数据格式不完整问题, 补充使用仿真时间戳, 相机内参矩阵, 畸变参数和旋转矩阵等信息
- 修复 45 版本机器人由于 mesh 文件路径错误导致 VR 启动报错, 无法遥操作控制手臂问题
- 修复 autogait 阈值和键盘控制 10%的行程重合导致行走和站立频繁切换的问题
- 修复后退时对斜面的处理导致踩脚,增加斜坡规划生效的阈值
- 修复在半身模式下使用 VR 推摇杆会导致程序挂掉问题
- 依据实体测量数据更新 KUAVO 4Pro 总质量配置，实现仿真系统与物理实体参数一致性
- 修复机械臂关节扭矩反馈系统的电流-扭矩转换(C2T)系数校准错误
- 修复 h12pro 遥控器没有正确加载`~/.bashrc`中定义的 ROS 环境变量的问题
- 修复机器人站立瞬间容易受外力影响导致的瞬间异响电流问题
- 修复 43 版本的 URDF mesh 文件路径错误
- 修复在 40，41等版本只使用上半身功能是手臂抽搐问题
- 修复由于 kuavo_assets 包中 drake 相关的 URDF 文件的 mesh 文件名称不对，导致无法正常使用 VR 和 IK 功能
- 修复机器人原地踏步抽搐无法行走，VR无法启动跟随手势的问题
- 修复 URDF 更改未重新编译 cppad 导致异常不生效的问题, 已通过严格检查 URDF 文件的 MD5 校验来实现
- 修复由于缺少 `lusb` 而导致的编译错误, 已通过在编译时先检查或安装`libusb-1.0-0-dev`来解决
- 修复 h12pro 遥控器无法控制机器人站立, 行走等问题
- 修复潜在的执行`sudo apt install ros-noetic-Pinocchio -y`而升级版本导致的函数接口不兼容编译报错问题
- 修复程序结束后手臂电机未正常掉使能的问题
- 修复由于`/humanoid_wbc_observation` 话题数据维度减少修改导致 h12 遥控器播放动作失败问题
- 更正  biped_s42 机器人的 URDF 文件, 新增雷达的 TF, 调整 mesh，惯量和限位
- 修复 biped_s42 机器人头部 yaw 电机方向问题, 已根据右手定更正

## 其他改进
- 限制 pitch limit 速度上下限，避免机器人弯腰时姿态调整过于剧烈
- 优化 cppad 缓存机制，将缓存目录改为基于 URDF 哈希值的文件夹，支持保留多个版本的 cppad 缓存，实现不同 URDF 版本间的快速切换而无需重新编译
- 统一 42、45 等各个版本机器人的 URDF 中的质量
- 去除头部关机限位硬编码，更正为从 URDF 中获取
- 优化手臂电机模块，提升手臂电机的刷新速率至 300 Hz
- IK 模块去除求解时的关节限制， 更正为在 URDF 中定义上下限位
- 优化 VR 控制乐聚夹爪的速度为 90 以提高夹爪控制反馈速度
- 添加 ROS snapshot 源用于指定安装`ros-noetic-pinocchio=2.6.21-1focal.20240830.092123`避免升级 pinocchio 版本导致的编译错误
- 提高遥控器指令截止频率以提高灵敏度
- 优化下肢电机控制模块，提高走路，站立等动作的稳定性
- 添加 hardware tools 用于硬件测试故障排查
- 遥控器控制改进: 使用五阶低通滤波对 cmdVel 进行滤波，截止频率1hz
- 头部 yaw 关节软限位修正, 放宽至 +- 80°

# 1.0.0

## Breaking Changes

## 文档相关
- 更新文档说明如何检测手臂电机运动方向, [文档链接](./docs/硬件基本设置/README.md)
- 更新当前机器人发布和订阅话题的详细描述, 数据单位与物理含义, [文档链接](/docs/运动控制API.md)
- 增加出厂流程文档, [文档链接](./docs/硬件基本设置/README.md)
- 更新 motion_capture_ik 使用文档和示例, [文档链接](./src/manipulation_nodes/motion_capture_ik/README.md)
- 更新 Quest3 VR 使用文档, 补充如何去除空间限制步骤  [文档链接](./ docs/Quest3_VR_basic.md)
- 更新 README 添加了开源版本容器镜像下载链接和使用指南 [文档链接](readme.md) 
- 更新运动控制 API 接口文档, 新增`/gesture/list`, `/gesture/execute` 手势相关服务接口 [文档链接](./docs/运动控制API.md)

## 新增功能
- 新增长手臂4.3版本机器人, 增加对应的 URDF 文件
- 新增手臂电机 CAN 模块识别与绑定功能, 避免与夹爪模块冲突
- 适配 MPC 不同手臂自由度的机器人
- 新增辅助校准功能用于快速调整手臂零点圈数回零, 实现校准圈数、以当前位置作为零点、使能掉使能等方便调试
- 新增遥控器按键启动 VR 遥操作功能
- 新增贝塞尔曲线插值和三次样曲线插值手臂轨迹规划功能和对应的 websocket 播放服务, [文档链接](src/humanoid-control/humanoid_plan_arm_trajectory/README.md)
- 新增 VR 启动时允许指定上扳机只控制拇指+食指 or 除拇指虎口方向外的全部手指自由度
- 新增`/cmd_pose`位置控制接口
- 新增使用基于 MPC 控制机器人行走正方形/圆形/S曲线的示例 [文档链接](src/demo/trace_path/README.md)
- 新增电机 cali 模式下辅助校准功能, 增加准备阶段按`h`可查看当前操作流程提示功能
- 新增开机自启动遥控器控制机器人的功能, 目前支持 h12pro controller, [文档链接](./src/humanoid-control/h12pro_controller_node/ocs2_README.md)
- 新增简单案例: 抓取案例, 使用`/cmd_vel`控制走正方形/圆形/S曲线, 按键控制手臂微调 [文档链接](src/humanoid-control/humanoid_arm_control/README.md),[文档链接1](src/humanoid-control/humanoid_arm_control/docs/arm-control-keyboard.md)
- 兼容 4代、4pro版本的机器人, 新增单步控制功能
- 新增使用 ROBOT_VERSION 环境变量设置机器人版本号, 根据版本号选择对应的配置文件
- 新增自动根据机器人总质量修改对应模型和编译 cppad 功能
- 使用 ROS 标准方式重构开源版本的编译安装方式
- 新增 4.0 与 4.2 仿真环境的头部模型和控制, 并发布头部关节到 TF 与 Rviz 可视化
- 新增 4.1 版本机器人模型与配置文件
- 新增实物 call_leg 参数用于校准腿部零点, 允许直接设置当前腿部位置为零点
- 新增 h12 遥控器启停录制 VR 的手臂轨迹和摄像头图像的功能
- 新增 Quest3 手势识别功能, 支持'握拳', '点赞', 'OK', '666' 等手势, 支持在一键启动中通过`predict_gesture`参数开启
- 新增 ROS 手势执行和获取手势列表服务接口, 接口详情和支持的手势列表见[文档](./docs/运动控制API.md)

## 修复问题 
- 修复 rosparam 获取`/mpc/mpcArmsDof`和`/armRealDof`参数时等待条件错误问题
- 在校准腿部时不检查关节限位, 在跳圈或者编码器和零点位置相差较大时不触发保护挂掉,而是允许进入cali_leg模式校准
- 修复py和shell脚本安装之后没有可执行权限  会导致开源仓库无法rosrun执行脚本,无法使用键盘控制脚本等现象
- 修复humanoid_wbc_observation时间和mode没有更新, 可能导致运行时间久了动作不执行
- 修复 4.2 版本各个 URDF 中的机器人初始质量不一致问题
- 修复全自由度buffer未初始化问题, 导致手臂切换到非遥操作mode会崩溃
- 修复启动时quest3中有几率姿态或者手柄消息为None导致的报错
- 修复 VR 卡顿问题, 原因是增加手势识别功能在每个周期都会查询 TF 树多次, 导致循环卡顿, 已通过修改为每个周期调用一遍解决
- 修复踝关节转换中v和t错误的问题, 用上个周期的电机位置和通过电机位置换算得到的关节位置作为程序的输入，得到期望电机速度和期望电机电流
- 消除长手臂平举时无法伸直的问题, 适配所有手臂, Quest手臂长度改为实时计算
- 修复切换手臂模式时跳变问题, 原因是简化的手臂维度在切换手臂模式时没有经过mpc的插值
- 修复启动时 Quest3 中偶现姿态或者手柄消息为 None 导致的报错
- 修复 手势识别导致的 VR 卡顿问题
- 修复手臂电机默认的相序辨别配置错误问题
- 修复硬件模块捕获不到 signalint 导致手臂电机无法掉使能的问题
- 修复由于 VRHandCommandNode 未初始化可能会阻塞 Ctrl+C 退出
- 修复 4pro mujoco 模型手臂高度错误导致仿真中需要修改 pitch 增益才能运行问题
- 修改手臂电机使能之后存在自动回零的问题，通过发送当前位置一次之后可以避免自动回零解决
- 修复 VR 使用提示无法找到 xx_ik.py 问题, 以及手指控制跳变问题
- 修复开机自启动遥控器控制机器人功能中错误的包路径导致的安装失败问题
- 修复 youda 驱动器版本机器人行走出现全身抖动问题, 原因是缺少髋关节力控、腿部楼空版本质量、手臂末端的零速度约束...
- 修复脚本文件无可执行权限问题, 已通过在 cmake install 中追加权限解决
- 修复开源仓库硬件包查找路径错误问题, 使用 rospack 获取而不是通过维护环境变量
- 修复由于适配驱动器引入的 CST, CSV 下索引错误问题
- 修复由于 IMU 校准导致机器人抖动的问题
- 修复 EcMaster 写入零点文件后导致文件权限和所有者变更问题
- 修复 ROS 接口读取头部位置数据无变化问题, 原因是 hardware 模块没有上传头部电机数据
- 修复容器脚本启动时报错提示`不支持 robot_version 34 版本`, 已更新启动容器脚步中 robot_version 环境变量解决
- 修复遥控器启动 VR 程序时用户无法感知切换手臂控制模式的反馈, 已通过简化操作步骤, 长按遥控器扳机键解决

## 性能优化
- 轻量化 kuavo_assets 的模型文件, 从 1.1G 减小为 130+M
- 修复头部动作执行时不断读取 json 降低效率问题

## 其他改进
- kuavo_assets 更新各个版本机器人的模型快照
- 增加遥控器topic维度不对时进行提示, 防止自动触发开始发生危险
- trot步态去除SS相
- 临时禁用不正确的 quest 头部控制, 等待后续修复
- ik的末端和手肘位置存放在配置文件中, 无需按照原来的link名字的手臂urdf也能使用
- motion_capture_ik IK 节点统一使用 kuavo_asserts包中的模型文件
- 简化 4 代机器人头部和躯干模型的面数,统一 xml 的引入格式为 obj
- 优化 launch 中通过 gnome-terminal 开启的节点作为 launch 的子进程,避免 launch 退出时没有关闭节点
- 根据版本、自由度、质心模型存放 cppad 缓存, 避免混淆
- dockerfile 增加 ros-noetic-rqt-graph ROS 包安装步骤
- 重构手臂电机零点调整功能, 使用单独的零点文件 arms_zero.yaml，不存在时自动获取一次当前位置作为零点
- 移除废弃的 GPU dockerfile 文件
- 整理 URDF 模型文件, mujoco 模型文件和硬件相关的配置文件到 kuavo_assets 包中统一进行管理
- 使用 config 配置目录的 EcMasterType.ini 来指定驱动器类型, 并在编译时提示选择驱动器版本(4.2版本之后生效)
- 添加 CPU 温度、频率、占用率记录和发布方便观测与调试
- 优化日志打印提示, 消除编译告警和补充开源仓库缺失的一些脚本和节点
- 移除废弃的代码,脚本和编译选项
- 新增 VR 相关节点运行状况监控功能, S如果异常会在终端打印提示用户
- 运行环境为容器时取消系统信息发布, 原因是在容器里无法读取温度传感器等信息，同时容器的性能也不能作为系统性能的参考
- 改进容器启动脚本, 自动使用最新的镜像, 每个文件夹启动的容器名字使用不同hash区分
