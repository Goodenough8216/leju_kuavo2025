# 更新日志
## 2024-09-19
使用Y键锁定/解锁手指，下扳机恢复为控制中指、无名指和小指。

## 2024-09-18
默认会启动quest3监听节点，必须手动指定ip（需要搭配kuavo_ros1_workspace仓库dev分支的最新版本）。例如:
```bash
roslaunch motion_capture_ik visualize.launch ip_address:=10.10.20.35
```
如果不需要，请在launch文件中设置`run_quest3_monitor:=false`:
```bash
roslaunch motion_capture_ik visualize.launch run_quest3_monitor:=false
```
## 2024-09-26
支持launch时指定手指控制方式。
- control_finger_type:=0：上扳机控制所有4指
- control_finger_type:=1：上扳机控制拇指（1维）、食指，下扳机控制中指、无名指、小指
# 说明
## 遥操作
请参考[kuavo_ros1_workspace仓库](https://www.lejuhub.com/highlydynamic/kuavo_ros1_workspace)的readme文件以使用遥操作。

## 调用ik求解
- 启动`motion_capture_ik`节点:
```bash
cd <path-to-kuavo-ws>
source devel/setup.bash
roslaunch motion_capture_ik visualize.launch visualize:=false robot_version:=4 control_hand_side:=2 send_srv:=1
```
> Note: 其中：visualize 置为 false 关闭可视化; robot_version 指代手臂版本，3代表 3.4 版本的手臂，4代表 4.0 版本的手臂；control_hand_side 指定控制的手，0表示左手，1表示右手，2表示双手；send_srv 置为 0 表示跳过ros服务调用（该服务用于启动机器人手臂控制），1表示调用ros服务，如果只想查看rviz可视化内容，请置为 0，否则程序会卡在等待ros服务response的地方。

- 参考`./scripts/test/sim_ik_cmd.py`文件发布末端位姿命令

## 通过键盘控制
- 启动`motion_capture_ik`节点:
```bash
cd <path-to-kuavo-ws>
source devel/setup.bash
roslaunch motion_capture_ik visualize.launch control_hand_side:=2 send_srv:=0 eef_z_bias:=-0.15 visualize:=1 enable_quest3:=0 use_cxx:=1
```

- 运行`keyboard_control_robot_arm_demo.py` 通过键盘控制机器人手臂运动（需要在kuavo终端开启手臂规划）
```bash
cd <path-to-kuavo-ws>
source devel/setup.bash
cd scripts/test
python3 keyboard_control_robot_arm_demo.py --hand left --step 0.08

Control Side: left
Step Size: 0.08
---------------------------------------------
-位置控制：
  前后:[按键<w>, 按键<s>]
  左右:[按键<a>, 按键<d>]
  上下:[按键<q>, 按键<e>]
-手部控制：
  抓取:[按键<h>]
  松开:[按键<k>]
----------------------------------------------
```

> Note: hand 指代控制的手，left 指代左手，right 指代右手。不支持双手控制。
> step 指代每次移动的距离，单位为米。

