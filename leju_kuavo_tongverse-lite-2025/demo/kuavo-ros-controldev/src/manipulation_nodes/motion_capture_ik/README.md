# 使用方法
## 编译
```bash
catkin build motion_capture_ik
```
## 运行
### ros调用
- 启动ik节点
```bash
roslaunch motion_capture_ik ik_node.launch visualize:=1 print_ik_info:=false
```
- 调用逆解
```bash
rosrun motion_capture_ik sim_ik_cmd.py
```
调用后会打印出求解结果，例如：
```bash
time_cost: 4.02 ms. left_pos_error: 2.13 mm, right_pos_error: 2.20 mm
```
- 调用正解
```bash
rosrun motion_capture_ik test_fk_srv.py
```
#### Note:
- 如果设置`print_ik_info:=true`，则会打印详细的求解信息如下：
```bash
++++++++++++++++++++++++++++++++++++++++++++++++++++++
+++++++++++++++++++ IK RESULT INFO +++++++++++++++++++
++++++++++++++++++++++++++++++++++++++++++++++++++++++
Eef cmd: 
  Left pos: 0.243 0.355 0.319
  Left quat:  0.426 -0.661 -0.352  0.508
  Right pos: -0.161 -0.336  0.004
  Right quat: -0.495  0.032  0.066  0.866
Result:
  q: -0.836  0.812 -0.453 -1.372 -0.269  0.161 -0.067  0.413 -0.205 -0.307  0.009  0.340 -0.289 -0.852
  Left eef pos: 0.242 0.352 0.317, Left eef quat:  0.426 -0.661 -0.351  0.508
  Left pos error: -0.001 -0.002 -0.002, error norm: 3.344 mm.
  Right eef pos: -0.159 -0.334  0.007, Right eef quat: -0.491  0.041  0.073  0.867
  Right pos error: 0.002 0.002 0.003, error norm: 4.276 mm.

```
- ik求解末端（eef）定义在手掌中心，如果把eef定义在最后一个关节上，则需要在launch时设置`eef_z_bias:=0.0`。eef_z_bias标准值为-0.17。

### 直接调用
```bash
rosrun motion_capture_ik plant_ik_test
```
- 参考`test/plant_ik_test.cpp`

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

