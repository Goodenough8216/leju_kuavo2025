# Demo - 按键控制手臂

本程序允许用户通过键盘来控制机器人手臂臂的末端执行器（eef）在三维空间中的移动。它使用逆运动学（IK）算法来计算达到给定位置所需的关节角度，并通过插值方法平滑地改变关节的位置。

## 使用说明

运行本脚本后，可通过键盘控制手臂的末端沿XYZ轴方向移动，每次步进为`0.03 m`:

- **w/s** 按键: 控制 x 加减，

- **a/d** 按键: 控制 y 加减，

- **q/e**按键: 控制 z 加减，

松开按键时，程序会计算新的关节角度，并将手臂沿着平滑的轨迹移动到新位置。

## 运行启动

```bash
chmod +x src/humanoid-control/humanoid_arm_control/scripts/arm_control_with_keyboard.py
source devel/setup.bash
rosrun  humanoid_arm_control arm_control_with_keyboard.py
```

## 初始化参数

- `x_gap`, `y_gap`, `z_gap` : 每次按键时EEF沿XYZ轴移动的距离，默认为0.03米。
- `time_gap` : 每次移动的时间间隔，默认为0.5秒。
- `which_hand` : 控制哪只手，默认为右手。
- `interpolate_num` : 插值时使用的点数，默认为50。

## 注意事项

 如果更改了手臂模型或相关配置，请相应地调整 model_file 和其他相关参数。
