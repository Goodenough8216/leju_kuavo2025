# 获取传感器数据信息

#### /sensors_data_raw

话题描述: 实物机器人, 仿真器发布的传感器原始数据

消息类型: `kuavo_sdk/sensorsData`

| 字段                | 类型                         | 描述                           |
| ----------------- | -------------------------- | ---------------------------- |
| sensor_time       | time                       | 时间戳                          |
| joint_data        | kuavo_msgs/jointData       | 关节数据: 位置,速度, 加速度, 电流         |
| imu_data          | kuavo_msgs/imuData         | 包含 gyro, acc, free_acc, quat |
| end_effector_data | kuavo_msgs/endEffectorData | 末端数据, 暂未使用                   |

- 其中关节数据的数组长度为`NUM_JOINT`, 对应的数据顺序为:
  - 前 12 个数据为下肢电机数据, 
    - 0~5 为左下肢数据, 从髋部到脚踝(l_leg_roll, l_leg_yaw, l_leg_pitch, l_knee, l_foot_pitch, l_foot_roll)或 (leg_l1_link ~ leg_l6_link),
    - 6 ~ 11 为右边下肢数据, 从髋部到脚踝(r_leg_roll, r_leg_yaw, r_leg_pitch, r_knee, r_foot_pitch, r_foot_roll)或 (leg_r1_link ~ leg_r6_link),
  - 接着 14 个数据为手臂电机数据, 
    - 12 ~ 18 左臂电机数据("l_arm_pitch","l_arm_roll","l_arm_yaw","l_forearm_pitch","l_hand_yaw","l_hand_pitch","l_hand_roll")或 (zarm_l1_link ~ zarm_l7_link), 
    - 19 ~ 25 为右臂电机数据("r_arm_pitch","r_arm_roll","r_arm_yaw","r_forearm_pitch","r_hand_yaw","r_hand_pitch","r_hand_roll")或 (zarm_r1_link ~ zarm_r7_link), 
  - 最后 2 个为头部电机数据, 分别为 head_yaw 和 head_pitch
- 位置单位(radian), 速度单位(radian/s), 加速度单位($\text{radian/s}^2$), 电流单位(A)
- imu 数据:
  - gyro: 表示陀螺仪的角速度，单位弧度每秒（rad/s）
  - acc: 表示加速度计的加速度，单位米每平方秒（m/s<sup>2</sup>）
  - quat: IMU的姿态（orientation）
