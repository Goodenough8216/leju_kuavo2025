#!/usr/bin/env python3
import rospy
import json
import math
import numpy as np
from kuavo_sdk.srv import planArmTrajectoryCubicSpline, planArmTrajectoryCubicSplineRequest
from sensor_msgs.msg import JointState
from kuavo_sdk.msg import JointTrajectory, JointTrajectoryPoint
from kuavo_sdk.srv import changeArmCtrlMode, changeArmCtrlModeRequest
from kuavo_sdk.msg import mpc_observation

# 全局变量定义
current_arm_joint_state = []  # 存储当前手臂关节状态

def deg_to_rad(deg):
    """
    将角度转换为弧度
    Args:
        deg: 角度值
    Returns:
        弧度值
    """
    return math.radians(deg)

def mpc_obs_callback(msg):
    """
    MPC观测回调函数
    从msg.state.value[24:]获取手臂关节状态并四舍五入到小数点后两位
    Args:
        msg: mpc_observation类型的消息,包含机器人状态数据
    """
    global current_arm_joint_state
    current_arm_joint_state = msg.state.value[24:]
    current_arm_joint_state = [round(pos, 2) for pos in current_arm_joint_state]

# 预定义的关节位置序列，每个位置包含14个关节角度值（弧度）
# 包括左右手臂各7个关节:
# - 左臂: l_arm_pitch, l_arm_roll, l_arm_yaw, l_forearm_pitch, l_hand_yaw, l_hand_pitch, l_hand_roll
# - 右臂: r_arm_pitch, r_arm_roll, r_arm_yaw, r_forearm_pitch, r_hand_yaw, r_hand_pitch, r_hand_roll
positions = [
    [deg_to_rad(angle) for angle in [0,0,0,0,0,0,0,0,0,0,0,0,0,0]],  # 初始位置
    [deg_to_rad(angle) for angle in [20,0,0,-30,0,0,0,20,0,0,-30,0,0,0]],  # 位置1
    [deg_to_rad(angle) for angle in [20,0,0,-30,0,0,0,-30,0,30,-88,8,-22,-4]],
    [deg_to_rad(angle) for angle in [20,0,0,-30,0,0,0,-30,-25,-54,-15,-6,-22,-4]],
    [deg_to_rad(angle) for angle in [10,10,-20,-70,0,0,-24,-30,-25,-54,-15,-6,-22,-4]],
    [deg_to_rad(angle) for angle in [14,20,33,-35,76,-18,3.5,-30,-25,-54,-15,-6,-22,-4]],
    [deg_to_rad(angle) for angle in [20,0,0,-30,0,0,0,20,0,0,-30,0,0,0]],
    [deg_to_rad(angle) for angle in [0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
]

# 定义时间序列，每个位置对应的时间点,每个位置间隔1.5秒
position_size = len(positions)
times = [3 + 1.5 * i for i in range(position_size)]  

# 用于存储关节状态的消息对象
joint_state = JointState()

def traj_callback(msg):
    """
    轨迹回调函数
    处理接收到的轨迹消息，更新joint_state
    Args:
        msg: JointTrajectory类型的消息,包含关节轨迹点数据
    """
    global joint_state
    if len(msg.points) == 0:
        return
    point = msg.points[0]
    # 设置关节名称
    joint_state.name = [
        "l_arm_pitch",
        "l_arm_roll",
        "l_arm_yaw",
        "l_forearm_pitch",
        "l_hand_yaw",
        "l_hand_pitch",
        "l_hand_roll",
        "r_arm_pitch",
        "r_arm_roll",
        "r_arm_yaw",
        "r_forearm_pitch",
        "r_hand_yaw",
        "r_hand_pitch",
        "r_hand_roll",
    ]
    # 将位置和速度从弧度转换为角度
    joint_state.position = [math.degrees(pos) for pos in point.positions[:14]]
    joint_state.velocity = [math.degrees(vel) for vel in point.velocities[:14]]
    joint_state.effort = [0] * 14

def call_change_arm_ctrl_mode_service(arm_ctrl_mode):
    """
    调用改变手臂控制模式的服务
    Args:
        arm_ctrl_mode: 控制模式（整数）
        - 0: 左臂控制
        - 1: 右臂控制  
        - 2: 双臂控制
    Returns:
        bool: 服务调用是否成功
    """
    result = True
    service_name = "humanoid_change_arm_ctrl_mode"
    try:
        rospy.wait_for_service(service_name, timeout=0.5)
        change_arm_ctrl_mode = rospy.ServiceProxy(service_name, changeArmCtrlMode)
        change_arm_ctrl_mode(control_mode=arm_ctrl_mode)
        rospy.loginfo("Service call successful")
    except (rospy.ServiceException, rospy.ROSException) as e:
        rospy.logerr(f"Service {service_name} call failed: {e}")
        result = False
    return result

def plan_arm_traj_cubicspline_demo():
    """
    使用三次样条插值规划手臂轨迹
    主要功能:
    1. 等待轨迹规划服务可用
    2. 创建轨迹规划请求
    3. 设置轨迹点和时间序列
    4. 发送规划请求并获取结果
    Returns:
        bool: 轨迹规划是否成功
    """
    rospy.wait_for_service('/cubic_spline/plan_arm_trajectory')
    plan_arm_traj_cubicspline = rospy.ServiceProxy('/cubic_spline/plan_arm_trajectory', planArmTrajectoryCubicSpline)
    request = planArmTrajectoryCubicSplineRequest()
    
    # 创建轨迹消息
    joint_trajectory = JointTrajectory()
    for i in range(len(times)):
        # 添加轨迹点
        joint_trajectory.points.append(JointTrajectoryPoint())
        joint_trajectory.points[-1].positions = positions[i]
        joint_trajectory.points[-1].time_from_start = rospy.Duration(times[i])
    
    # 设置请求参数
    request.joint_trajectory = joint_trajectory
    request.joint_trajectory.joint_names = ["l_arm_pitch", "l_arm_roll", "l_arm_yaw", "l_forearm_pitch", "l_hand_yaw", "l_hand_pitch", "l_hand_roll", "r_arm_pitch", "r_arm_roll", "r_arm_yaw", "r_forearm_pitch", "r_hand_yaw", "r_hand_pitch", "r_hand_roll"]
    
    # 发送请求并返回结果
    response = plan_arm_traj_cubicspline(request)
    return response.success

def main():
    """
    主函数：初始化节点，设置订阅者和发布者，执行轨迹规划
    主要流程:
    1. 初始化ROS节点
    2. 创建订阅者和发布者
    3. 设置手臂控制模式为双臂控制
    4. 等待获取当前关节状态
    5. 执行轨迹规划
    6. 以100Hz频率发布轨迹状态
    """
    # 初始化ROS节点
    rospy.init_node('arm_trajectory_cubicspline_demo')
    
    # 设置订阅者和发布者
    traj_sub = rospy.Subscriber('/cubic_spline/arm_traj', JointTrajectory, traj_callback, queue_size=1, tcp_nodelay=True)
    kuavo_arm_traj_pub = rospy.Publisher('/kuavo_arm_traj', JointState, queue_size=1, tcp_nodelay=True)
    mpc_obs_sub = rospy.Subscriber('/humanoid_mpc_observation', mpc_observation, mpc_obs_callback)
    
    # 设置手臂控制模式为双臂控制
    call_change_arm_ctrl_mode_service(2)
    
    # 等待获取当前关节状态
    while len(current_arm_joint_state) != 0:
        break
    
    # 将当前状态添加到轨迹起点
    times.insert(0, 2)
    positions.insert(0, current_arm_joint_state)
    
    # 执行轨迹规划
    success = plan_arm_traj_cubicspline_demo()
    if success:
        rospy.loginfo("Arm trajectory planned successfully")
    else:
        rospy.logerr("Failed to plan arm trajectory")

    # 主循环：以100Hz频率发布轨迹状态
    rate = 100  
    while not rospy.is_shutdown():
        try:
            if len(joint_state.position) == 0:
                continue
            kuavo_arm_traj_pub.publish(joint_state)
        except Exception as e:
            rospy.logerr(f"Failed to publish arm trajectory: {e}")
        except KeyboardInterrupt:
            break
        rospy.sleep(1/rate)

if __name__ == "__main__":
    main()
