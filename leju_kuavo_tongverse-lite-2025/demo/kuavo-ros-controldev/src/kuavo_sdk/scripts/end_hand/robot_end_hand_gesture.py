#!/usr/bin/env python3
# coding: utf-8

import rospy
import time
import argparse
from kuavo_sdk.msg import gestureTask
from kuavo_sdk.srv import gestureExecute, gestureExecuteRequest, gestureExecuteState, gestureExecuteStateRequest

def gesture_client(gesture_name, hand_side):
    """请求执行特定手势的服务"""
    # 服务的名称，需要与服务器端注册的服务名一致
    service_name = 'gesture/execute'
    
    # 等待服务可用
    rospy.wait_for_service(service_name)
    
    try:
        # 创建服务代理
        gesture_service = rospy.ServiceProxy(service_name, gestureExecute)
        
        # 创建请求对象
        request = gestureExecuteRequest()
        gesture1 = gestureTask(gesture_name=gesture_name, hand_side=hand_side)
        request.gestures = [gesture1]
        
        # 调用服务
        response = gesture_service(request)
        
        # 处理响应
        if response.success:
            print(f"Gesture '{gesture_name}' executed successfully.")
        else:
            print(f"Failed to execute gesture '{gesture_name}': {response.message}")
    
    except rospy.ServiceException as e:
        print(f"Service call failed: {e}")


def gesture_state_client()->bool:
    service_name = 'gesture/execute_state'
    
    # 等待服务可用
    rospy.wait_for_service(service_name)
    
    try:
        # 创建服务代理
        gesture_state_service = rospy.ServiceProxy(service_name, gestureExecuteState)
        
        # 创建请求对象
        request = gestureExecuteStateRequest()
        
        response = gesture_state_service(request)
        
        return response.is_executing
    
    except rospy.ServiceException as e:
        print(f"Service call failed: {e}")
    
    return None

if __name__ == "__main__":
    # 初始化节点
    rospy.init_node('gesture_client')
    
    # 获取手势名称，
    gesture_name_list = ['number_1', 'number_2', 'number_3', 'number_4', 'number_5', 
                         'number_6', 'number_7', 'number_8', 'two-finger-spread-unopposed', 'precision-pinch-opposed', 
                         'mouse-control', 'rock-and-roll', 'tripod-pinch-unpposed', 'flick-index-finger','flick-middle-finger',
                         'four-finger-straight', 'fist', 'thumbs-up', 'side-pinch', 'pen-grip1',
                         'pen-grip2', 'cylindrical-grip', 'five-finger-pinch', 'pen-grip3'
                         ]

    # 解析命令行参数  
    parser = argparse.ArgumentParser(description="选择不同的 gesture_name")
    parser.add_argument("--gesture_name_id", type=int, choices=range(1, 25), required=True, help="选择 gesture_name 的 ID (1-24)")
    args = parser.parse_args()

    # 根据传入的 gesture_name_id 选择对应的 gesture_name
    gesture_name = gesture_name_list[args.gesture_name_id - 1]
    print(f"Selected gesture_name: {gesture_name}")

    # 定义手部位置
    hand_side = 0  # 左手
    
    # 调用服务
    gesture_client(gesture_name, hand_side)
    
    # 循环 20 次，每次 sleep 100ms
    for _ in range(20):
        state = gesture_state_client()
        if state is not None:
            print(f"Current gesture state: {state}")
        time.sleep(0.1)  # 等待 100ms

    # reset hand gesture.    
    gesture_client("empty", hand_side=2)