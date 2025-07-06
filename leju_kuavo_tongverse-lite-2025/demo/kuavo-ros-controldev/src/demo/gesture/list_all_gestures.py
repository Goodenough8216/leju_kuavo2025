#!/usr/bin/env python3
# coding: utf-8

import rospy
from kuavo_msgs.msg import gestureInfo
from kuavo_msgs.srv import gestureList, gestureListRequest

def list_gestures_client():
    """请求所有可用手势的服务"""
    # 服务的名称
    service_name = 'gesture/list'
    
    # 等待服务可用
    rospy.wait_for_service(service_name)
    
    try:
        # 创建服务代理
        list_gestures_service = rospy.ServiceProxy(service_name, gestureList)
        
        # 创建请求对象
        request = gestureListRequest()
        
        # 调用服务
        response = list_gestures_service(request)
        
        # 处理响应
        if response.success:
            print(f"Number of gestures: {response.gesture_count}")
            for gesture_info in response.gesture_infos:
                print(f"Gesture Name: {gesture_info.gesture_name}")
                print(f"Aliases: {', '.join(gesture_info.alias)}")
                print(f"Description: {gesture_info.description}\n")
        else:
            print(f"Failed to list gestures: {response.message}")
    
    except rospy.ServiceException as e:
        print(f"Service call failed: {e}")

if __name__ == "__main__":
    # 初始化节点
    rospy.init_node('list_gestures_client')
    
    # 调用服务
    list_gestures_client()