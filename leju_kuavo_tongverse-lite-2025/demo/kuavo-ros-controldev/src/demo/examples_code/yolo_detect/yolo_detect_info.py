#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import numpy as np
from vision_msgs.msg import Detection2DArray


def normalize_quaternion(quat):
    norm = np.linalg.norm(quat)
    if norm == 0:
        raise ValueError("Cannot normalize a zero-length quaternion")
    return quat / norm

def get_position_and_orientation():
    # 等待并获取一次消息
    data = rospy.wait_for_message("/object_yolo_box_tf2_torso_result", Detection2DArray)
    
    # 假设我们只关心第一个检测结果
    if data.detections:
        detection = data.detections[0]
        position = detection.results[0].pose.pose.position
        orientation = detection.results[0].pose.pose.orientation
        
        # 返回 Position 和 Orientation
        return (position, orientation)
    else:
        return (None, None)


if __name__ == '__main__':
    rospy.init_node('object_yolo_box_listener', anonymous=True)
    position, orientation = get_position_and_orientation()
    
    # 提取位置和四元数
    xyz = [position.x, position.y, position.z]
    quat = [orientation.x, orientation.y, orientation.z, orientation.w]
    
    # 对四元数进行归一化
    quat_normalized = normalize_quaternion(quat)