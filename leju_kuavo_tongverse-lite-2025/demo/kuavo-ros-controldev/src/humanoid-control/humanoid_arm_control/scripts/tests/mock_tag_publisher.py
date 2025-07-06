#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Mock 相机识别 apriltag 的位置信息
"""

import rospy
import copy
from apriltag_ros.msg import AprilTagDetection, AprilTagDetectionArray
from geometry_msgs.msg import PoseWithCovarianceStamped, Point, Quaternion, TransformStamped, Vector3
from tf.transformations import quaternion_from_euler
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker

# 定义标签ID、大小和位置
tag_id = 0
tag_size = 0.05
tag_position = Point(0.42, -0.20, 0.15)
tag_orientation = Quaternion(*quaternion_from_euler(0, 0, 0))
###########################

def create_mock_detection():
    # 创建篮子标签的检测数据
    tag = AprilTagDetection()
    tag.id.append(tag_id)
    tag.size = [tag_size]
    pose = PoseWithCovarianceStamped()
    pose.header.stamp = rospy.Time.now()
    pose.header.frame_id = "base_link" 

    pose.pose.pose.position, pose.pose.pose.orientation = tag_position, tag_orientation
    tag.pose = pose
 
    # 构造消息
    msg = AprilTagDetectionArray()
    msg.detections.append(tag)
    return msg

def create_and_publish_marker(marker_pub):
    # 创建marker消息
    marker = Marker()
    marker.header.frame_id = "base_link"
    marker.header.stamp = rospy.Time.now()
    marker.ns = "apriltag_markers"
    marker.id = 0
    marker.type = Marker.CYLINDER
    marker.action = Marker.ADD
    
    # 设置圆柱体的颜色和尺寸
    marker.color.a = 1.0
    marker.color.r = 1.0
    marker.color.g = 0.0
    marker.color.b = 0.0
    marker_height = 0.15
    marker.scale = Vector3(
        x=0.04,  # 半径
        y=0.04,  # 半径
        z=marker_height
    )

    # 设置圆柱体的位置
    marker.pose.position = copy.deepcopy(tag_position)
    marker.pose.position.z -= marker_height/2
    marker.pose.orientation = tag_orientation

    # 发布marker消息
    marker_pub.publish(marker)

def broadcast_transforms():
    """
        RVIZ TF 查看
    """
    br = TransformBroadcaster()
    t = TransformStamped()
    t.header.stamp = rospy.Time.now()
    t.header.frame_id = "base_link"  # 设置父坐标系
    t.child_frame_id = f"tag_{tag_id}"  # 子坐标系为 tag_加上id
    t.transform.translation.x = tag_position.x
    t.transform.translation.y = tag_position.y
    t.transform.translation.z = tag_position.z
    t.transform.rotation = tag_orientation
    br.sendTransform(t)


def mock_publisher():
    # 初始化ROS节点
    rospy.init_node('mock_apriltag_publisher', anonymous=True)
    pub = rospy.Publisher('/robot_tag_info', AprilTagDetectionArray, queue_size=10)
    marker_pub = rospy.Publisher('/visualization_marker', Marker, queue_size=10)

    rate = rospy.Rate(10)  # 设置循环频率为10Hz
    while not rospy.is_shutdown():
        broadcast_transforms()
        msg = create_mock_detection() 
        create_and_publish_marker(marker_pub)
        if msg:
            pub.publish(msg)
        rate.sleep()

if __name__ == '__main__':
    try:
        mock_publisher()
    except rospy.ROSInterruptException:
        pass