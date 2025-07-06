#!/usr/bin/env python3
# coding=utf-8

import rospy
import numpy as np
from apriltag_ros.msg import AprilTagDetectionArray, AprilTagDetection
from geometry_msgs.msg import PoseWithCovarianceStamped, Pose, Point, Quaternion
from nav_msgs.msg import Odometry
from tf.transformations import quaternion_matrix, quaternion_multiply, quaternion_inverse
from visualization_msgs.msg import Marker
from kuavo_msgs.srv import SetTagPose, SetTagPoseResponse  # 需要创建这个服务消息类型


class AprilTagPublisher:
    def __init__(self):
        rospy.init_node('apriltag_publisher', anonymous=True)
        self.pub = rospy.Publisher('/robot_tag_info', AprilTagDetectionArray, queue_size=10)
        self.marker_pub = rospy.Publisher('/visualization_box_marker', Marker, queue_size=10)
        self.rate = rospy.Rate(10)  # Publish frequency: 10Hz

        self.robot_pose = None
        self.tag_poses_world = [
            {   # position=Point(2.74896, 0.51245, 0.78557),
                "id": 1, # 取
                "pose": Pose(
                    position=Point(1.15, 0.00245, 0.78),
                    orientation=Quaternion(0.0, -0.707, 0.0, 0.707)
                ),
                "size": 0.06,
            },
            {
                "id": 2, # 放
                "pose": Pose(
                    position=Point(-1.004896, 2.00245, 0.78),
                    orientation=Quaternion(0.707, 0.0, 0.707, 0 )
                ),
                "size": 0.06,
            },
            {
                "id": 0, # 无作用
                "pose": Pose(
                    position=Point(0.16089, 0.00148, 1.5434),
                    orientation=Quaternion(0.0, -0.707, 0.0, 0.707)
                    # w=0.5,x=−0.5,y=−0.5,z=0.5
                ),
                "size": 0.06,
            },
        ]

        # 添加服务
        self.set_tag_pose_srv = rospy.Service('set_tag_pose', SetTagPose, self.set_tag_pose_callback)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)

    def set_tag_pose_callback(self, req):
        """处理设置tag位置的服务请求"""
        response = SetTagPoseResponse()
        try:
            # 查找对应id的tag
            for tag in self.tag_poses_world:
                if tag["id"] == req.tag_id:
                    # 更新位置
                    tag["pose"].position = req.pose.position
                    tag["pose"].orientation = req.pose.orientation
                    response.success = True
                    response.message = f"Successfully updated tag {req.tag_id} pose"
                    print(f"successfully updated tag {req.tag_id} pose to {tag['pose'].position}\n{tag['pose'].orientation}")
                    print(f"tag_poses_world: {self.tag_poses_world}")
                    return response
            
            # 如果没找到对应的tag
            response.success = False
            response.message = f"Tag ID {req.tag_id} not found"
            return response
            
        except Exception as e:
            response.success = False
            response.message = f"Failed to update tag pose: {str(e)}"
            return response

    def odom_callback(self, msg):
        self.robot_pose = msg.pose.pose

    def publish_marker(self, tag_pose_world, tag_id):
        marker = Marker()
        marker.header.frame_id = 'odom'
        marker.header.stamp = rospy.Time.now()
        marker.ns = 'apriltag'
        marker.id = tag_id
        marker.type = Marker.CUBE
        marker.action = Marker.ADD

        marker.pose = tag_pose_world

        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1

        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        self.marker_pub.publish(marker)

    def get_transformed_tag_pose(self, tag_pose_world, robot_pose):
        robot_position = np.array([robot_pose.position.x, robot_pose.position.y, robot_pose.position.z])
        robot_orientation = np.array([
            robot_pose.orientation.x, robot_pose.orientation.y,
            robot_pose.orientation.z, robot_pose.orientation.w
        ])
        tag_position = np.array([tag_pose_world.position.x, tag_pose_world.position.y, tag_pose_world.position.z])
        tag_orientation = np.array([
            tag_pose_world.orientation.x, tag_pose_world.orientation.y,
            tag_pose_world.orientation.z, tag_pose_world.orientation.w
        ])

        relative_position = tag_position - robot_position
        rotation_matrix = quaternion_matrix(quaternion_inverse(robot_orientation))[:3, :3]
        transformed_position = rotation_matrix.dot(relative_position)

        tag_pose_relative = Pose()
        tag_pose_relative.position.x = transformed_position[0]
        tag_pose_relative.position.y = transformed_position[1]
        tag_pose_relative.position.z = transformed_position[2]

        relative_orientation = quaternion_multiply(quaternion_inverse(robot_orientation), tag_orientation)
        tag_pose_relative.orientation.x = relative_orientation[0]
        tag_pose_relative.orientation.y = relative_orientation[1]
        tag_pose_relative.orientation.z = relative_orientation[2]
        tag_pose_relative.orientation.w = relative_orientation[3]

        return tag_pose_relative

    def run(self):
        while not rospy.is_shutdown():
            if self.robot_pose is None:
                rospy.logwarn("Waiting for robot pose data...")
                rospy.sleep(0.1)
                continue

            tag_detection_array = AprilTagDetectionArray()

            for tag_info in self.tag_poses_world:
                tag_id = tag_info["id"]
                tag_pose_world = tag_info["pose"]
                tag_size = tag_info["size"]

                tag_pose_relative = self.get_transformed_tag_pose(tag_pose_world, self.robot_pose)

                detection = AprilTagDetection()
                detection.id = [tag_id]
                detection.size = [tag_size]

                pose = PoseWithCovarianceStamped()
                pose.header.stamp = rospy.Time.now()
                pose.header.frame_id = 'odom'
                pose.pose.pose = tag_pose_relative
                pose.pose.covariance = [0] * 36

                detection.pose = pose
                tag_detection_array.detections.append(detection)

                self.publish_marker(tag_pose_world, tag_id)

            self.pub.publish(tag_detection_array)
            self.rate.sleep()


if __name__ == '__main__':
    try:
        publisher = AprilTagPublisher()
        publisher.run()
    except rospy.ROSInterruptException:
        pass
