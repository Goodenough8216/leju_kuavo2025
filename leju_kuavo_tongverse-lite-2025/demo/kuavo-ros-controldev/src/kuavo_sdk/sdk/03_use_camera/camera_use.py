#!/usr/bin/env python

# 导入必要的库
import rospy  # ROS的Python接口
from sensor_msgs.msg import Image  # ROS中用于图像数据的消息类型
from cv_bridge import CvBridge, CvBridgeError  # 用于在ROS图像消息和OpenCV图像之间转换
import cv2  # OpenCV库，用于图像处理


def image_callback(msg):
    """
    回调函数，当接收到图像消息时调用。
    :param msg: ROS图像消息
    """
    bridge = CvBridge()  # 创建CvBridge对象，用于图像格式转换

    try:
        # 将ROS图像消息转换为OpenCV格式的图像
        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
    except CvBridgeError as e:
        # 如果转换失败，记录错误信息
        rospy.logerr("CvBridge Error: {0}".format(e))
        return

    # 显示图像窗口，窗口名为"Camera Image"
    cv2.imshow("Camera Image", cv_image)
    # 等待1毫秒以处理窗口事件
    cv2.waitKey(1)

def main():
    """
    主函数，初始化ROS节点并订阅图像话题。
    """
    # 初始化ROS节点，节点名为'image_subscriber'
    rospy.init_node('image_subscriber', anonymous=True)
    # 订阅图像话题，当接收到消息时调用image_callback函数
    rospy.Subscriber("/camera/color/image_raw", Image, image_callback)

    # 保持节点运行，直到节点被关闭
    rospy.spin()


if __name__ == '__main__':
    # 如果此脚本是主程序，则调用main函数
    main()