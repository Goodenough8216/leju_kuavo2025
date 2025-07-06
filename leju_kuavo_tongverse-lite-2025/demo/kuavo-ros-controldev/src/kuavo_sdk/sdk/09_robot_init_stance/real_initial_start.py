import rospy
from std_srvs.srv import Trigger, TriggerResponse


def call_init_trigger_service():
    """
    调用初始化触发服务。
    """
    try:
        # 初始化服务代理，用于触发初始化服务
        trigger_init_service = rospy.ServiceProxy('/humanoid_controller/real_initial_start', Trigger)
        # 调用服务并获取响应
        response = trigger_init_service()

        # 打印服务响应
        if response.success:
            rospy.loginfo("服务调用成功: %s", response.message)
        else:
            rospy.logwarn("服务调用失败: %s", response.message)

    except rospy.ServiceException as e:
        # 记录错误日志
        rospy.logerr("服务调用失败: %s", e)

def main():
    # 初始化ROS节点
    rospy.init_node('init_trigger_service_caller')

    # 调用初始化触发服务
    call_init_trigger_service()


if __name__ == "__main__":
    main()