import rospy
from kuavo_sdk.srv import recordmusic, recordmusicRequest, recordmusicResponse


def srv_recordmusic_call(music_file: str, time_out: int) -> bool:
    """
    调用录制音乐服务。

    参数:
    music_file (str): 音乐文件编号。
    time_out (int): 超时时间。

    返回:
    bool: 服务调用结果，成功返回True，失败返回False。
    """
    try:
        # 初始化服务代理，用于录制音乐
        robot_record_music_client = rospy.ServiceProxy("/record_music", recordmusic)
        # 创建请求对象
        request = recordmusicRequest()
        request.music_number = music_file
        request.time_out = time_out

        # 调用服务并获取响应
        response = robot_record_music_client(request)

        # 返回结果
        return response.success_flag

    except rospy.ServiceException as e:
        # 记录错误日志
        rospy.logerr(f"Record_music_call 服务调用失败: {e}")
        return False

def main():
    # 初始化ROS节点
    rospy.init_node('music_recorder')

    # 示例音乐文件编号和超时时间
    music_file = "example_music"
    time_out = 30

    # 调用服务函数
    result = srv_recordmusic_call(music_file, time_out)

    # 输出结果
    if result:
        rospy.loginfo("音乐录制成功")
    else:
        rospy.loginfo("音乐录制失败")


if __name__ == "__main__":
    main()