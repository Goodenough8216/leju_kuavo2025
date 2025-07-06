#!/usr/bin/env python

import rospy
from kuavo_sdk.srv import playmusic, playmusicRequest, playmusicResponse


def srv_playmusic_call(music_file: str, music_volume: int) -> bool:
    """
    调用 playmusic 服务以播放音乐。

    参数:
    music_file (str): 音乐文件的名称或编号。
    music_volume (int): 音乐的音量。

    返回:
    bool: 服务调用是否成功。
    """
    # 创建服务代理
    robot_music_play_client = rospy.ServiceProxy("/play_music", playmusic)

    try:
        # 创建请求对象
        request = playmusicRequest()
        request.music_number = music_file
        request.volume = music_volume

        # 发送请求并接收响应
        response = robot_music_play_client(request)
        return response.success_flag

    except rospy.ServiceException as e:
        rospy.logerr(f"playmusic_call 服务调用失败: {e}")
        return False


if __name__ == "__main__":
    # 初始化 ROS 节点
    rospy.init_node('music_player_client')

    # 示例调用,传入路径为上位机音频文件所在路径，无需加.mp3
    music_file = "/home/kuavo/你好"
    music_volume = 80

    success = srv_playmusic_call(music_file, music_volume)

    if success:
        rospy.loginfo("音乐播放成功")
    else:
        rospy.loginfo("音乐播放失败")