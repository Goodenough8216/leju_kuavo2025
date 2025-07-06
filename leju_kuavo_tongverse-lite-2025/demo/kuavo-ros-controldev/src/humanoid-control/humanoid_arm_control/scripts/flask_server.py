#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from flask import Flask, request, jsonify
from humanoid_arm_control.srv import armControl

app = Flask(__name__)
app.debug = True

# 初始化 ROS 节点
rospy.init_node('flask_ros_server', anonymous=True)

# 等待 /arm_control 服务启动
rospy.wait_for_service('/arm_control')
arm_control_service = rospy.ServiceProxy('/arm_control', armControl)

# 将命令映射为 ROS 服务的请求值
command_mapping = {
    "left": 1,
    "right": 2,
    "middle": 0
}

def call_arm_control(req):
    try:
        response = arm_control_service(req)  # 向ROS服务发送请求
        return response.success, response.time
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")
        return False, 0.0

@app.route('/humanoid_arm_control', methods=['POST'])
def control():
    data = request.json  # 获取POST请求中的JSON数据
    command = data.get("command")
    
    if command not in command_mapping:
        return jsonify({"error": "Invalid command"}), 400

    # 根据 command 映射为相应的 int64 值
    req = command_mapping[command]

    # 调用 ROS 服务
    success, duration = call_arm_control(req)

    # 如果成功，返回message和duration
    if success:
        return jsonify({"message": "Command executed successfully", "duration": duration}), 200
    # 如果不成功，返回message和501
    else:
        return jsonify({"message": "Failed to execute command"}), 501

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8031)
