#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import csv
import math
import os
import sys
import rospy
import rosbag
from std_msgs.msg import String, Int32
from kuavo_msgs.msg import sensorsData, jointData, imuData
from geometry_msgs.msg import Vector3, Quaternion
from std_msgs.msg import Float32MultiArray

mode_map ={
    "SS": 15,
    "SF":12,
    "FS":3
}
def generate_default_output_path(csv_file):
    base, ext = os.path.splitext(csv_file)
    return base + '_expanded.bag'
def csv_to_rosbag(csv_file, bag_file):
    # 初始化ROS节点
    # rospy.init_node('csv_to_bag', anonymous=True)
    bag_file = bag_file if bag_file else generate_default_output_path(csv_file)
    id = 0

    data_cache = {
        'sensors/joint/v/': {}, 
        'sensors/joint/q/': {}, 
        'sensors/joint/vdot/': {}, 
        'sensors/imu/acc/': {}, 
        'sensors/imu/gyro/': {}, 
        'sensors/imu/free_acc/': {}, 
        'sensors/imu/quat/': {}, 
        'wbc/tau/': {}, 
        'state/q/': {}, 
        'state/v/': {},
        'desire/walk_contact/': {},
    }
    # 创建bag文件
    with rosbag.Bag(bag_file, 'w') as bag:
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)  # 读取表头
            start_time = 0.0
            for row in reader:
                id += 1
                if id % 1000 == 0:
                    print(f"Processed {id} lines", end='\r')
                if id == 1:
                    start_time = float(row[0])
                timestamp = rospy.Time.from_sec((float(row[0])-start_time)/1000.0)
                
                for i, name in enumerate(headers[1:], start=1):
                    # 分离出消息类型和字段名
                    if 'data.' not in name or len(row[i]) == 0:
                        continue
                    base_name, index = name.rsplit('data.', 1)
                    if base_name not in data_cache.keys():
                        continue
                    data_cache[base_name][int(index)] = float(row[i])
                    
                if all(len(data_cache[key]) > 0 for key in data_cache):
                    sensors_msg = sensorsData()
                    joint_msg = jointData()
                    imu_msg = imuData()

                    tau_msg = Float32MultiArray()
                    qv_msg = Float32MultiArray()
                    walk_contact_msg = Int32()

                    for base_name, values in data_cache.items():
                        sorted_values = [values[i] for i in sorted(values)]
                        if base_name == 'sensors/joint/v/':
                            joint_msg.joint_v = [value*math.pi/180.0 for value in sorted_values]
                        elif base_name == 'sensors/joint/q/':
                            joint_msg.joint_q = [value*math.pi/180.0 for value in sorted_values]
                        elif base_name == 'sensors/joint/vdot/':
                            joint_msg.joint_vd = [value*math.pi/180.0 for value in sorted_values]
                        elif base_name == 'sensors/imu/acc/':
                            imu_msg.acc = Vector3(*sorted_values)
                        elif base_name == 'sensors/imu/gyro/':
                            imu_msg.gyro = Vector3(*sorted_values)
                        elif base_name == 'sensors/imu/free_acc/':
                            imu_msg.free_acc = Vector3(*sorted_values)
                        elif base_name == 'sensors/imu/quat/':
                            imu_msg.quat = Quaternion(*sorted_values)
                        elif base_name == 'wbc/tau/':
                            tau_msg.data = sorted_values
                            joint_msg.joint_current = sorted_values
                        elif base_name == 'desire/walk_contact/':
                            contact_name = 'SF' if float(sorted_values[0]) > 0.1 else 'FS' if float(sorted_values[0]) < -0.1 else 'SS' 
                            walk_contact_msg.data = mode_map[contact_name]

                    q = data_cache['state/q/']
                    v = data_cache['state/v/']
                    sorted_q = [q[i] for i in sorted(q)]
                    sorted_v = [v[i] for i in sorted(v)]
                    qv_msg.data = sorted_q + sorted_v

                    # 将joint_msg和imu_msg放入sensors_msg中
                    sensors_msg.joint_data = joint_msg
                    sensors_msg.imu_data = imu_msg
                    sensors_msg.sensor_time = timestamp
                    
                    bag.write('sensors_data_raw', sensors_msg, t=timestamp)
                    bag.write('input_tau', tau_msg, t=timestamp)
                    bag.write("qv", qv_msg, t=timestamp)
                    bag.write("mode", walk_contact_msg, t=timestamp)
                    
                    # 清空缓存数据
                    data_cache = {key: {} for key in data_cache}

    print(f"Converted {csv_file} to {bag_file}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert CSV log to ROS bag.")
    parser.add_argument('-i', '--input_path', type=str, help='Path to the input CSV file.')
    parser.add_argument('-o', '--output_path', type=str, help='Path to the output ROS bag file.')

    args = parser.parse_args()

    if not args.input_path:
        parser.print_help()
        exit(1)

    csv_to_rosbag(args.input_path, args.output_path)
