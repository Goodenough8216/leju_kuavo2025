#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os
import rospy
import rosbag
import numpy as np
from std_msgs.msg import Float64MultiArray
from scipy import signal

# 全局变量用于保存读取的消息、时间和topic的映射关系
global_bag_data = []

def get_latest_bag_file(folder_path):
    folder_path = os.path.expanduser(folder_path)
    bag_files = [f for f in os.listdir(folder_path) if f.endswith(".bag") or f.endswith(".bag.active")]
    bag_files.sort(key=lambda f: os.path.getmtime(os.path.join(folder_path, f)), reverse=True)
    
    if bag_files:
        return os.path.join(folder_path, bag_files[0])
    else:
        return None

def read_bag_data(bag_path):
    """
    从指定的bag文件读取所有数据并保存在全局变量中
    """
    global global_bag_data
    global_bag_data = []

    try:
        with rosbag.Bag(bag_path, 'r') as bag:
            for topic, msg, t in bag.read_messages():
                # 保存数据、topic名字和时间戳到二维列表中
                global_bag_data.append([topic, msg, t.to_sec()])
    except Exception as e:
        print(f"Error reading bag file: {e}")

def second_order_lowpass(data, cutoff, fs, damping_ratio=0.6):
    print(f"Filtering data with cutoff frequency {cutoff} Hz and damping ratio {damping_ratio}")
    natural_freq = 2 * np.pi * cutoff
    Q = 1 / (2 * damping_ratio)
    w0 = cutoff / (fs / 2)
    b, a = signal.iirfilter(2, w0, btype='low', analog=False, ftype='butter', output='ba', fs=None)
    filtered_data = signal.filtfilt(b, a, data, axis=0)
    return filtered_data
# def second_order_lowpass(data, cutoff_freq, sampling_rate,damping_ratio=0.6):
#     # 计算自然频率 (弧度/秒)
#     omega_n = 2 * np.pi * cutoff_freq
    
#     # 系数化为离散域 (s-domain -> z-domain) 通过双线性变换
#     num = [omega_n**2]  # 分子
#     den = [1, 2*damping_ratio*omega_n, omega_n**2]  # 分母
    
#     # 将连续时间传递函数转换为离散时间
#     system = signal.TransferFunction(num, den)
#     discrete_system = system.to_discrete(1 / sampling_rate, method='bilinear')

#     # 使用滤波器对信号进行滤波
#     filtered_data = signal.lfilter(discrete_system.num, discrete_system.den, data)
#     return filtered_data


def filter_and_insert_to_global_data(cutoff_freq, sampling_rate, damping_ratio=0.6):
    """
    对某个topic的数据进行滤波处理，并将处理后的数据添加到global_bag_data中，保留原时间戳
    """
    global global_bag_data
    original_topic = "/joint_cmd"
    filtered_topic = "/cmd_joint_v_filtered"
    # 从全局数据中筛选出目标topic的消息
    original_data = [entry for entry in global_bag_data if entry[0] == original_topic]
    
    if not original_data:
        print(f"No data found for topic '{original_topic}'")
        return

    # 提取数据和时间戳
    data = []
    timestamps = []
    for entry in original_data:
        joint_v = entry[1].joint_v
        data.append(joint_v)
        timestamps.append(entry[2])

    data = np.array(data)

    # 对数据进行二阶低通滤波
    filtered_data = second_order_lowpass(data, cutoff_freq, sampling_rate, damping_ratio)

    # 将滤波后的数据与原时间戳添加到global_bag_data中，并标记为新topic
    for i, filtered_row in enumerate(filtered_data):
        filtered_msg = Float64MultiArray()
        filtered_msg.data = filtered_row.tolist()

        global_bag_data.append([filtered_topic, filtered_msg, timestamps[i]])

def write_sorted_data_to_bag(output_bag_path):
    """
    将global_bag_data中的所有数据按时间戳排序后，写入到新的bag文件中
    """
    global global_bag_data

    # 按时间戳排序
    global_bag_data.sort(key=lambda x: x[2])
    
    length = len(global_bag_data)
    index = 0
    try:
        with rosbag.Bag(output_bag_path, 'w') as outbag:
            for entry in global_bag_data:
                topic = entry[0]
                msg = entry[1]
                timestamp = rospy.Time.from_sec(entry[2])
                # if topic not in ["/joint_cmd", "/cmd_joint_v_filtered", "/sensors_data_raw"]:
                #     continue
                # 写入新的bag文件
                outbag.write(topic, msg, timestamp)
                index += 1
                if index % 1000 == 0 or index == length-1:
                    print(f"writing {index+1}/{length} messages to bag file...", end="\r")
        
        print(f"All data written to bag file '{output_bag_path}' in sorted order.")
    
    except Exception as e:
        print(f"Error writing to bag file: {e}")

def main():
    parser = argparse.ArgumentParser(description="Process ROS bag files, apply lowpass filter on a specific topic, and write the results to a new topic.")
    parser.add_argument('-i', '--input_path', type=str, help='Path to the input bag file.')
    parser.add_argument('-s', '--sampling_rate', type=float, required=True, help='Sampling rate of the data.')
    parser.add_argument('-c', '--cutoff_frequency', type=float, required=True, help='Cutoff frequency for lowpass filter.')
    parser.add_argument('-d', '--damping_ratio', type=float, default=0.6, help='Damping ratio for the filter (default: 0.6).')
    parser.add_argument('-o', '--output_bag', type=str, default="filtered_data.bag", help='Path to save the output bag file with filtered data.')

    args = parser.parse_args()

    if args.input_path is None:
        print("No input bag file specified, using latest bag file in ~/.ros")
        args.input_path = get_latest_bag_file("~/.ros")
        print(f"Found latest bag file: {args.input_path}")
        assert args.input_path is not None, "No input bag file specified and no bag file found in ~/.ros"

    # 读取bag文件中的所有数据
    print(f"Reading bag file '{args.input_path}'...")
    read_bag_data(args.input_path)
    print(f"Read {len(global_bag_data)} messages from bag file.")
    # 对指定的topic进行滤波并将结果添加到global_bag_data中
    filter_and_insert_to_global_data(args.cutoff_frequency, args.sampling_rate, args.damping_ratio)
    print(f"Filtered data added to global_bag_data.writing to {args.output_bag}")
    # 按时间戳排序并写入新的bag文件
    write_sorted_data_to_bag(args.output_bag)
    print("Done.")

if __name__ == "__main__":
    main()
