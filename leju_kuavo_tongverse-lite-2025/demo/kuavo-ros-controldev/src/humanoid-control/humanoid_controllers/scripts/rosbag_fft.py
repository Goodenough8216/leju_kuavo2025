#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os
import rosbag
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from std_msgs.msg import Float32MultiArray

def get_latest_bag_file(folder_path):
    folder_path = os.path.expanduser(folder_path)
    bag_files = [f for f in os.listdir(folder_path) if f.endswith(".bag") or f.endswith(".bag.active")]
    bag_files.sort(key=lambda f: os.path.getmtime(os.path.join(folder_path, f)), reverse=True)
    
    if bag_files:
        return os.path.join(folder_path, bag_files[0])
    else:
        return None

def read_topic_data(bag_path):
    topic_name = "/sensors_data_raw"
    data = []
    try:
        with rosbag.Bag(bag_path, 'r') as bag:
            for topic, msg, t in bag.read_messages(topics=[topic_name]):
                imu_data = msg.imu_data
                acc_data = imu_data.free_acc
                data.append([acc_data.x, acc_data.y, acc_data.z])
    except Exception as e:
        print(f"Error reading bag file: {e}")
    return data

def perform_fft(data, sampling_rate):
    data = np.array(data)
    N = data.shape[0]
    T = 1.0 / sampling_rate
    xf = np.fft.fftfreq(N, T)[:N//2]
    yfs = []
    for i in range(data.shape[1]):
        yf = np.fft.fft(data[:, i])
        yfs.append(2.0/N * np.abs(yf[:N//2]))
    return xf, yfs

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data, axis=0)
    return y

def plot_time_domain(original_data, filtered_data, num_dimensions, sampling_rate):
    t = np.arange(len(original_data)) / sampling_rate
    plt.figure(figsize=(10, 8))
    for i in range(num_dimensions):
        plt.subplot(num_dimensions, 1, i + 1)
        plt.plot(t, original_data[:, i], label='Original')
        plt.plot(t, filtered_data[:, i], label='Filtered')
        plt.title(f'Time Domain - Dimension {i + 1}')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid()
    plt.tight_layout()
    plt.show()

def plot_frequency_domain(xf, yfs, num_dimensions, filtered_yfs):
    plt.figure(figsize=(10, 8))
    for i in range(num_dimensions):
        plt.subplot(num_dimensions, 1, i + 1)
        plt.plot(xf, yfs[i], label='Original')
        plt.plot(xf, filtered_yfs[i], label='Filtered')
        plt.title(f'Frequency Domain - Dimension {i + 1}')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid()
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Process ROS bag files, perform FFT and lowpass filter on a specific topic.")
    parser.add_argument('-i', '--input_path', type=str, help='Path to the input bag file.')
    parser.add_argument('-s', '--sampling_rate', type=float, required=True, help='Sampling rate of the data.')
    parser.add_argument('-c', '--cutoff_frequency', type=float, required=True, help='Cutoff frequency for lowpass filter.')

    args = parser.parse_args()

    if args.input_path is None:
        print("No input bag file specified, using latest bag file in ~/.ros")
        args.input_path = get_latest_bag_file("~/.ros")
        print(f"Found latest bag file: {args.input_path}")
        assert args.input_path is not None, "No input bag file specified and no bag file found in ~/.ros"

    data = read_topic_data(args.input_path)
    if data:
        data = np.array(data)
        num_dimensions = data.shape[1]

        # 对每个维度的数据应用低通滤波器
        filtered_data = lowpass_filter(data, args.cutoff_frequency, args.sampling_rate)
        
        # 进行FFT
        xf, yfs = perform_fft(data, args.sampling_rate)
        _, filtered_yfs = perform_fft(filtered_data, args.sampling_rate)
        
        # 绘制时域图
        plot_time_domain(data, filtered_data, num_dimensions, args.sampling_rate)
        
        # 绘制频域图
        plot_frequency_domain(xf, yfs, num_dimensions, filtered_yfs)
    else:
        print("No data found for the specified topic.")

if __name__ == "__main__":
    main()
