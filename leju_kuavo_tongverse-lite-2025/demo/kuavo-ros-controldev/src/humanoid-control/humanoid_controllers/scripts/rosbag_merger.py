#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os
import rosbag
import rospy
import glob
from datetime import datetime
import multiprocessing
from functools import partial
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
def parse_bag_filename(filename):
    """Parse bag filename to extract timestamp and index."""
    base = os.path.basename(filename)
    try:
        timestamp_str, index_str = base.split('_')
        timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d-%H-%M-%S')
        index = int(index_str.split('.')[0])
        return timestamp, index
    except (ValueError, IndexError):
        print(f"Warning: Ignoring file '{filename}' due to incorrect format.")
        return None, None
def find_consecutive_bags(start_file):
    """Find consecutive bag files starting from the given file."""
    directory = os.path.dirname(start_file)
    start_timestamp, start_index = parse_bag_filename(start_file)

    if start_timestamp is None or start_index is None:
        return []

    all_files = glob.glob(os.path.join(directory, "*.bag"))
    all_files_info = [(f, *parse_bag_filename(f)) for f in all_files]
    
    # 过滤掉 None 值
    all_files_info = [info for info in all_files_info if info[1] is not None]

    all_files_info.sort(key=lambda x: (x[1], x[2]))  # Sort by timestamp first, then by index
    
    consecutive_files = [start_file]
    current_timestamp, current_index = start_timestamp, start_index
    
    start_found = False
    for file, timestamp, index in all_files_info:
        if file == start_file:
            start_found = True
            continue
        if not start_found:
            continue
        if timestamp >= current_timestamp and index == current_index + 1:
            consecutive_files.append(file)
            current_timestamp, current_index = timestamp, index
        elif timestamp > current_timestamp:
            break
    
    return consecutive_files

def merge_bags(input_files, output_file):
    print(f"Merging {len(input_files)} bag files into {output_file}")
    with rosbag.Bag(output_file, 'w') as outbag:
        for i, input_file in enumerate(input_files):
            print(f"Processing file {i+1}/{len(input_files)}: {input_file}")
            with rosbag.Bag(input_file, 'r') as inbag:
                for topic, msg, t in inbag.read_messages():
                    outbag.write(topic, msg, t)
    print(f"\nMerged {len(input_files)} bag files into {output_file}")
    print("Merging completed successfully!")
    
def generate_default_output_name(input_files):
    if not input_files:
        return 'merged.bag'
    
    first_file = input_files[0]
    last_file = input_files[-1]
    
    first_timestamp, first_index = parse_bag_filename(first_file)
    last_timestamp, last_index = parse_bag_filename(last_file)
    output_dir = os.path.dirname(first_file)

    output_name = f"{first_timestamp.strftime('%Y-%m-%d-%H-%M-%S')}_{first_index}_to_{last_index}.bag"
    full_output_path = os.path.join(output_dir, output_name)
    return full_output_path

def main():
    parser = argparse.ArgumentParser(description="Merge multiple ROS bag files.")
    parser.add_argument('-i', '--input', type=str, nargs='+', help='Input bag files or patterns (e.g., "*.bag")')
    parser.add_argument('-o', '--output', type=str, default='notdefined.bag', help='Output bag file name')
    parser.add_argument('-f', '--first', type=str, help='First file to start merging from')
    
    args = parser.parse_args()

    input_files = []

    if args.first:
        print(f"Starting from {args.first}")
        # If a starting file is specified, find all subsequent files
        input_files = find_consecutive_bags(args.first)
        print(f"Found {len(input_files)} consecutive bag files starting from {args.first}")
    elif args.input:
        # If input patterns are provided, use them to find files
        for pattern in args.input:
            if '*' in pattern or '?' in pattern:
                input_files.extend(glob.glob(pattern))
            else:
                input_files.append(pattern)
        print(f"Found {len(input_files)} bag files matching the input pattern(s)")
    else:
        print("Error: Either -i (input files/patterns) or -f (first file) must be specified.")
        return
    print("Input files:")
    for i, f in enumerate(input_files):
        print(f"{i+1}. {f}")
    if not input_files:
        print("No input files found.")
        return
    if args.output == 'notdefined.bag':
        args.output = generate_default_output_name(input_files)
        print(f"Output file default: {args.output}")
    # Merge the bags
    merge_bags(input_files, args.output)

if __name__ == "__main__":
    main()
