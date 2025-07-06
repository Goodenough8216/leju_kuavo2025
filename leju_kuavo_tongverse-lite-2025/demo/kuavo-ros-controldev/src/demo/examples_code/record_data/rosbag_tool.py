#!/usr/bin/env python3

import os
import subprocess
import sys

try:
    import questionary
    from rich.console import Console
except ImportError:
    print("Installing required packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "questionary", "rich"])
    import questionary
    from rich.console import Console

import datetime
import json


console = Console()
rosbag_log_save_path = "~/.log/vr_remote_control/rosbag"

record_topics_path = os.path.join(os.path.dirname(__file__), "record_topics.json")

with open(record_topics_path, "r") as f:
    config = json.load(f)


def record_rosbag(config):
    try:
        base_dir = os.path.expanduser(rosbag_log_save_path)
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        date_folder = os.path.join(base_dir, current_date)
        if not os.path.exists(date_folder):
            os.makedirs(date_folder)
        
        base_filename = "vr_record"
        bag_file_base = os.path.join(date_folder, base_filename)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        actual_bag_file = f"{bag_file_base}_{timestamp}.bag"  # 实际会被创建的文件路径
        
        command = [
            "rosbag",
            "record",
            "-o",
            bag_file_base  # rosbag 会自动添加时间戳
        ]

        recorded_topics = []
        if config["record_topics"] == []:
            command.append("-a")  # all topics
            recorded_topics.append("all topics")
        else:
            for topic in config["record_topics"]:
                command.append(topic)
                recorded_topics.append(topic)

        process = subprocess.Popen(
            command,
            start_new_session=True,
        )
        console.print("[green]Recording started... Press Ctrl+C to stop.[/green]")
        process.wait()
        
        console.print(f"[blue]Bag file saved at: {actual_bag_file}[/blue]")
    except KeyboardInterrupt:
        process.terminate()
        console.print("[red]Recording stopped.[/red]")
        console.print(f"[blue]Bag file saved at: {actual_bag_file}[/blue]")

def playback_rosbag():
    base_dir = os.path.expanduser(rosbag_log_save_path)
    date_folders = [folder for folder in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, folder))]
    
    if not date_folders:
        console.print("[red]No date folders found.[/red]")
        return

    selected_date = questionary.select("请选择一个日期文件夹", choices=date_folders).ask()
    if selected_date is None:
        return
    date_folder_path = os.path.join(base_dir, selected_date)
    rosbag_files = [file for file in os.listdir(date_folder_path) if file.endswith(".bag")]

    if not rosbag_files:
        console.print("[red]No rosbag files found in the selected date folder.[/red]")
        return

    selected_rosbag = questionary.select("请选择一个rosbag文件进行回放", choices=rosbag_files).ask()
    if selected_rosbag is None:
        return
    rosbag_file_path = os.path.join(date_folder_path, selected_rosbag)

    command = ["rosbag", "play", rosbag_file_path]
    subprocess.Popen(command)
    console.print(f"[green]Playing back rosbag: {rosbag_file_path}[/green]")

def main():
    
    action = questionary.select("请选择一个操作", choices=["录制 VR 手臂数据和相机数据 ROSBAG", "回放 VR 手臂数据和相机数据 ROSBAG"]).ask()

    if action == "录制 VR 手臂数据和相机数据 ROSBAG":
        record_rosbag(config)
    elif action == "回放 VR 手臂数据和相机数据 ROSBAG":
        playback_rosbag()


if __name__ == "__main__":
    main()