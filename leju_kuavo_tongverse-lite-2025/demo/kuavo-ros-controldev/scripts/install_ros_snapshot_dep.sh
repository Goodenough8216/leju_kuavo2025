#!/bin/bash

sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key 4B63CF8FDE49746E98FA01DDAD19BAB3CBF125EA

# snapshot 日期根据需要修改
sudo sh -c 'echo "deb http://snapshots.ros.org/noetic/2024-10-31/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-snapshots.list'

sudo apt-get update

sudo apt install ros-noetic-pinocchio=2.6.21-1focal.20240830.092123 --reinstall -y
