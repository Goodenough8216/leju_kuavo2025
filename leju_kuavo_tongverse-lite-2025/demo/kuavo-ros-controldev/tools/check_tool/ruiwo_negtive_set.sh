#!/bin/bash
###
 # @Author: dongdongmingming
 # @Date: 2024-05-17 09:56:55
 # @LastEditors: Please set LastEditors
 # @LastEditTime: 2024-05-17 10:00:24
 # @FilePath: /kuavo/tools/check_tool/ruiwo_negtive_set.sh
 # @Description: 瑞沃电机设置方向
### 


# 获取当前脚本所在文件夹的绝对路径
current_script_dir=$(dirname "$(realpath "$0")")


# 使用字符串操作来截取前部分路径
prefix="${current_script_dir%/tools/check_tool}"



kuavo_ros_folder_path="$prefix/src/kuavo-ros-control-lejulib/hardware_node/lib/ruiwo_controller"
kuavo_ros_file_path="$kuavo_ros_folder_path/Negtive.sh"

kuavo_open_folder_path="$prefix/installed/share/hardware_node/lib/ruiwo_controller"
kuavo_open_file_path="$kuavo_open_folder_path/Negtive.sh"

# 检查文件是否存在
if [ -f "$kuavo_ros_file_path" ]; then
    setZeroPath=$kuavo_ros_folder_path
elif [ -f "$kuavo_open_file_path" ]; then
    setZeroPath=$kuavo_open_folder_path
else
    echo "The file does not exist."
    exit 1
fi



cd $setZeroPath

./Negtive.sh
