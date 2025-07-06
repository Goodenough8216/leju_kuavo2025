###
 # @Author: dongdongmingming
 # @Date: 2024-04-18 14:45:05
 # @LastEditors: Please set LastEditors
 # @LastEditTime: 2024-04-18 16:56:38
 # @FilePath: /kuavo/tools/check_tool/folder_backups.sh
 # @Description: 保存配置文件同步到 kuavo_opensource   新配置文件保存在 /home/lab/.config/lejuconfig  不再需要备份
### 

#!/bin/bash



# 获取当前脚本所在文件夹的绝对路径
current_script_dir=$(dirname "$(realpath "$0")")


# 使用字符串操作来截取前部分路径
prefix="${current_script_dir%/tools/check_tool}"




source_offset_dir=/home/lab/.config/lejuconfig/offset.csv


source_config_dir="/home/lab/.config/lejuconfig/"



# ----------------------------------- 打包备份 ----------------------------------- #
# 定义要打包的文件和目录列表
files=(
    $source_config_dir
    $source_urdf_dir
)

# 定义压缩文件的名称
zipfile=/home/lab/confirm_backups.zip

# 打包文件
zip -r $zipfile "${files[@]}"


rm -r /home/lab/ssh/*

# 提示打包完成
echo "文件已打包为 $zipfile ，已删除 ssh key"





