###
 # @Author: dongdongmingming
 # @Date: 2024-04-18 14:45:05
 # @LastEditors: Please set LastEditors
 # @LastEditTime: 2024-04-20 14:48:38
 # @FilePath: /kuavo/tools/check_tool/compile_kuavo.sh
 # @Description: 编译当前目录
### 


#!/bin/bash




# 获取当前脚本所在文件夹的绝对路径
current_script_dir=$(dirname "$(realpath "$0")")


# 使用字符串操作来截取前部分路径
prefix="${current_script_dir%/tools/check_tool}"

rm -rf $prefix/build
rm -rf $prefix/devel


cd $prefix
catkin config -DCMAKE_ASM_COMPILER=/usr/bin/as -DCMAKE_BUILD_TYPE=Release # Important!
catkin build  humanoid_controllers
