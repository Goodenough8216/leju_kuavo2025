###
 # @Author: dongdongmingming
 # @Date: 2024-12-18 15:46:28
 # @LastEditors: Please set LastEditors
 # @LastEditTime: 2024-12-20 14:53:53
 # @FilePath: /kuavo/tools/check_tool/ecMACget.sh
 # @Description: ec 获取 MAC 地址
### 

#!/bin/bash


# 获取当前脚本所在文件夹的绝对路径
current_script_dir=$(dirname "$(realpath "$0")")


# 使用字符串操作来截取前部分路径
kuavo_dir="${current_script_dir%/tools/check_tool}"


# 切换到仓库目录
cd ${current_script_dir}



 ./EcDemo -i8254x 1 1 -f ../config/elmo_1_eni.xml -auxclk 500 -dcmmode mastershift -t 0  > output_file.txt
