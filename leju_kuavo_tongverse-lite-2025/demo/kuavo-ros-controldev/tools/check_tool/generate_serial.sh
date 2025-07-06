#!/bin/bash
###
 # @Author: dongdongmingming
 # @Date: 2024-05-17 09:56:55
 # @LastEditors: Please set LastEditors
 # @LastEditTime: 2024-05-17 10:00:24
 # @FilePath: /kuavo/tools/check_tool/generate_serial.sh
 # @Description: 将指定左右强脑手 USB 设备生成 udev 规则
### 


# 检查是否确实传入了两个参数
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 arg1 arg2"
    exit 1
fi


echo "脚本名称是: $0"

# 打印第一个参数
echo "第一个参数是: $1"

# 打印第二个参数
echo "第二个参数是: $2"


dev=$1

rule_name=$2


echo "你选择的串口为: $dev"

echo "自定义的串口名为: $rule_name"


serial=$(udevadm info --attribute-walk --name=$dev|grep ATTRS{serial} | cut -d= -f3 | sed 's/"//g'|head -n 1)
# 如果属性不为空，创建一个udev规则文件
if [ -n "$serial" ]; then
echo '正在为序列号为'$serial'的设备生成udev规则...'
echo 'KERNEL=="ttyUSB*", ATTRS{serial}=="'$serial'", MODE:="0777", SYMLINK+="'$rule_name'"' > /etc/udev/rules.d/$rule_name.rules
echo '生成成功! 请重启计算机或者插拔设备以使规则生效。'
else 
echo '未找到序列号，请检查设备是否已连接。'
fi

sleep 1


# 重新加载udev规则
udevadm control --reload-rules
udevadm trigger