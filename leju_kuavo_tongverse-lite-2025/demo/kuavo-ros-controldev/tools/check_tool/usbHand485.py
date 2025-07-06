
#!/usr/bin/env python
# coding=utf-8
'''
Author: dongdongmingming
Date: 2024-05-14 17:46:47
LastEditors: Please set LastEditors
LastEditTime: 2024-05-16 11:31:26
FilePath: \kuavo\tools\check_tool\serial_tool.py
Description: 配置绑定左右强脑手 485 usb 名称
'''


import time
import os,sys
import subprocess
import re

folder_path = os.path.dirname(os.path.abspath(__file__))    # check_tool/


sys.path.append('/home/lab/.local/lib/python3.8/site-packages/')
sys.path.append(os.path.join(folder_path,"Ruierman"))

import serial
import serial.tools.list_ports
import dynamixel_servo


import ruierman
# import claw_rs485
import subprocess


servo_usb_path = "/dev/usb_servo"
claw_usb_path = "/dev/claw_serial"

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    
 

if __name__ == '__main__':
    device_list = []


    # 获取所有连接的串口设备
    ports = list(serial.tools.list_ports.comports())
    for port in ports:
        # print(port)
        # print("设备: {}".format(port.device))
        # print("描述: {}".format(port.description))
        # print("hwid: {}".format(port.hwid))
        # print("------------------------------")
        

        # if(port.description == "USB-RS485-WE - USB-RS485-WE"):
        #     hwid_string = port.hwid
        #     # 编写正则表达式
        #     pattern = re.compile(r"SER=([0-9A-Z]+)")
        #     # 使用findall方法查找所有匹配项，这将返回一个包含所有匹配结果的列表
        #     matches = pattern.findall(hwid_string)
        #     # 输出SER值
        #     for match in matches:
        #         print("串口：", port.device,"SER", match)  # 输出: DB81ASSB

        #         device_list.append(port.device)

        
        if(port.description == "LJ485A - LJ485A"):
            hwid_string = port.hwid
            # 编写正则表达式
            pattern = re.compile(r"SER=([0-9A-Z]+)")
            # 使用findall方法查找所有匹配项，这将返回一个包含所有匹配结果的列表
            matches = pattern.findall(hwid_string)
            # 输出SER值
            for match in matches:
                print("串口：", port.device,"SER", match)  # 输出: DB81ASSB

                device_list.append(port.device)

                
        if(port.description == "LJ485B - LJ485B"):
            hwid_string = port.hwid
            # 编写正则表达式
            pattern = re.compile(r"SER=([0-9A-Z]+)")
            # 使用findall方法查找所有匹配项，这将返回一个包含所有匹配结果的列表
            matches = pattern.findall(hwid_string)
            # 输出SER值
            for match in matches:
                print("串口：", port.device,"SER", match)  # 输出: DB81ASSB

                device_list.append(port.device)



print(device_list)

if len(device_list) == 2:
    # 定义脚本路径和参数
    arg1 = device_list[1]
    arg2 = "stark_serial_R"
    # 定义要运行的命令
    command = "sudo bash "+ folder_path +"/generate_serial.sh "  +  arg1  + " " + arg2
    print(command)
    # 使用 subprocess.run() 运行命令
    subprocess.run(command, shell=True)

    
    # 定义脚本路径和参数
    arg1 = device_list[0]
    arg2 = "stark_serial_L"
    # 定义要运行的命令
    command = "sudo bash "+ folder_path +"/generate_serial.sh "  +  arg1  + " " + arg2
    print(command)
    # 使用 subprocess.run() 运行命令
    subprocess.run(command, shell=True)


# sudo ./tools/check_tool/hand_test_R

# sudo ./tools/check_tool/hand_test_L

# sudo udevadm control --reload-rules
# sudo udevadm trigger

# ls /dev/stark_serial_L