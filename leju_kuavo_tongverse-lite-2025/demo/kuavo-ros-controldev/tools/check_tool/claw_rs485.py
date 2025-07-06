#!/usr/bin/env python
# coding=utf-8
'''
Author: dongdongmingming
Date: 2024-03-15 11:38:27
LastEditors: Please set LastEditors
LastEditTime: 2024-03-23 15:25:16
FilePath: /kuavo/tools/check_tool/claw_rs485.py
Description: 二指夹爪测试接口
'''

# class CLAW_JUNDUO():
#     def __init__(self, port, baur):
#         self.port = port
#         self.baur = baur

#         self.id_list = [1,2]
#         self.claw_status = [0,0]

import sys
sys.path.append('/home/lab/.local/lib/python3.8/site-packages/')
from jodellSdk.jodellSdkDemo import ClawEpgTool
import time


claw_tool = ClawEpgTool


def claw_status_task(CLAW_ID):
    for index in range(0, 10):
        status_list = claw_tool.getStatus(claw_tool, CLAW_ID, 2000, 3, 4) 
        claw_status = status_list[0] & 0xFF
        claw_check = (claw_status & 0xC0) >> 6

        if claw_check == 0x0:
            claw_check_str = '夹爪移动中未检测到物体'
        elif claw_check == 0x1:
            claw_check_str = '夹爪张开时检测到物体'
        elif claw_check == 0x2:
            claw_check_str = '夹爪闭合时检测到物体'
        elif claw_check == 0x3:
            claw_check_str = '夹爪已到指定位置未检测到物体'

        print(claw_check_str)
        print('夹爪状态:' + str(hex(claw_status)))
        mode_status = (status_list[0] & 0xFF00) >> 8
        print('Mode状态:' + str(hex(mode_status)))

        error_status = status_list[1] & 0xFF
        print('错误状态:' + str(hex(error_status)))
        current_pos = (status_list[1] & 0xFF00) >> 8
        print('当前位置:' + str(current_pos))

        current_speed = status_list[2] & 0xFF
        print('当前速度:' + str(current_speed))
        current_torque = (status_list[2] & 0xFF00) >> 8
        print('当前力矩:' + str(current_torque))

        current_vol = status_list[3] & 0xFF
        print('当前电压:' + str(current_vol))
        current_tempera =  (status_list[3] & 0xFF00) >> 8
        print('当前温度:' + str(current_tempera))

        print('')
        time.sleep(0.1)


def claw_enable(CLAW_ID):
    enable_flag = claw_tool.clawEnable(claw_tool, CLAW_ID, True)
    if enable_flag == 1:
        print(f'ID{CLAW_ID} 夹爪使能成功')
    else:
        print(f'ID{CLAW_ID} 夹爪使能失败，错误代码:{enable_flag}')


def claw_move(CLAW_ID):
        param_listA = [255,10,10]
        param_listB = [0,10,10]
        run_flag = claw_tool.runWithParam(claw_tool, CLAW_ID, int(param_listA[0]),int(param_listA[1]), int(param_listA[2]))
        if run_flag == 1:
            print(f'ID{CLAW_ID} 操作正常')
        else:
            print(f'ID{CLAW_ID} 操作失败，错误代码:{run_flag}')

        time.sleep(2)
        run_flag = claw_tool.runWithParam(claw_tool, CLAW_ID, int(param_listB[0]),int(param_listB[1]), int(param_listB[2]))
        if run_flag == 1:
            print(f'ID{CLAW_ID} 操作正常')
        else:
            print(f'ID{CLAW_ID} 操作失败，错误代码:{run_flag}')


def claw_test(dev, baur):

    com_flag = claw_tool.serialOperation(claw_tool, dev, baur, True)
    if com_flag == 1:
        print('打开串口成功')
    else:
        print(f'打开串口失败，错误代码:{com_flag}')

    claw_enable(1)
    claw_enable(2)
    time.sleep(3)

    claw_move(1)
    time.sleep(3)
    claw_move(2)
    time.sleep(3)

    # claw_status_task(1)
    # claw_status_task(2)

if __name__ == "__main__":

    claw_test("/dev/claw_serial", 115200)



