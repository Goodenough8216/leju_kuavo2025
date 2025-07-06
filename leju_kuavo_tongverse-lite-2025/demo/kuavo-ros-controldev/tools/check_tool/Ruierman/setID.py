#!/usr/bin/env python
# coding=utf-8
'''
Author: dongdongmingming
Date: 2024-03-23 10:39:06
LastEditors: Please set LastEditors
LastEditTime: 2024-03-23 14:57:11
FilePath: /kuavo/tools/check_tool/Ruierman/setID.py
Description: 
'''
from SimpleSDK import WHJ30Tools
import time

if __name__ == "__main__":
    # 创建WHJ30Tools对象实例
    whj30 = WHJ30Tools()

    # 开启CAN总线通讯
    whj30.open_canbus()
    newid = 0x01
    oldid = 0x03
    # 跳过IAP更新操作，传入关节ID id
    whj30.skip_iap_update(oldid)
    time.sleep(2)


    # 失能关节，传入关节ID id
    whj30.disable_joint(oldid)
    time.sleep(2)

    # 使能关节，传入关节ID id
    whj30.enable_joint(oldid)
    time.sleep(4)

    whj30.set_joint_id(oldid,newid)
    time.sleep(1)
    whj30.save_joint_param(oldid)
    # whj30.set_joint_zero_position(id)
    # whj30.save_joint_param(id)
    
    time.sleep(1)


    whj30.close_canbus()
