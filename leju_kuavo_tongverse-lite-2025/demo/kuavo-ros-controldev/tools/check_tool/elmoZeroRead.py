#!/usr/bin/env python
# coding=utf-8
'''
Author: dongdongmingming
Date: 2024-03-15 11:38:27
LastEditors: Please set LastEditors
LastEditTime: 2024-03-23 15:25:16
FilePath: /kuavo/tools/check_tool/elmoZeroRead.py
Description: elmo 电机零点转换零点数据
'''

import re

# 粘贴要设置的电机位置内容
text = """

"""

# 运行sudo ./src/biped_v2/kuavoController --real --cali  
# 当显示所有电机位置后马上Ctrl+C终止程序
# 将如下所示的电机位置值复制粘贴到上文 text = """ """ 引号中去
# 然后再次运行

""" 
0000003030: Slave 1 actual position 14.8820495,Encoder 97531.0000000
0000003040: Rated current 35.3600000
0000003050: Slave 2 actual position 4.3945312,Encoder 16000.0000000
0000003060: Rated current 11.7900000
0000003070: Slave 3 actual position 24.7629089,Encoder 90159.0000000
0000003080: Rated current 28.9900000
0000003090: Slave 4 actual position 4.2613220,Encoder 15515.0000000
0000003100: Rated current 42.4300000
0000003110: Slave 5 actual position 8.6946868,Encoder 113963.0000000
0000003120: Rated current 8.4900000
0000003130: Slave 6 actual position 7.3580932,Encoder 96444.0000000
0000003140: Rated current 8.4900000
0000003150: Slave 7 actual position 13.2957458,Encoder 87135.0000000
0000003160: Rated current 35.3600000
0000003171: Slave 8 actual position 30.0594177,Encoder 109443.0000000
0000003181: Rated current 11.7900000
0000003190: Slave 9 actual position 20.7718505,Encoder 75628.0000000
0000003201: Rated current 28.9900000
0000003210: Slave 10 actual position 29.4551696,Encoder 107243.0000000
0000003220: Rated current 42.4300000
0000003230: Slave 11 actual position 7.5830078,Encoder 99392.0000000
0000003240: Rated current 8.4900000
0000003250: Slave 12 actual position 8.0300903,Encoder 105252.0000000
0000003260: Rated current 8.4900000
0000003270: Slave 13 actual position 18.3853454,Encoder 66939.0000000
0000003280: Rated current 14.9900000
0000003290: Slave 14 actual position 34.5780944,Encoder 125895.0000000
0000003301: Rated current 14.9900000

"""

matches = re.findall(r'position\s+(-?\d+\.\d+),Encoder', text) 
for match in matches:
    print(match)
for match in matches:
    if(float(match) > 70):
        print("有电机超过一圈范围，电机需要重新上电！！！")