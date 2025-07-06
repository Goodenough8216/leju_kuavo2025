#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
from common.hotspot_util import (
    check_connection_exists, 
    create_or_update_connection, 
    delete_connection, 
    up_connection, 
    down_connection,
    get_connection_info
)

def create_kauvo_hotspot(ifname:str, robot_name: str, password: str) -> bool:
    con_name = 'kuavo-hotspot'
    ssid = f"{robot_name}的热点"
    
    print(f"Creating kuavo hotspot: ifname:{ifname}, ssid:{ssid}")
    try:
        return create_or_update_connection(con_name=con_name, ifname=ifname, ssid=ssid, password=password)
    except Exception as e:
        print(f"create_kauvo_hotspot except: {e}")
        return False
    
def check_kuavo_hotspot_exists() -> bool:
    return check_connection_exists('kuavo-hotspot')

def delete_kuavo_hotspot() -> bool:
    return delete_connection('kuavo-hotspot')

def up_kuavo_hotspot() -> bool:
    return up_connection('kuavo-hotspot')

def down_kuavo_hotspot() -> bool:
    return down_connection('kuavo-hotspot')

def get_kuavo_hotspot_info():
    return get_connection_info('kuavo-hotspot')

if __name__ == '__main__':
    create_kauvo_hotspot('wlo1', 'test', '12345678')
    print('check_kuavo_hotspot_exists:', check_kuavo_hotspot_exists())
    print("up kuavo hotspot")
    print(up_kuavo_hotspot())
    
    time.sleep(15)
    print("down kuavo hotspot")
    # print(down_kuavo_hotspot())

    time.sleep(3)
    print("delete kuavo hotspot")

    # print(delete_kuavo_hotspot())
