#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess

import subprocess

import subprocess

def is_wifi_enabled():
    try:
        # 检查Wi-Fi是否已启用
        result = subprocess.run(['nmcli', 'radio', 'wifi'], stdout=subprocess.PIPE, text=True)
        wifi_status = result.stdout.strip()
        return wifi_status == 'enabled'
    except Exception as e:
        print(f"An error occurred while checking Wi-Fi status: {e}")
        return False
    
def enable_wifi():
    try:
        # 检查Wi-Fi是否已启用
        result = subprocess.run(['nmcli', 'radio', 'wifi'], stdout=subprocess.PIPE, text=True)
        wifi_status = result.stdout.strip()

        if wifi_status != 'enabled':
            # 启用Wi-Fi
            subprocess.run(['nmcli', 'radio', 'wifi', 'on'])
            print("Wi-Fi has been enabled.")
        else:
            print("Wi-Fi is already enabled.")
    except Exception as e:
        print(f"An error occurred while enabling Wi-Fi: {e}")

def check_connection_exists(con_name:str)->bool:
    try:
        result = subprocess.run(['nmcli', '-t', '-f', 'NAME', 'connection', 'show'], stdout=subprocess.PIPE, text=True)
        connections = result.stdout.strip().split('\n')
        
        for connection in connections:
            if connection == con_name:
                return True
        
        return False
    except Exception as e:
        print(f"An error occurred: {e}")
        return False
    
def create_or_update_connection(con_name: str, ifname:str, ssid: str, password: str) -> bool:
    try:
        if not check_connection_exists(con_name):
            print(f"Creating AP connection: {con_name}, SSID: {ssid}, Password: {password}")
            subprocess.run(['nmcli', 'con', 'add', 'type', 'wifi', 'ifname', ifname, 'con-name', con_name, 'ssid', ssid, 'autoconnect', 'no'], check=True)
        
        subprocess.run(['nmcli', 'con', 'modify', con_name, '802-11-wireless.mode', 'ap', '802-11-wireless.band', 'bg', 'ipv4.method', 'shared'], check=True)
        subprocess.run(['nmcli', 'con', 'modify', con_name, 'wifi-sec.key-mgmt', 'wpa-psk', 'wifi-sec.psk', password], check=True)
        subprocess.run(['nmcli', 'con', 'modify', con_name, '802-11-wireless.ssid', ssid], check=True)

        return True
    except subprocess.CalledProcessError as e:
        print(f"Error creating/updating connection: {e}")
        return False

def delete_connection(con_name: str) -> bool:
    try:
        if check_connection_exists(con_name):
            subprocess.run(['nmcli', 'con', 'delete', con_name], check=True)
            return True
        else:
            return True
    except subprocess.CalledProcessError as e:
        print(f"Error deleting connection: {e}")
        return False

def up_connection(con_name: str) -> bool:
    try:
        if check_connection_exists(con_name):
            if enable_wifi():
                print("up_connection Enabling Wi-Fi failed.")
                return False
            subprocess.run(['nmcli', 'con', 'up', con_name], check=True)
            return True
        else:
            return False
    except subprocess.CalledProcessError as e:
        print(f"Error connecting to connection: {e}")
        return False

def down_connection(con_name: str) -> bool:    
    try:
        if check_connection_exists(con_name):
            subprocess.run(['nmcli', 'con', 'down', con_name], check=True)
            return True
        else:
            return False
    except subprocess.CalledProcessError as e:
        print(f"Error disconnecting from connection: {e}")
        return False
    
def get_connection_info(con_name):
    try:
        # nmcli --show-secrets -t -f 802-11-wireless.ssid,ip4.address,802-11-wireless-security.psk con show $con_name
        result = subprocess.run(['sudo', 'nmcli', '--show-secrets', '-t', '-f', '802-11-wireless.ssid,ip4.address,802-11-wireless-security.psk', 'con', 'show', con_name], stdout=subprocess.PIPE, text=True)
        output = result.stdout.strip()

        lines = output.split('\n')
        ssid = None
        ip_address = None
        password = None

        for line in lines:
            if line.startswith('802-11-wireless.ssid:'):
                ssid = line.split(':')[1].strip()
            elif line.startswith('IP4.ADDRESS['):
                ip_address = line.split(':')[1].strip().split('/')[0]
            elif line.startswith('802-11-wireless-security.psk:'):
                password = line.split(':')[1].strip()
                if password == '<hidden>' or password == '':
                    print("Password is hidden")
                    password = None

        return ssid, ip_address, password
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None, None
    
    
if __name__ == '__main__':
    ssid, ip_address, password = get_connection_info('kuavo-hotspot')
    print("ssid:", ssid)
    print('ip_address:', ip_address)
    print('password:', password)