#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess

class WiFiDetails:
    def __init__(self, ssid, device, uuid, mode, ip, password):
        self._ssid = ssid
        self._device = device
        self._uuid = uuid
        self._mode = mode
        self._ip = ip
        self._password = password
    
    def __str__(self):
        return f"SSID: {self.ssid}, Device: {self.device}, UUID:{self._uuid}, MODE: {self._mode}, IP: {self.ip}, Password: {self.password}"
    
    @property
    def ssid(self):
        return self._ssid
    
    @property
    def device(self):
        return self._device
    
    @property
    def uuid(self):
        return self._uuid
    
    @property
    def mode(self):
        return self._mode
    
    @property
    def ip(self):
        return self._ip

    @property
    def password(self):
        return self._password
class WiFiUtil:
    def __init__(self):
        pass

    def _get_connected_wifi2(self):
        # nmcli -t -f active,ssid,device,signal dev wifi
        nmcli_output = subprocess.check_output(['nmcli', '-t', '-f', 'active,ssid,device,signal', 'dev', 'wifi']).decode('utf-8')
        
        lines = nmcli_output.strip().split('\n')
        wifi_infos = []
        for line in lines:
            parts = line.split(':')
            if len(parts) == 4 and (parts[0] == 'yes' or parts[0] == '是'):  # 考虑 yes 是中文的 `是`
                _, ssid, device, _ = parts
                ip = self._get_ip_address(device)
                wifi_info = WiFiDetails(ssid=ssid, device=device, uuid='', mode='infrastructure', ip=ip, password=None)
                wifi_infos.append(wifi_info)
        return wifi_infos
        
    def _get_connected_wifi(self):
        wifi_infos = []
        try:
            cmd = "nmcli -t -f TYPE,DEVICE,UUID connection show --active | grep '802-11-wireless'"
            result = subprocess.check_output(cmd, shell=True).decode('utf-8').strip()

            for line in result.split('\n'):
                if line:
                    parts = line.split(':')
                    if len(parts) == 3:
                        device, uuid = parts[1], parts[2]
                        # 根据 UUID 获取更多详细信息
                        # nmcli --show-secrets -t -f 802-11-wireless.ssid,IP4.ADDRESS,802-11-wireless.mode,802-11-wireless-security.psk con show $uuid
                        detail_cmd = f"nmcli --show-secrets -t -f 802-11-wireless.ssid,IP4.ADDRESS,802-11-wireless.mode,802-11-wireless-security.psk con show {uuid}"
                        detail_result = subprocess.check_output(detail_cmd, shell=True).decode('utf-8').strip()
                        
                        ssid, ip, mode,password = None, None, None, None
                        for detail_line in detail_result.split('\n'):
                            if '802-11-wireless.ssid' in detail_line:
                                ssid = detail_line.split(':')[1]
                            elif 'IP4.ADDRESS' in detail_line:
                                ip = detail_line.split(':')[1].split('/')[0]
                            elif '802-11-wireless.mode' in detail_line:
                                mode = detail_line.split(':')[1]
                            elif detail_line.startswith('802-11-wireless-security.psk:'):
                                password = detail_line.split(':')[1].strip()
                                if password == '<hidden>' or password == '':
                                    password = None
                        if ssid and ip and mode:
                            wifi_info = WiFiDetails(ssid=ssid, device=device, uuid=uuid, mode=mode, ip=ip, password=password)
                            wifi_infos.append(wifi_info)
        except Exception as e:
              print(f"{e}")  
        return wifi_infos

    def get_connected_wifi(self):
        return self._get_connected_wifi()
    def _get_ip_address(self, device):
        ip_output = subprocess.check_output(['ip', 'addr', 'show', device]).decode('utf-8')
        
        for line in ip_output.split('\n'):
            if 'inet ' in line:
                parts = line.strip().split()
                ip_address = parts[1].split('/')[0] 
                return ip_address
        
        return None 

def check_ap_support():
    """
        check if the device support AP mode.
    """
    try:
        cmd = f"iw list | grep -A 20 'Supported interface modes' | grep '* AP'"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        if result.stdout.strip():
            return True
        else:
            return  False
    except Exception as e:
        return  False

def get_wireless_interfaces()->list:
    """
        get all wireless interfaces.
        ---
    return:    
        phy0 wlan0
        phy1 wlan1
    """
    result = subprocess.run(['iw', 'dev'], capture_output=True, text=True)
    output = result.stdout

    phy = None
    interfaces = []

    for line in output.splitlines():
        if line.startswith('phy#'):
            phy = line.strip().replace('#', '')
        elif 'Interface' in line:
            interface = line.split()[1]
            if not interface.startswith('ap') and phy:
                interfaces.append((phy, interface))
    return interfaces

def get_supported_ap_interfaces()->list:
    """
        get the ap interface name from phy name.
    """
    ws = []
    ifs = get_wireless_interfaces()
    for phy, iface in ifs:
        cmd = f"iw phy|grep -A 50 \"{phy}\"| grep -A 10 \"Supported interface modes\" | grep -B 10 \"* AP\"" 
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        if result.stdout.strip():
            ws.append(iface)

    return ws     

if __name__ == "__main__":
    wifi_util = WiFiUtil()
    wifi_infos = wifi_util.get_connected_wifi()
    for wifi in wifi_infos:
        print("get_connected_wifi:", wifi)

    wifi_infos2 = wifi_util._get_connected_wifi2()
    for wifi in wifi_infos2:
        print("_get_connected_wifi2:", wifi)

    print("connectd wifi:", )
    print("get_supported_ap_interfaces:", get_supported_ap_interfaces())
    print("check_ap_support:", check_ap_support())    