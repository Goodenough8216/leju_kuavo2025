#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import signal
import argparse
from common.config import Config 
from common.utils import play_audio, ip_address_zh_cn, set_volume, text_zh_cn
from common.wifi_util import WiFiUtil, WiFiDetails, check_ap_support, get_supported_ap_interfaces
from tts_generate import TTSGenerator
from common.logger import KuavoLogger
from common.hotspot_util import enable_wifi
from kuavo_hotspot import (check_kuavo_hotspot_exists, 
                           create_kauvo_hotspot, 
                           get_kuavo_hotspot_info,
                           down_kuavo_hotspot,
                           up_kuavo_hotspot
                           )
class WifiAnnouceService:
    def  __init__(self):
        self._config_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../config/config.json')
        self._config = Config(self._config_file_path)
        self._wifi_util = WiFiUtil()
        self._tts = TTSGenerator()

    def init(self):
        self._tts.load_model() # load tts model

    def play_text_audio(self, text:str, filename:str)->bool:
        """
            call tts-model to generate audio file, and play the audio file.
        """
        # audio assets dir
        assets_dir = '/tmp/lejurobot/kuavo-wifi-announce'
        os.system(f'mkdir -p {assets_dir}')

        # remove old audio file
        audio_path = f'{assets_dir}/{filename}.wav'
        if os.path.exists(audio_path):
            os.remove(audio_path)

        # call tts-model to generate audio file
        tts_text_zh_cn = text_zh_cn(text)
        if not self._tts.generate_text_to_speech(tts_text_zh_cn, audio_path):
            KuavoLogger.error("play_text_audio generate tts audio failed!")
            return False
        
        # amixer set Master 100% unmute
        set_volume(volume_level=100)
        return play_audio(audio_path)
    
    def play_wifi_connected_audio(self, wifi_info:WiFiDetails):
        text = self._config.connected_text
        ip_zh_cn = ip_address_zh_cn(wifi_info.ip)
        
        tts_text = text.format(wifi_name=wifi_info.ssid, 
                               ip_address=ip_zh_cn)
        
        KuavoLogger.warning(f"play wifi connected text: {tts_text}")
        max_retries  = max(2, self._config.play_retries)
        for i in range(max_retries):
            if self.play_text_audio(tts_text, 'wifi_connected'):
                return 
            time.sleep(1.0)
    
    def play_hotspot_active_onboot_audio(self, wifi_info:WiFiDetails):
        text = self._config.hotspot_active_onboot_text
        ip_zh_cn = ip_address_zh_cn(wifi_info.ip)
        
        tts_text = text.format(hotspot_name=wifi_info.ssid, 
                               ip_address=ip_zh_cn)
        
        KuavoLogger.warning(f"play hotspot active on boot text: {tts_text}")
        max_retries  = max(2, self._config.play_retries)
        for i in range(max_retries):
            if self.play_text_audio(tts_text, 'hotspot_active_onboot'):
                return 
            time.sleep(1.0)

    def play_ap_model_audio(self, ssid, ip_address, password):
        text = self._config.disconnected_text
        ip_zh_cn = ip_address_zh_cn(ip_address)
        tts_text = text.format(hotspot_name=ssid, 
                               ip_address=ip_zh_cn)
        if password is not None:
            # "ssawd" --> "'s' 's' 'a' 'w' 'd'"
            pswd_text = "' '".join(f"'{char}'" for char in password)
            tts_text += f",  密码是: \"{text_zh_cn(pswd_text)}\""
        KuavoLogger.warning(f"play change to ap  text: {tts_text}")

        max_retries  = max(2, self._config.play_retries)
        for i in range(max_retries):
            if  self.play_text_audio(tts_text, 'chngeto_ap'):
                return
            time.sleep(1.0)

    def play_unspported_ap_audio(self):
        tts_text = self._config.unspported_ap_text
        KuavoLogger.warning(f"play unspported ap text: {tts_text}")

        max_retries  = max(2, self._config.play_retries)
        for i in range(max_retries):
            if self.play_text_audio(tts_text, 'unspported_ap'):
                return
            time.sleep(1.0)

    def waitting_wifi_connected(self, timeout:int)->bool:
        """
            wait wifi connected, and play wifi connected tips.
        """
        total = 0
        while total < timeout:
            connected_wifi_infos = self._wifi_util.get_connected_wifi()
            # connected_wifi_infos.append(WiFiDetails('lejurobot', 'lejurobot', 'lejurobot', '192.110.158.64')) # Debug!
            if len(connected_wifi_infos) != 0: # connected wifi tips.
                wifi_info = connected_wifi_infos[0]
                KuavoLogger.info(f"WifiAnnouceService connected wifi/hotspot: {wifi_info.ssid}, ip: {wifi_info.ip}")
                
                if wifi_info.mode == 'ap':
                    self.play_hotspot_active_onboot_audio(wifi_info)
                else:    
                    self.play_wifi_connected_audio(wifi_info)
                return True
            
            #################################################
            # TODO: maybe, try connect to other wifi!
            #################################################

            time.sleep(self._config.interval_s)
            total += self._config.interval_s

        KuavoLogger.warning("WifiAnnouceService wifi connected dectect timeout!")
        return False
    def run(self):
        KuavoLogger.info("WifiAnnouceService start running...")
        detection_duration_seconds = int(self._config.detection_wifi_connected_mins * 60)

        # enable wifi.
        enable_wifi()

        # 1.wait wifi connected.    
        if self.waitting_wifi_connected(timeout=detection_duration_seconds):
            KuavoLogger.info("WifiAnnouceService wifi connected, happy to go!")
            return
        
        KuavoLogger.warning("WifiAnnouceService wifi disconnected!")

        # only announce wifi, don't change to ap(hotspot) mode.
        if self._config.only_annouce_wifi:
            KuavoLogger.info("WifiAnnouceService config only_annouce_wifi is true, exit.")
            return
        
        KuavoLogger.warning("WifiAnnouceService change to ap(hotspot) mode!")

        # 2.change to ap mode.
        if not check_ap_support(): 
            self.play_unspported_ap_audio()  # not devices support to ap model, play unspported ap audio.
            KuavoLogger.error("WifiAnnouceService no devices support to ap(hotspot) mode.")
            return
        
        # 3.choose a device to change AP model. => We use the first one. wlan[0]
        if not check_kuavo_hotspot_exists():
            KuavoLogger.info("WifiAnnouceService kuavo hotspot not exists, create it.")
            wlan = get_supported_ap_interfaces()
            if len(wlan) == 0:
                KuavoLogger.error("WifiAnnouceService create kuavo hotspot fail, because no supported wifi device.")
                self.play_unspported_ap_audio()
                return 

            if not create_kauvo_hotspot(ifname=wlan[0], robot_name=self._config.robot_name, password='kuavo123456'):
                KuavoLogger.error(f"WifiAnnouceService create kuavo hotspot failed, ifname:{wlan[0]}, robot_name:{self._config.robot_name}")
                self.play_unspported_ap_audio()
                return 
        
        # 4. up kuavo-hotspot
        KuavoLogger.info("WifiAnnouceService kuavo hotspot exists, up it.")
        try_times = 3
        up_success = False
        for i in range(try_times):
            if up_kuavo_hotspot():
                up_success = True
                KuavoLogger.info("WifiAnnouceService up kuavo hotspot success!") 
                break

        if not up_success:
            self.play_text_audio(self._config.ap_change_fail_text, 'change_ap_mode_fail')
            KuavoLogger.info("WifiAnnouceService up kuavo hotspot failed!") 
            return
        
        # get kuavo hotspot info
        ssid, ip_address, password = get_kuavo_hotspot_info()
        KuavoLogger.info(f"WifiAnnouceService get kuavo hotspot info, hotspot:{ssid}, ip:{ip_address}")
        if ssid is None or ip_address is None:
            down_kuavo_hotspot()
            KuavoLogger.error("WifiAnnouceService get kuavo hotspot info failed.")
            return
        
        self.play_ap_model_audio(ssid=ssid, ip_address=ip_address, password=password)
        return
               
def signal_handler(sig, frame):
    sys.exit(0)

if __name__ == "__main__":    
    # register signal handler.
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    parser = argparse.ArgumentParser(description="Kuavo Wifi Announce Service")
    parser.add_argument('--play-text', type=str, help='play custom text')
    args = parser.parse_args()

    play_text_mode = args.play_text is not None
    custom_text = args.play_text
    
    ascii_logo = r"""
          _                                         _  __ _                                                    
         | | ___   _  __ ___   _____      __      _(_)/ _(_)       __ _ _ __  _ __   ___  _   _ _ __   ___ ___ 
         | |/ / | | |/ _` \ \ / / _ \ ____\ \ /\ / / | |_| |_____ / _` | '_ \| '_ \ / _ \| | | | '_ \ / __/ _ \
         |   <| |_| | (_| |\ V / (_) |_____\ V  V /| |  _| |_____| (_| | | | | | | | (_) | |_| | | | | (_|  __/
         |_|\_\\__,_|\__,_| \_/ \___/       \_/\_/ |_|_| |_|      \__,_|_| |_|_| |_|\___/ \__,_|_| |_|\___\___|
                                                                                                               
    """
    KuavoLogger.info(ascii_logo)
    KuavoLogger.info("\n--------------- Kuavo Wifi Announce Service Init ---------------\n")

    wifi_announce = WifiAnnouceService()
    wifi_announce.init()
   
    if play_text_mode:
        KuavoLogger.info(" ---------------- play text mode -------------------")
        wifi_announce.play_text_audio(custom_text, 'custom')
    else:
        # service run.
        wifi_announce.play_text_audio("已开机", 'hello')
        wifi_announce.run()

