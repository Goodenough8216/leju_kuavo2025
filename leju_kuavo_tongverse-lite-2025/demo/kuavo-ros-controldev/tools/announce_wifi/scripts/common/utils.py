#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import copy
import pygame
import subprocess

digit_to_chinese = {
    '0': '零',
    '1': '一',
    '2': '二',
    '3': '三',
    '4': '四',
    '5': '五',
    '6': '六',
    '7': '七',
    '8': '八',
    '9': '九',
    '.': '点',
}

digit_to_english = {
    '0': 'zero',
    '1': 'one',
    '2': 'two',
    '3': 'three',
    '4': 'four',
    '5': 'five',
    '6': 'six',
    '7': 'seven',
    '8': 'eight',
    '9': 'nine',
    '.': 'dot',
    '-': 'dash',
}

def play_audio(file_path)->bool:
    """
        play audio file
    """
    
    try:
        pygame.mixer.init()
        pygame.mixer.music.load(file_path)
        print(f"playing audio file: {file_path}")

        pygame.mixer.music.play()
        
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        pygame.mixer.quit()
    except pygame.error as e:
        print(f"Error loading MP3 file: {e}")
        return False
    
    return True

def set_volume(volume_level):
    try:
        command = f"amixer set Master {volume_level}% unmute"
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error setting volume: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        
def ip_address_zh_cn(ip_address: str) -> str:
    """
        将 ip 地址转换为中文
    """
    global digit_to_chinese
    result = ' '.join(digit_to_chinese[char] for char in ip_address)
    return result

chinese_symbol_table = {
    '.':'点',
    '-':'杠',
    '@': ' \'at\' '
}
def text_zh_cn(text: str) -> str:
    global chinese_symbol_table
    text_zh_cn = copy.copy(text)
    for en_symbol, zh_symbol in chinese_symbol_table.items():
        text_zh_cn = text_zh_cn.replace(en_symbol, zh_symbol)
    
    return text_zh_cn

if __name__ == "__main__":
    ip = '192.168.1.1'
    print(ip_address_zh_cn(ip))

    print(text_zh_cn('hello@测试'))
    print(text_zh_cn('hello-测试'))
    print(text_zh_cn('hello.测试'))
