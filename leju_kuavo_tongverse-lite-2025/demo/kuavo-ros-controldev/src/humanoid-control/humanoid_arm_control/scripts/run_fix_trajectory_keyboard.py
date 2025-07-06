#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import rospy
import json
from pynput import keyboard

from BezierWrap import BezierWrap
from KuavoSDK import kuavo

class KeyboardControl():
    def __init__(self, map_file_path) -> None:
        try:
            with open(map_file_path, 'r') as file:
                self.map = json.load(file)
                print("- Load armservice config success")
        except FileNotFoundError:
            print("- KeyboardControl: 配置文件未找到")
            exit(0)
        except json.JSONDecodeError:
            print("- KeyboardControl:JSON格式错误")
            exit(0)
        
        self.trajectorys = {}
        self.kuavo = kuavo()
        self.kuavo.set_arm_control_mode(2)
        
        # 加载所有轨迹
        self.bezierwrap = BezierWrap()
        print("----------------开始加载轨迹----------------")
        for key in self.map:
            print("----------------加载按键{}的轨迹----------------".format(key))
            action_filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.map[key])
            self.bezierwrap.load_pre_bezier_traj(action_filepath)
            curves = self.bezierwrap.get_standard_bezier_line_format()
            curves = curves[0:14]
            
            # 获取插值后的点
            # points_all 的 shape 为(14, n, 2), 14个关节，18个点，2
            points_all = self.bezierwrap.get_traj_points_after_interpolate(curves, 100)
            
            # 把time和q分开
            result = self.bezierwrap.get_traj_points_after_seperate_time_q(points_all)
            self.trajectorys[key] = result
        # print(self.trajectorys["1"])
        
    def test(self, char):
        print("执行按键  {}  的轨迹".format(char))
        self.kuavo.move_with_trajactory(self.trajectorys[char][0], self.trajectorys[char][1])
        print("执行结束")

    # 定义按键按下处理函数
    def on_press(self, key):
        zeros = [0,0,0,0,0,0]
        thumb = [0,100,0,0,0,0]
        fist = [100,100,100,100,100,100]
        try:
            char = key.char  # 获取按下的字符
            if char in self.trajectorys.keys():
                print("执行按键  {}  的轨迹".format(char))
                self.kuavo.move_with_trajactory(self.trajectorys[char][0], self.trajectorys[char][1])
                if char == "1":
                    rospy.sleep(4)
                    self.kuavo.publish_hand_position(zeros,thumb)
                    rospy.sleep(3.3)
                    self.kuavo.publish_hand_position(zeros, fist)
                    rospy.sleep(7)
                    self.kuavo.publish_hand_position(zeros, thumb)
                    rospy.sleep(3)
                    self.kuavo.publish_hand_position(zeros, zeros)
                else:
                    rospy.sleep(4)
                    self.kuavo.publish_hand_position(thumb,zeros)
                    rospy.sleep(3.3)
                    self.kuavo.publish_hand_position(fist, zeros)
                    rospy.sleep(7)
                    self.kuavo.publish_hand_position(thumb, zeros)
                    rospy.sleep(3)
                    self.kuavo.publish_hand_position(zeros, zeros)
                
            else:
                if char == "c":
                    self.kuavo.publish_hand_position(zeros,thumb)
                    rospy.sleep(0.5)
                    self.kuavo.publish_hand_position(zeros, fist)
                elif char == "o":
                    self.kuavo.publish_hand_position(zeros, zeros)
                    
        except AttributeError:
            print("wrong key")  # 忽略特殊按键
        except Exception as e:
            print(e)

    # 定义按键释放处理函数（可选）
    def on_release(self, key):
        # 如果按下的是 'esc' 键，则停止监听
        if key == keyboard.Key.esc:
            return False

if __name__ == "__main__":
    
    rospy.init_node("keyboard_fix_traj")
    
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config/keyboard_map.json")
    keyboard_control = KeyboardControl(config_path)
    # 创建监听器对象
    listener = keyboard.Listener(on_press=keyboard_control.on_press, on_release=keyboard_control.on_release)
    # 开始监听
    listener.start()
    
    rospy.logwarn("------------------------------------------")
    rospy.logwarn("按 '1' 键执行固定轨迹 1")
    rospy.logwarn("按 '2' 键执行固定轨迹 2")
    rospy.logwarn("按 '3' 键执行固定轨迹 3")
    rospy.logwarn("按 'esc' 键退出程序")
    rospy.logwarn("------------------------------------------")

    # 保持程序运行，直到按下 'esc' 键
    listener.join()