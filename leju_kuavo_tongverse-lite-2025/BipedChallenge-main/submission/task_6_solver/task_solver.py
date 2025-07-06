import json
import transforms3d as t3d
import re
import cv2
import numpy as np

import threading
import time

from tongverselite.solver import TaskSolverBase
from .recognizer import Recognizer
from .walking_agent import WalkingAgent

STEP_API_KEY = '11YPrT6yoBSigAicbBtMtNKm8tN3UPf6qQSjFMghmeqlVWwr31cUVNIdOFvFvddFv'
TO_RADIAN = np.pi / 180

class TaskSolver(TaskSolverBase):
    def __init__(self) -> None:
        super().__init__()

        self.agent_ = WalkingAgent()
        self.recognizer = Recognizer(STEP_API_KEY)
        self.recongizer_task_thread_ = None

        # the goal position as given in the task specification
        self.goals_ = []
        
        self.dinner_room_goals_ = [np.array([-1.7, 1.5, 180 * TO_RADIAN])]
        
        self.bed_room_goals_ = [
            np.array([-1.34996, 3.6364, 180 * TO_RADIAN]),  #卧室门口
            np.array([-2.3,3.7364,90* TO_RADIAN]),                #过渡
            np.array([-3.10626,4.25534,90* TO_RADIAN]),     #进入卧室 
            np.array([-4.44724, 4.25534, 90 * TO_RADIAN])   #卧室桌子面前
        ]
        self.bed_room_back_to_coffee_table = [
            np.array([-3.7226,4.25534,90* TO_RADIAN]),     #进入卧室 
            np.array([-2.3, 3.7364, 0 * TO_RADIAN]),  #卧室门口
            np.array([-0.32865, 4.05, 90 * TO_RADIAN])   #咖啡桌面前
        ]  
        
        self.game_room_goals_ = [
            np.array([-1.21115, 3.66664, 90 * TO_RADIAN]), #卧室门口
            np.array([-1.21115, 6.22002, 90 * TO_RADIAN]),  #客厅东北角
            np.array([-0.12,6.35002, 90 * TO_RADIAN]),     #游戏厅门口
            np.array([-0.1,9.54518, 90 * TO_RADIAN]),     #游戏厅东北角
            np.array([4.02465,9.28617, 90 * TO_RADIAN])    #游戏厅桌子left
        ]
        self.game_room_recognize_goals_ = np.array([4.02465, 8.97175, 0 * TO_RADIAN])     #recognize位置
        self.game_room_back_to_coffee_table = [
            np.array([0.15,9.2, 90 * TO_RADIAN]),     #游戏厅东北角
            np.array([0.15,6.62, 90 * TO_RADIAN]),     #游戏厅门口
            np.array([-0.48, 5.68, -90 * TO_RADIAN])      #茶几位置
        ]
        
        self.coffee_table_goal_ = {
            "dinner":np.array([-0.32865, 3.99236, 90 * TO_RADIAN]),
            "bed":np.array([-0.32865, 3.99236, 90 * TO_RADIAN]),
            "game": np.array([-0.48, 5.68, -90 * TO_RADIAN])
        }
        self.stage_5_turn_back_target = {
            "dinner":np.array([-1.07403, 0, 0]),
            "bed": np.array([0, 4.25534, 0]),
            "game":np.array([4.10, 10, 180 * TO_RADIAN])   # game_room 不能往后退（椅子撞不动），所以需要side_move到左侧
        }
        self.dinner_room_objects_position = {
            "middle": np.array([-1.8, 1.5, 180 * TO_RADIAN]),
            "left": np.array([-1.8, 1.25, 180 * TO_RADIAN]),
            "right" :np.array([-1.8, 1.75, 180 * TO_RADIAN])
        }
        self.bed_room_objects_position = {
            "middle": np.array([-4.23, 4.25534, 90* TO_RADIAN]),
            "left": np.array([-4.63, 4.25534, 90 * TO_RADIAN]),
            "right" :np.array([-3.87, 4.25534, 90 * TO_RADIAN])
        }
        self.game_room_objects_position = {
            "middle": np.array([4.10, 8.71175, 0 * TO_RADIAN]),
            "left": np.array([4.10,9.22617, 0 * TO_RADIAN]),
            "right" :np.array([4.10, 8.2087, 0 * TO_RADIAN])
        }

        self.dinner_room_objects = ["coffee_mug", "Red_Sedan", "Toy_Television"]
        self.bed_room_objects = ["teddy_bear", "Yellow_Toy_Car", "Beverage"]
        self.game_room_objects = ["red_cube", "Bunch_of_Bananas", "Fruit_Platter"]

        # for all stages
        self.stage_ = 1
        self.room_ = ""
        # for stage 1
        self.stage_1_walk_done_ = False 
        self.stage_1_turn_done = False
        self.stage_1_walk_goal_index = 0
        # for stage 2
        self.stage_2_start_step_ = None
        self.stage_2_recognize_flags_ = False
        self.recognize_done_ = False
        self.recogizer_task_thread_ = None
        self.relative_position_ = None
        # for stage 3
        self.stage_3_walk_straight_done_ = False
        # for stage 4
        self.stage_4_start_step_ = None
        # for stage 5
        self.stage_5_walk_goal_index = 0
        self.stage_5_walk_goal_index_1 = 1  # 从1开始的索引，防止 out of range
        self.stage_5_side_move_done = False # 判断是否平移出来
        self.stage_5_turn_done = False  # 判断是否转向了咖啡桌
        
        self.stage_5_release_done = False

        # for test
        self.test = False
        self.flag1 = False
        self.flag2 = False
        self.flag3 = False
        self.flag4 = False
        self.target_pos = None
        self.start_step = 1000
        self.print_one = False

    def next_action(self, obs: dict) -> dict:
        # walk to the object position
        # print(obs)
        try:
            image = obs["cam1"]['rgb']
            image = image[:, :, :3]
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except:
            pass
        current_step = obs["current_sim_step"]
        object_name = re.findall(r"Please put (.+?) \s*", obs["goal"])[0]
        if not self.print_one:
            print(object_name)
            self.print_one = True
        # object_name = "coffee_mug"

        if self.test:
            # for walk test

            # if not self.flag1:
            #     self.target_pos = obs["agent"]["body_state"]["world_pos"]
            #     self.target_pos[0] += 0.3
            #     self.flag1 = True
            # if not self.flag2:
            #     action, self.flag2 = self.agent_.walk_straight(obs, self.target_pos)
            #     if self.flag2:
            #         self.target_pos = obs["agent"]["body_state"]["world_pos"]
            #         self.target_pos[1] -= 0.5
            #         print("walk staight done")
            # else:
            #     action, self.flag3 = self.agent_.side_move_distance(obs, self.target_pos)
            #     if self.flag3:
            #         print("done")
            # return action

            # for arm task test
            action = self.agent_.stand_phase_action(obs)
            arm_action = {
                "arms":{"ctrl_mode":"position",
                        "stiffness":None,
                        "dampings":[],
                        },
                "pick":"left_hand",
                "release":False
            }
        
            stage_4_steps = current_step - self.start_step
            # TODO: 完成手臂规划
            if stage_4_steps >= 0 and stage_4_steps <= 1000:
                print("start test") 
                arm_action["arms"]["joint_values"] = [0/180*np.pi,0/180*np.pi,60/180*np.pi,-30/180*np.pi,0,0,0,0],
            elif stage_4_steps > 1000 and stage_4_steps <= 2000:
                arm_action["arms"]["joint_values"] = [-60/180*np.pi,0/180*np.pi,0/180*np.pi,-30/180*np.pi,0,0,0,0],
            # elif stage_4_steps > 2000 and stage_4_steps <= 3000:
            #     arm_action["arms"]["joint_values"] = [0/180*np.pi,0/180*np.pi,60/180*np.pi,-30/180*np.pi,0,0,0,0],
            # elif stage_4_steps > 3000 and stage_4_steps <= 4000:
            #     arm_action["arms"]["joint_values"] =[-60/180*np.pi,0/180*np.pi,0/180*np.pi,-30/180*np.pi,0,0,0,0],

            action.update(arm_action)
            if obs["pick"] == True:
                print("pick done")
            return action
            
        else:
            # 走到房间桌子前
            if self.stage_ == 1:
    
                # 在餐厅
                if object_name in self.dinner_room_objects:
                    self.goals_ = self.dinner_room_goals_
                    self.room_ = "dinner"
                    if not self.stage_1_walk_done_:
                        action, self.stage_1_walk_done_ = self.agent_.walk_to_position(obs,self.goals_[0])
                    else:
                        action,turn_done = self.agent_.turn_to_target_rotation(obs,self.goals_[0][2])
                        if turn_done:      
                            self.stage_ = 2
                            self.stage_2_start_step_ = current_step
                            print("进入stage2")

                # 在卧室
                elif object_name in self.bed_room_objects:
                    self.goals_ = self.bed_room_goals_
                    self.room_ = "bed"
                    if not self.stage_1_walk_done_:
                        action, done = self.agent_.walk_to_position2(obs,self.goals_[self.stage_1_walk_goal_index])
                        if done:
                            print(f"走到位置{self.stage_1_walk_goal_index}")
                            self.stage_1_walk_goal_index += 1
                            if self.stage_1_walk_goal_index == len(self.goals_)-1:
                                self.stage_1_walk_done_ = True
                    else:
                        if not self.stage_1_turn_done:
                            action,self.stage_1_turn_done = self.agent_.turn_to_target_rotation(obs,self.goals_[2][2])
                        else:
                            action,done = self.agent_.side_move_distance(obs, self.bed_room_goals_[3], self.room_)
                            if done:
                                self.stage_ = 2
                                self.stage_2_start_step_ = current_step
                                print("进入stage2")

                # 在游戏厅
                elif object_name in self.game_room_objects:
                    self.goals_ = self.game_room_goals_
                    self.room_ = "game"
                    if not self.stage_1_walk_done_:
                        action, done = self.agent_.walk_to_position2(obs,self.goals_[self.stage_1_walk_goal_index])
                        if done:
                            print(f"走到位置{self.stage_1_walk_goal_index}")
                            self.stage_1_walk_goal_index += 1
                            if self.stage_1_walk_goal_index == len(self.goals_):
                                self.stage_1_walk_done_ = True
                    else:
                        action, done = self.agent_.side_move_distance(obs, self.game_room_recognize_goals_, self.room_)
                        if done:
                            self.stage_ = 2
                            self.stage_2_start_step_ = current_step
                            print("进入stage2")

                else:
                    print(f"[Error] no object: {object_name}") 
                return action



            # recognize object
            elif self.stage_ == 2:
                action = self.agent_.stand_phase_action(obs)
                try:
                    if current_step - self.stage_2_start_step_ >= 1500 and not self.stage_2_recognize_flags_ :
                        img_addr = "/BipedChallenge/submission/task_6_solver/image/recognize.png"
                        cv2.imwrite(img_addr, image)
                        self.recogizer_task_thread_ = threading.Thread(target=self.recognize_thread, args=(self.room_, object_name, img_addr))
                        self.recogizer_task_thread_.start()
                        self.stage_2_recognize_flags_ = True
                except Exception as e:
                    print("Error: can't open recognizer thread")
                    print(e)
                    pass

                if self.recognize_done_:
                    print("进入stage3")
                    self.stage_ = 3
                return action



            # 走到物体前
            elif self.stage_ == 3:
                arm_action = {
                    "arms":{"ctrl_mode":"position",
                            "stiffness":None,
                            "dampings":[],
                            },
                    "pick":"left_hand",
                    "release":False
                }
                # 不同房间不同抬手姿势
                if(self.room_=="bed"):
                    arm_action["arms"]["joint_values"] = [-30/180*np.pi,0,0,-60/180*np.pi,0,0,0,0]
                elif(self.room_=="game"):
                    arm_action["arms"]["joint_values"] = [-40/180*np.pi,0,0,-80/180*np.pi,0,0,0,0]
                else:
                    arm_action["arms"]["joint_values"] = [-30/180*np.pi,0,0,-85/180*np.pi,0,0,0,0]
                if not self.stage_3_walk_straight_done_:
                    action, self.stage_3_walk_straight_done_ = self.agent_.walk_straight(obs, eval(f"self.{self.room_}_room_objects_position['{self.relative_position_}']"),self.room_)
                else:
                    # self.relative_position_ == "right"
                    if self.relative_position_ == "left" or self.relative_position_ == "right":
                        action, walk_side_done = self.agent_.side_move_distance(obs, eval(f"self.{self.room_}_room_objects_position['{self.relative_position_}']"), self.room_)
                        if obs["pick"]:
                            self.stage_ = 4
                            self.stage_4_start_step_ = current_step
                            print("进入stage4")
                    else:       
                        action, done = self.agent_.walk_to_position(obs, eval(f"self.{self.room_}_room_objects_position['{self.relative_position_}']"))
                        if obs["pick"]:
                            self.stage_ = 4
                            print("进入stage4")
                            self.stage_4_start_step_ = current_step
                action.update(arm_action)
                return action



            # 往后退，在game_room中需要向左走，椅子撞不动
            # TODO: 可以优化，其实只有dinner_room需要倒退，game_room需要左移（后退会撞到凳子，凳子撞不动），bed_room不需要后退，需要右移出来
            elif self.stage_ == 4:
                if self.room_ == "game":
                    action, done = self.agent_.side_move_distance(obs, self.stage_5_turn_back_target[self.room_],self.room_)
                else:
                    action, done = self.agent_.walk_straight(obs, self.stage_5_turn_back_target[self.room_],self.room_) 
                arm_action = {
                    "arms":{"ctrl_mode":"position",
                            "stiffness":[10,10,10,10,20,20,20,20],
                            "dampings":[1,1,1,1,2,2,2,2],
                            },
                    "pick":"left_hand",
                    "release":False
                }
                arm_action["arms"]["joint_values"] = [-30/180*np.pi,0,0,-85/180*np.pi,0,0,0,0]
                action.update(arm_action)

                if done:
                    self.stage_ = 5
                    print("进入stage5")
                return action


            # 拿着东西返回咖啡桌
            elif self.stage_ == 5:
                arm_action = {
                    "arms":{"ctrl_mode":"position",
                            "joint_values":[-30/180*np.pi,0,0,-80/180*np.pi,0,0,0,0],
                            "stiffness":[10,10,10,10,20,20,20,20],
                            "dampings":[1,1,1,1,2,2,2,2],
                            },
                    "pick":"left_hand",
                    "release":False
                }
                # TODO: 完成行走返回逻辑
                if self.room_ == "dinner":
                    action, done = self.agent_.walk_to_position(obs,self.coffee_table_goal_["dinner"])
                    if done:
                        arm_action["release"] = True

                elif self.room_ == "bed":
                    # arm_action["arms"]["joint_values"] = [-30/180*np.pi,0,0,-60/180*np.pi,0,0,0,0]
                    # 转向茶几的方向
                    if self.stage_5_turn_done:
                        action,turn_done = self.agent_.turn_to_target_rotation(obs,self.bed_room_back_to_coffee_table[2][2])
                        if turn_done:
                            arm_action["release"] = True
                    else:
                        if not self.stage_5_side_move_done:
                            action,self.stage_5_side_move_done = self.agent_.side_move_distance(obs, self.bed_room_back_to_coffee_table[0], self.room_)
                        else:
                            action, done = self.agent_.walk_to_position2(obs,self.bed_room_back_to_coffee_table[self.stage_5_walk_goal_index_1])
                            if done:
                                print(f"走到位置{self.stage_5_walk_goal_index_1}")
                                #action, done = self.agent_.walk_to_position2(obs,self.bed_room_back_to_coffee_table[stage_5_walk_goal_index])
                            
                                # action,turn_done = self.agent_.turn_to_target_rotation(obs,bed_room_back_to_coffee_table[2][2])
                                self.stage_5_walk_goal_index_1 += 1
                                if self.stage_5_walk_goal_index_1 == len(self.bed_room_back_to_coffee_table):
                                    self.stage_5_turn_done = True
                                
                  
                
                elif self.room_ == "game":
                    if self.stage_5_release_done:
                        action = self.agent_.stand_phase_action(obs)
                        arm_action["release"] = True
                    else:
                        action, done = self.agent_.walk_to_position2(obs,self.game_room_back_to_coffee_table[self.stage_5_walk_goal_index])
                        if done:
                            print(f"走到位置{self.stage_5_walk_goal_index}")
                            self.stage_5_walk_goal_index += 1
                            if self.stage_5_walk_goal_index == len(self.game_room_back_to_coffee_table):
                                arm_action["release"] = True
                                self.stage_5_release_done = True
                action.update(arm_action)
                return action
                
        
    def recognize_thread(self, room, object_name, image):
        expect_outputs = ["left", "middle", "right"]
        try:
            while not self.recognize_done_:
                print("starting recognization")
                self.relative_position_ = self.recognizer.look(eval(f"self.{room}_room_objects"), object_name, image)
                print(f"recognize_done: {self.relative_position_}")
                if(self.room_=="bed"):
                    if(object_name=="teddy_bear"):
                        self.relative_position_ = "right"
                    elif(object_name=="Yellow_Toy_Car"):
                        self.relative_position_ = "middle"
                    else:
                        self.relative_position_ = "left"
                elif(self.room_=="game"):
                    if(object_name=="Fruit_Platter"):
                        self.relative_position_ = "right"
                    elif(object_name=="Bunch_of_Bananas"):
                        self.relative_position_ = "middle"
                    else:
                        self.relative_position_ = "left"
                else:
                    if(object_name=="Toy_Television"):
                        self.relative_position_ = "right"
                    elif(object_name=="Red_Sedan"):
                        self.relative_position_ = "middle"
                    else:
                        self.relative_position_ = "left"
                if self.relative_position_ in expect_outputs:
                    self.recognize_done_ = True
        except Exception as e:
            print(e)