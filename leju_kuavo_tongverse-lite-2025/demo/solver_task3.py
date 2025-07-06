import sys
import os
import time
import numpy as np  # 添加 numpy 导入
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tongverse.env import Env, app
from demo.DemoController import DemoController

TASK_SEED = {1: 66, 2: 133, 3: 888}
TASK_ID = 3

def process_task_info(task_params, robot_init_pos):
    """处理任务信息，提取箱子和货架位置
    
    Args:
        task_params: 任务参数字典
        robot_init_pos: 机器人初始位置
        
    Returns:
        boxes_info: 箱子信息字典
        shelves_info: 货架信息字典
    """
    # 提取箱子位置
    boxes_info = {
        'red_box': task_params['initial_box_world_position'][0],
        'blue_box': task_params['initial_box_world_position'][1],
        'yellow_box': task_params['initial_box_world_position'][2]
    }
    print("original boxes_info", boxes_info)
    
    # 提取货架位置
    shelves_info = task_params['shelve_world_position']
    
    # 计算位置偏移
    robot_offset = robot_init_pos  # 使用机器人初始位置作为参考
    
    # 调整所有位置
    for key in boxes_info:
        print(f"key：{key}, boxes_info[key]：{boxes_info[key]} robot_offset：{robot_offset}")    

        boxes_info[key] = boxes_info[key] - robot_offset
        boxes_info[key][2] = 0.78
        print(f"key：{key}, boxes_info[key]：{boxes_info[key]}")    
    for key in shelves_info:
        shelves_info[key] = shelves_info[key] - robot_offset
        shelves_info[key][2] = 1.2
    return boxes_info, shelves_info


def set_box_and_shelf_positions(controller, task_params, robot_pos):
    # 处理任务信息
    boxes_info, shelves_info = process_task_info(task_params, robot_pos)
    
    # 设置第一个箱子(红箱子)和对应货架的位置
    select_color = 'red' # red blue yellow
    box_pose = {
        'x': float(boxes_info[select_color + '_box'][0]) - 0.1,
        'y': float(boxes_info[select_color + '_box'][1]),
        'z': float(boxes_info[select_color + '_box'][2])
    }
    
    shelf_pose = {
        'x': float(shelves_info[select_color + '_shelf'][0]),
        'y': float(shelves_info[select_color + '_shelf'][1]),
        'z': float(shelves_info[select_color + '_shelf'][2])
    }
    
    # # 设置位置
    success = controller.set_box_and_shelf_positions(box_pose, shelf_pose)
    
    if not success:
        print("Failed to set box and shelf positions")
        exit(0)

def main():
    # 初始化控制器
    controller = DemoController()
    
    # 启动命令行命令
    controller.start_launch()
    print("start launch")
    

    # 初始化环境
    env = Env(task_id=TASK_ID, seed=TASK_SEED[TASK_ID])
    env.reset()

    # 初始化机器人姿态，可以指定高度或使用默认值
    controller.launch_grab_box()
    
    # 获取任务信息
    task_params = env.get_task_params()
    print("TASK INFO")
    print(f"{task_params}")
    agent_params = env.get_robot_params()
    print("AGENT INFO")
    print(f"{agent_params}")
    
    # 创建初始空action
    action = {
        "arms": {
            "ctrl_mode": "position",
            "joint_values": [0.0] * 14,
            "stiffness": [50.0, 0, 0, 0, 50.0, 0, 0, 0, 50.0, 0, 0, 0, 0, 0],
            "dampings": [0.0] * 14,
        },
        "legs": {
            "ctrl_mode": "effort",
            "joint_values": np.zeros(12),
            "stiffness": None,
            "dampings": None,
        },
        "head": {
            "ctrl_mode": "position",
            "joint_values": np.zeros(2),
            "stiffness": None,
            "dampings": None,
        }
    }

    i = 0   
    print("Press 's' to grab box...")
    
    while app.is_running():
        # 执行一步仿真获取观测数据
        obs, is_done = env.step(action)
        # print(f"step {i}, time: {obs['imu_data']['imu_time']}")

        if i == 0:
            # 设置箱子和货架的位置
            robot_pos = obs["Kuavo"]["body_state"]["world_position"]
            set_box_and_shelf_positions(controller, task_params, robot_pos)
            

        # 处理观测数据并发布，同时等待新的action
        action = controller.get_action(obs)
        # print("get action")
        # 检查任务是否完成
        if is_done:
            print(obs["extras"])
            break
        i += 1
        
    controller.cleanup()

if __name__ == "__main__":
    try:
        main()
    finally:
        app.close() 
