import sys
import os
import time
import numpy as np  # 添加 numpy 导入
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tongverse.env import Env, app
from demo.DemoController import DemoController

TASK_SEED = {1: 66, 2: 133, 3: 888}
TASK_ID = 1

def main():
    # 初始化控制器
    controller = DemoController()
    
    # 启动命令行命令
    controller.start_launch()
    print("start launch")

    # 初始化环境
    env = Env(task_id=TASK_ID, seed=TASK_SEED[TASK_ID])
    env.reset()

    # 打印任务信息
    task_params = env.get_task_params()
    print("TASK INFO")
    print(f"{task_params}")
    agent_params = env.get_robot_params()
    print("AGENT INFO")
    print(f"{agent_params}")

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
    while app.is_running():
        # 执行一步仿真获取观测数据
        obs, is_done = env.step(action)
        # print(f"step {i}, time: {obs['imu_data']['imu_time']}")
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
