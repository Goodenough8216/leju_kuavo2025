import json
import numpy as np
import transforms3d as t3d

from tongverselite.tcp import PersistentTcpClient, json2bin
from tongverselite.solver import TaskSolverBase


class BipedWalkingCtrlClient(PersistentTcpClient):
    def send_request(self, msg):
        data_bin = json2bin(msg)
        return json.loads(self.send(data_bin).decode("ascii"))

    def get_cmd(self, obs, vx, vy, theta, state):
        obs_agent = obs["agent"]
        q_leg = obs_agent["joint_state"]["legs_positions"]
        dq_leg = obs_agent["joint_state"]["legs_velocities"]

        q_arm = obs_agent["joint_state"]["arms_positions"]
        dq_arm = obs_agent["joint_state"]["arms_velocities"]

        p_wb = obs_agent["body_state"]["world_pos"]
        quat_wb = obs_agent["body_state"]["world_orient"]
        v_wb = obs_agent["body_state"]["linear_velocities"]
        w_wb = obs_agent["body_state"]["angular_velocities"]

        msg = {
            "q_leg": q_leg.tolist(),
            "dq_leg": dq_leg.tolist(),
            "q_arm": q_arm.tolist(),
            "dq_arm": dq_arm.tolist(),
            "p_wb": p_wb.tolist(),
            "quat_wb": quat_wb.tolist(),
            "v_wb": v_wb.tolist(),
            "w_wb": w_wb.tolist(),
            "command": [vx, vy, theta],
            "change_state": state,
        }
        joint_efforts = self.send_request(msg)

        return joint_efforts


class TaskSolver(TaskSolverBase):
    def __init__(self) -> None:
        super().__init__()
        # create a controller client that request joint effort commands from the server
        # note that ip MUST be "0.0.0.0" and port MUST be 8800
        self.ctrl_client_ = BipedWalkingCtrlClient(ip="0.0.0.0", port=8800)
        # self.goal_ = np.array([10.94795, -9.1162, 0.1])
        # the goal position as given in the task specification
        self.goal_position = np.array([-0.00454,-9.43024,0.1]) #KUAVO所在位置,抓取阀门的位置 之前容易转圈的位置[ 0.5820786, -9.300483 ,  0.7742238]
        self.goal_position_end = np.array([0.33159947, -8.925875  ,  0.780797]) 
 

        # 位置1_world_pos  0.8057828, -9.126119 ,  0.7783028
        # 角度1_world_orient  0.19703159, -0.05712217,  0.0111281 ,  0.97866833

        # pos2 'world_pos': array([ 0.5820786, -9.300483 ,  0.7742238]
        # 'world_orient': array([ 0.20658003, -0.08499989,  0.00352739,  0.97472435]

        # last 'world_pos': array([ 0.33159947, -8.925875  ,  0.780797  ]

        # 阀门中心位置 [-0.00454,-9.43024,0.1]
    def next_action(self, obs: dict) -> dict:
        agent = obs["agent"]
    
        pos = agent["body_state"]["world_pos"]
        quat = agent["body_state"]["world_orient"]
        rpy = t3d.euler.quat2euler(quat, axes="sxyz")
        
        pick = obs["pick"]       
        if(pick):
            #print("pick")
            dx_end = self.goal_position_end[0] - pos[0]
            dy_end = self.goal_position_end[1] - pos[1]
            # target_orient = t3d.euler.quat2euler([ 0.20658003, -0.08499989,  0.00352739,  0.97472435],axes="sxyz") #期望机器人抓住阀门后，面对终点朝向
            target_orient = np.arctan2(dy_end, dx_end)
            diff_orient = target_orient - rpy[2]
            # if abs(diff_orient) > np.pi:
            #     diff_orient += 2 * np.pi if diff_orient < 0 else -2 * np.pi
            # if abs(diff_orient) < np.pi / 90:
            #     #print("go")
            #     joint_efforts = self.ctrl_client_.get_cmd(obs, 0.1, 0, 0, False)
            # elif diff_orient > 0:
            #     #print("turn left")
            #     joint_efforts = self.ctrl_client_.get_cmd(obs,0,0,0.1,False)
            # else:
            #     #print("turn right")
            #     joint_efforts = self.ctrl_client_.get_cmd(obs,0,0,-0.1,False)
            joint_efforts = self.ctrl_client_.get_cmd(obs,0,-0.15,0,False)
            action = {
                "legs": {
                        "ctrl_mode": joint_efforts["mode"],
                        "joint_values": joint_efforts["effort"],
                        "stiffness": [],
                        "dampings": [],
                    },
                "arms":{"ctrl_mode":"position",
                        "joint_values":[-20/180*np.pi,0,0,-90/180*np.pi,0,0,0,0],
                        "stiffness":[10,10,10,10,20,20,20,20],
                        "dampings":[1,1,1,1,2,2,2,2],
                        },
            
                "pick":"left_hand",
                "release":False
            }
        else:
            dx = self.goal_position[0] - pos[0]
            dy = self.goal_position[1] - pos[1]
            #8.978 39.181 -5.652
            # arm_euler = t3d.euler.quat2euler([0.93937,0.05716,0.3375,-0.02009],axes="sxyz")
            # calculate the target direction
            target_dir = np.arctan2(dy, dx)

            # calculate the difference between the target direction and the current direction
            diff_rot = target_dir - rpy[2]

            # round the rotation difference, to avoid the agent to rotate more than 180 degrees
            if abs(diff_rot) > np.pi:
                diff_rot += 2 * np.pi if diff_rot < 0 else -2 * np.pi

            # if the agent is nearly facing the goal, move forward
            if abs(diff_rot) < np.pi / 90:
                # call bipedal controller to get joint effort given a target velocity
                joint_efforts = self.ctrl_client_.get_cmd(
                    obs, 0.25, 0, 0, False
                )
            elif diff_rot > 0:
                # call bipedal controller to get joint effort given a target angular velocity
                # left
                joint_efforts = self.ctrl_client_.get_cmd(
                    obs, 0, 0, 0.3, False
                )
            else:
                # call bipedal controller to get joint effort given a target angular velocity
                #right
                joint_efforts = self.ctrl_client_.get_cmd(
                    obs, 0, 0, -0.3, False
                )
        # wrap joint effort into tongverse-lite action format
            action = {
                "legs": {
                    "ctrl_mode": joint_efforts["mode"],
                    "joint_values": joint_efforts["effort"],
                    "stiffness": [],
                    "dampings": [],
                },
                "arms":{"ctrl_mode":"position",
                        "joint_values":[-30/180*np.pi,0,0,-85/180*np.pi,0,0,0,0],
                        "stiffness":None,
                        "dampings":[],
                        },
                "pick":"left_hand",
                "release":False
            }

        return action
