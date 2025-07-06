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
        self.ctrl_client_ = BipedWalkingCtrlClient(ip="0.0.0.0", port=8800)
        

    def next_action(self, obs: dict) -> dict:
        agent = obs["agent"]
        pos = agent["body_state"]["world_pos"]
        quat = agent["body_state"]["world_orient"]
        rpy = t3d.euler.quat2euler(quat, axes="sxyz")
        obstacle = obs["obstacle"]
        obstacle_pos = obstacle["position"]
        obstacle_quat = obstacle["orientation"]
        dy = obstacle_pos[1] - pos[1]
        if abs(dy)<=0.65:
            joint_efforts = self.ctrl_client_.get_cmd(obs, 0, 0, 0, 2)
        else:
            joint_efforts = self.ctrl_client_.get_cmd(obs, 0, 0, 0, False)
        action = {
            "legs": {
                "ctrl_mode": joint_efforts["mode"],
                "joint_values": joint_efforts["effort"],
                "stiffness": [],
                "dampings": [],
            }
        }
        return action
