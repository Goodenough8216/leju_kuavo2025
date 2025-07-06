from tongverselite.tcp import PersistentTcpClient, json2bin
import json
import transforms3d as t3d
import numpy as np

TO_RADIAN = np.pi / 180

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

class WalkingAgent:
    def __init__(self):
        self.ctrl_client_ = BipedWalkingCtrlClient(ip="0.0.0.0", port=8800)
        self.agent = None
        self.pos = None
        self.quat = None
        
    def turn_to_target_rotation(self, obs, target_rotation):
        self.agent = obs["agent"]
        self.pos = self.agent["body_state"]["world_pos"]
        self.quat = self.agent["body_state"]["world_orient"]
        rpy = t3d.euler.quat2euler(self.quat, axes="sxyz")
        diff_rot = target_rotation - rpy[2]
        if abs(diff_rot) > np.pi:
            diff_rot += 2 * np.pi if diff_rot < 0 else -2 * np.pi

        if diff_rot > 0:
            # call bipedal controller to get joint effort given a target angular velocity
            joint_efforts = self.ctrl_client_.get_cmd(
                obs, 0, 0, 0.2, False
            )
        else:
            # call bipedal controller to get joint effort given a target angular velocity
            joint_efforts = self.ctrl_client_.get_cmd(
                obs, 0, 0, -0.2, False
            )
        action = {
            "legs": {
                "ctrl_mode": joint_efforts["mode"],
                "joint_values": joint_efforts["effort"],
                "stiffness": [],
                "dampings": [],
            }
        }
        done = abs(diff_rot) < 5 * TO_RADIAN
        return action, done
    
    def walk_to_position(self, obs, position_goal):
         # get the position and rotation of the agent
        self.agent = obs["agent"]
        self.pos = self.agent["body_state"]["world_pos"]
        self.quat = self.agent["body_state"]["world_orient"]
        rpy = t3d.euler.quat2euler(self.quat, axes="sxyz")

        # calculate the difference between current position and the goal
        dx = position_goal[0] - self.pos[0]
        dy = position_goal[1] - self.pos[1]

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
                obs, 0.4, 0, 0, False
            )
        elif diff_rot > 0:
            # call bipedal controller to get joint effort given a target angular velocity
            joint_efforts = self.ctrl_client_.get_cmd(
                obs, 0.05, 0, 0.3, False
            )
        else:
            # call bipedal controller to get joint effort given a target angular velocity
            joint_efforts = self.ctrl_client_.get_cmd(
                obs, 0.05, 0, -0.3, False
            )

        # wrap joint effort into tongverse-lite action format
        action = {
            "legs": {
                "ctrl_mode": joint_efforts["mode"],
                "joint_values": joint_efforts["effort"],
                "stiffness": [],
                "dampings": [],
            }
        }

        done = (abs(dx) < 0.05) and (abs(dy) < 0.05)
        return action, done

    def stand_phase_action(self, obs):
        joint_efforts = self.ctrl_client_.get_cmd(
                obs, 0, 0, 0, False
        )
        action = {
            "legs": {
                "ctrl_mode": joint_efforts["mode"],
                "joint_values": joint_efforts["effort"],
                "stiffness": [],
                "dampings": [],
            }
        }
        return action
    
    def side_move_distance(self, obs, target, room):
         # get the position and rotation of the agent
        self.agent = obs["agent"]
        self.pos = self.agent["body_state"]["world_pos"]
        self.quat = self.agent["body_state"]["world_orient"]
        rpy = t3d.euler.quat2euler(self.quat, axes="sxyz")

        if room == "dinner":
            # calculate the difference between current position and the goal
            dy = target[1] - self.pos[1]

            if dy > 0:
                # call bipedal controller to get joint effort given a target velocity
                joint_efforts = self.ctrl_client_.get_cmd(
                    obs, 0, -0.1, 0, False
                )
            else:
                joint_efforts = self.ctrl_client_.get_cmd(
                    obs, 0, 0.1, 0, False
                )
        if room == "game":
            # calculate the difference between current position and the goal
            dy = target[1] - self.pos[1]

            if dy > 0:
                # call bipedal controller to get joint effort given a target velocity
                joint_efforts = self.ctrl_client_.get_cmd(
                    obs, 0, 0.1, 0, False
                )
            else:
                joint_efforts = self.ctrl_client_.get_cmd(
                    obs, 0, -0.1, 0, False
                )
        if room == "bed":
            dx = target[0] - self.pos[0]
            if dx < 0:
                joint_efforts = self.ctrl_client_.get_cmd(
                    obs, 0, 0.15, 0, False
                )
            else:
                joint_efforts = self.ctrl_client_.get_cmd(
                    obs, 0, -0.15, 0, False
                )

        # wrap joint effort into tongverse-lite action format
        action = {
            "legs": {
                "ctrl_mode": joint_efforts["mode"],
                "joint_values": joint_efforts["effort"],
                "stiffness": [],
                "dampings": [],
            }
        }
        if room == "bed":
            done = (abs(dx) < 0.05)
        else:
            done = (abs(dy) < 0.05)
        return action, done

    def walk_straight(self, obs, target, room):
         # get the position and rotation of the agent
        self.agent = obs["agent"]
        self.pos = self.agent["body_state"]["world_pos"]
        self.quat = self.agent["body_state"]["world_orient"]
        rpy = t3d.euler.quat2euler(self.quat, axes="sxyz")

        # calculate the difference between current position and the goal
        
        if room == "dinner":
            dx = target[0] - self.pos[0]
            if dx > 0:
                joint_efforts = self.ctrl_client_.get_cmd(
                    obs, -0.2, 0, 0, False
                )
            else:
                joint_efforts = self.ctrl_client_.get_cmd(
                    obs, 0.2, 0, 0, False
                )

        if room == "game":
            dx = target[0] - self.pos[0]
            if dx > 0:
                joint_efforts = self.ctrl_client_.get_cmd(
                    obs, 0.2, 0, 0, False
                )
            else:
                joint_efforts = self.ctrl_client_.get_cmd(
                    obs, -0.2, 0, 0, False
                )
        if room == "bed":
            dy = target[1] - self.pos[1]
            if dy > 0:
                joint_efforts = self.ctrl_client_.get_cmd(
                    obs, 0, 0, 0, False
                )
            else:
                joint_efforts = self.ctrl_client_.get_cmd(
                    obs, 0, 0, 0, False
                )
            

        # wrap joint effort into tongverse-lite action format
        action = {
            "legs": {
                "ctrl_mode": joint_efforts["mode"],
                "joint_values": joint_efforts["effort"],
                "stiffness": [],
                "dampings": [],
            }
        }

        if room == "bed":
            done = True
        else:
            done = (abs(dx) < 0.05)
        return action, done


    def walk_to_position2(self, obs, position_goal):
         # get the position and rotation of the agent
        self.agent = obs["agent"]
        self.pos = self.agent["body_state"]["world_pos"]
        self.quat = self.agent["body_state"]["world_orient"]
        rpy = t3d.euler.quat2euler(self.quat, axes="sxyz")

        # calculate the difference between current position and the goal
        dx = position_goal[0] - self.pos[0]
        dy = position_goal[1] - self.pos[1]

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
                obs, 0.4, 0, 0, False
            )
        elif diff_rot > 0:
            # call bipedal controller to get joint effort given a target angular velocity
            joint_efforts = self.ctrl_client_.get_cmd(
                obs, 0.07, 0, 0.3, False
            )
        else:
            # call bipedal controller to get joint effort given a target angular velocity
            joint_efforts = self.ctrl_client_.get_cmd(
                obs, 0.07, 0, -0.3, False
            )

        # wrap joint effort into tongverse-lite action format
        action = {
            "legs": {
                "ctrl_mode": joint_efforts["mode"],
                "joint_values": joint_efforts["effort"],
                "stiffness": [],
                "dampings": [],
            }
        }
        done = (abs(dx) < 0.15) and (abs(dy) < 0.15)
        return action, done