#!/usr/bin/env python3

import rospy
from kuavo_msgs.msg import footPose, footPoseTargetTrajectories
from kuavo_msgs.srv import singleStepControl, singleStepControlRequest, singleStepControlResponse

import numpy as np
import os
import subprocess
import sys
import importlib
from enum import Enum

x_limit_range = [-0.15, 0.15]
y_limit_range = [-0.05, 0.05]
z_limit_range = [-0.05, 0.05]
yaw_limit_range = [-60, 60]

forward_single_step = [0.08, 0, 0, 0]
backward_single_step = [-0.08, 0, 0, 0]
left_single_step = [0, 0.05, 0, 0]
right_single_step = [0, -0.05, 0, 0]
left_rotate_single_step = [0, 0, 0, np.radians(20)]
right_rotate_single_step = [0, 0, 0, np.radians(-20)]

class StepType(Enum):
    FORWARD = '\x1b[A'
    BACKWARD = '\x1b[B'
    LEFT = '\x1b[D'
    RIGHT = '\x1b[C'
    ROTATE_LEFT = 'q'
    ROTATE_RIGHT = 'e'

STEP_MAPPING = {
    StepType.FORWARD: forward_single_step,
    StepType.BACKWARD: backward_single_step,
    StepType.LEFT: left_single_step,
    StepType.RIGHT: right_single_step,
    StepType.ROTATE_LEFT: left_rotate_single_step,
    StepType.ROTATE_RIGHT: right_rotate_single_step,
}

def install(package):
    print(f"Installing {package}...")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

def import_or_install(package):
    try:
        importlib.import_module(package)
    except ImportError:
        install(package)

import_or_install('rich')
from rich.console import Console

console = Console()
# foor_pose_target_trajectories_pub = rospy.Publisher('/humanoid_mpc_foot_pose_target_trajectories', footPoseTargetTrajectories, queue_size=10)
single_step_control_srv = rospy.ServiceProxy('/humanoid_single_step_control', singleStepControl)
rospy.sleep(1) # wait for the publisher to be initialized


def call_single_step_control(time_traj, torso_traj):
    req = singleStepControlRequest()
    foot_pose_target_trajectories = footPoseTargetTrajectories()
    foot_pose_target_trajectories.timeTrajectory = time_traj
    footPoseTrajectory = []
    for i in range(len(time_traj)):
        foot_pose_msg = footPose()
        foot_pose_msg.torsoPose = torso_traj[i]
        footPoseTrajectory.append(foot_pose_msg)
    foot_pose_target_trajectories.footPoseTrajectory = footPoseTrajectory
    req.foot_pose_target_trajectories = foot_pose_target_trajectories
    res = single_step_control_srv(req)
    rospy.sleep(0.5) # wait the trajectory to be executed
    if not res.success:
        console.print(f"[bold red]{res.message}[/bold red]")
    return res.success

def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_instructions():
    console.print("[bold magenta]===== Step Control =====[/bold magenta]")
    console.print("[bold blue]⬆️ ⬇️ ⬅️ ➡️ to move and 'q' and 'e' to rotate with default step size [/bold blue]")
    console.print("")
    console.print("[bold magenta]===== Each Step Safety Range =====[/bold magenta]")
    console.print("  [green]x: [-0.15, 0.15] m[/green]")
    console.print("  [green]y: [-0.05, 0.05] m[/green]")
    console.print("  [green]z: [-0.05, 0.05] m[/green]")
    console.print("  [green]yaw: [-60, 60] degrees[/green]")
    console.print("")
    console.print("[bold magenta]===== Input =====[/bold magenta]")
    console.print("[bold blue]Please input torso pose:[/bold blue] [italic](format: x y z yaw)[/italic]")
    console.print("[bold red]Enter 'c' or 'exit' to quit[/bold red]")

def get_user_input():
    print_instructions()
    console.print("[bold cyan]> [/bold cyan]", end="")
    try:
        user_input = input().split()
        if not user_input:
            return None

        if user_input[0] in [step.value for step in StepType]:
            return STEP_MAPPING[StepType(user_input[0])]

        if user_input[0] in ["exit", "c"]:
            console.print("[bold red]Exiting program...[/bold red]")
            raise KeyboardInterrupt

        if len(user_input) != 4:
            console.print("[bold red]Input format error, please enter 4 numbers (x y z yaw)[/bold red]")
            return None

        x, y, z = map(float, user_input[:3])
        yaw = float(user_input[3])  # Keep yaw in degrees for now

        # if not all([
        #     check_range(x, x_limit_range, "X"),
        #     check_range(y, y_limit_range, "Y"),
        #     check_range(z, z_limit_range, "Z"),
        #     check_range(yaw, yaw_limit_range, "Yaw")
        # ]):
        #     return None

        return [x, y, z, np.radians(yaw)]  # Convert yaw to radians here
    except ValueError:
        console.print("[bold red]Input format error, please ensure all inputs are numbers[/bold red]")
        return None

def check_range(value, range_limits, name):
    if not (range_limits[0] <= value <= range_limits[1]):
        console.print(f"[bold red]{name} value out of range, please enter a value between {range_limits[0]} and {range_limits[1]}[/bold red]")
        return False
    return True

def publish_user_input_pose():
    while not rospy.is_shutdown():
        try:
            torso_pose = get_user_input()
            if torso_pose is None:
                continue
        except KeyboardInterrupt:
            break
        dt = 1.2
        time_traj = [dt]
        torso_traj = []
        torso_traj.append(torso_pose)

        if call_single_step_control(time_traj, torso_traj):
            clear_console()

if __name__ == '__main__':
    try:
        rospy.init_node('foot_pose_publisher', anonymous=True)
        publish_user_input_pose()
    except rospy.ROSInterruptException:
        pass