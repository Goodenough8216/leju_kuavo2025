#!/usr/bin/env python3
# coding: utf-8
"""
This layer is responsible for robot state transitions.
The robot has three states:
- stance: Standing still state
- walk: Walking state 
- trot: Trotting state

State transitions are managed by a state machine that ensures valid transitions between states.
The state machine enforces the following transitions:
- stance <-> walk
- stance <-> trot
- walk <-> trot

Each state has an entry callback that handles initialization when entering that state.
"""


import time
import math
import rospy
import threading
import numpy as np
from typing import Tuple
from transitions import Machine, State

from kuavo_humanoid_sdk.interfaces.data_types import KuavoArmCtrlMode, KuavoIKParams, KuavoPose
from kuavo_humanoid_sdk.kuavo.core.ros.control import KuavoRobotControl
from kuavo_humanoid_sdk.kuavo.core.ros.state import KuavoRobotStateCore
from kuavo_humanoid_sdk.kuavo.core.ros.param import make_robot_param, kuavo_ros_param
from kuavo_humanoid_sdk.common.logger import SDKLogger

# Define robot states
ROBOT_STATES = [
    State(name='stance', on_enter=['_on_enter_stance']),
    State(name='walk', on_enter=['_on_enter_walk']), 
    State(name='trot', on_enter=['_on_enter_trot']),
    State(name='custom_gait', on_enter=['_on_enter_custom_gait'])
]

# Define state transitions
ROBOT_TRANSITIONS = [
    {'trigger': 'to_stance', 'source': ['walk', 'trot', 'custom_gait'], 'dest': 'stance'},
    {'trigger': 'to_walk', 'source': ['stance', 'trot', 'custom_gait'], 'dest': 'walk'},
    {'trigger': 'to_trot', 'source': ['stance', 'walk', 'custom_gait'], 'dest': 'trot'},
    {'trigger': 'to_custom_gait', 'source': ['stance', 'custom_gait'], 'dest': 'custom_gait'},
]

class KuavoRobotCore:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(KuavoRobotCore, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self.machine = Machine(
                model=self,
                states=ROBOT_STATES,
                transitions=ROBOT_TRANSITIONS,
                initial='stance',
                send_event=True
            )
            # robot control
            self._control = KuavoRobotControl()
            self._rb_state = KuavoRobotStateCore()
            self._arm_ctrl_mode = KuavoArmCtrlMode.AutoSwing
            # register gait changed callback
            self._rb_state.register_gait_changed_callback(self._humanoid_gait_changed)
            # initialized
            self._initialized = True

    def initialize(self, debug: bool=False)->bool:
        """
         raise RuntimeError if initialize failed.
        """
        try:
            info = make_robot_param()
            # init state by gait_name
            gait_name = self._rb_state.gait_name()
            if gait_name is not None:
                to_method = f'to_{gait_name}'
                if hasattr(self, to_method):
                    SDKLogger.debug(f"[Core] initialize state: {gait_name}")
                    # Call the transition method if it exists
                    getattr(self, to_method)()
            else:
                SDKLogger.warn(f"[Core] gait_name is None, use default `stance`")
            # init arm control mode
            arm_ctrl_mode = self._rb_state.arm_control_mode
            if arm_ctrl_mode is not None:
                self._arm_ctrl_mode = arm_ctrl_mode
                SDKLogger.debug(f"[Core] initialize arm control mode: {arm_ctrl_mode}")
        except Exception as e:
            raise RuntimeError(f"[Core] initialize failed: \n"
                             f"{e}, please check the robot is launched, "
                             f"e.g. `roslaunch humanoid_controllers load_kuavo_real.launch`")
        rb_info = make_robot_param()
        success, err_msg = self._control.initialize(eef_type=rb_info["end_effector_type"], debug=debug)
        if not success:
            raise RuntimeError(f"[Core] initialize failed: \n{err_msg}, please check the robot is launched, "
                             f"e.g. `roslaunch humanoid_controllers load_kuavo_real.launch`")
        return True

    """ ----------------------- Machine State -----------------------"""
    def _on_enter_stance(self, event):
        previous_state = event.transition.source
        if self.state  == previous_state:
            SDKLogger.debug(f"[Core] [StateMachine] State unchanged: already in stance state")
            return
        
        SDKLogger.debug(f"[Core] [StateMachine] Entering stance state, from {previous_state}")
        if previous_state == 'walk':
            self._control.robot_walk(0.0, 0.0, 0.0) # stop walk state
            start_time = time.time()
            # slow down walk
            try:
                while time.time() - start_time < 1.5:
                    self._control.robot_walk(0.0, 0.0, 0.0)
                    # linear_x, linear_y, angular_z
                    if (abs(self._rb_state.odom_data.linear[0]) < 0.05 and abs(self._rb_state.odom_data.linear[1]) < 0.08 
                        and abs(self._rb_state.odom_data.angular[2]) < 0.05):
                        SDKLogger.debug(f"walk stop, time_cost:{time.time() - start_time}, odom_data:{self._rb_state.odom_data.linear}")
                        break
                    # SDKLogger.debug(f"kuavo robot linear: {self._rb_state.odom_data.linear}")
                    time.sleep(0.1)
            except KeyboardInterrupt:
                pass
            self._control.robot_stance()
        else:
            self._control.robot_stance() 
        time.sleep(0.5)

    def _on_enter_walk(self, event):
        previous_state = event.transition.source
        if self.state  == previous_state:
            SDKLogger.debug(f"[Core] [StateMachine] State unchanged: already in walk state")
            return
        SDKLogger.debug(f"[Core] [StateMachine] Entering walk state, from {previous_state}")

    def _on_enter_trot(self, event):
        previous_state = event.transition.source
        if self.state  == previous_state:
            SDKLogger.debug(f"[Core] [StateMachine] State unchanged: already in trot state")
            return
        SDKLogger.debug(f"[Core] [StateMachine] Entering trot state, from {previous_state}")
        self._control.robot_trot()

    def _on_enter_custom_gait(self, event):
        previous_state = event.transition.source
        if self.state  == previous_state:
            SDKLogger.debug(f"[Core] [StateMachine] State unchanged: already in custom_gait state")
            return
        SDKLogger.debug(f"[Core] [StateMachine] Entering custom_gait state, from {previous_state}")
    """ -------------------------------------------------------------"""

    """ --------------------------- Control -------------------------"""
    def walk(self, linear_x:float, linear_y:float, angular_z:float)-> bool:
        if self.state != 'walk':
            self.to_walk()
        
        # +-0.4, +-0.2, +-0.4 => linear_x, linear_y, angular_z
        limited_linear_x = min(0.4, abs(linear_x)) * (1 if linear_x >= 0 else -1)
        limited_linear_y = min(0.2, abs(linear_y)) * (1 if linear_y >= 0 else -1)
        limited_angular_z = min(0.4, abs(angular_z)) * (1 if angular_z >= 0 else -1)
        return self._control.robot_walk(limited_linear_x, limited_linear_y, limited_angular_z)
    
    def squat(self, height:float, pitch:float)->bool:
        if self.state != 'stance':
            SDKLogger.warn(f"[Core] control torso height failed, robot is not in stance state({self.state})!")
            return False
        
        # Limit height range to [-0.3, 0.0]
        limited_height = min(0.0, max(-0.3, height))
        if height > 0.0 or height < -0.3:
            SDKLogger.warn(f"[Core] height {height} exceeds limit [-0.3, 0.0], will be limited")
        
        # Limit pitch range to [-0.4, 0.4]
        limited_pitch = min(0.4, max(-0.4, pitch))
        if abs(pitch) > 0.4:
            SDKLogger.warn(f"[Core] pitch {pitch} exceeds limit [-0.4, 0.4], will be limited")

        return self._control.control_torso_height(limited_height, limited_pitch)

    def step_control(self, target_pose:list, dt:float=0.4, is_left_first_default:bool=True, collision_check:bool=True)->bool:
        """
        Control the robot's motion by step.
        Raises:
            ValueError: If target_pose length is not 4.
            RuntimeError: If the robot is not in stance state when trying to control step motion.
        """
        if len(target_pose) != 4:
            raise ValueError(f"[Core] target_pose length must be 4, but got {len(target_pose)}")
    
        # Wait up to 1.0s for stance state
        wait_time = 0
        while self._rb_state.gait_name() != 'stance' and wait_time < 1.0:
            time.sleep(0.1)
            wait_time += 0.1
            
        if self._rb_state.gait_name() != 'stance':
            raise RuntimeError(f"[Core] control robot step failed, robot is not in stance state, {self._rb_state.gait_name()}!")

        if self.state != 'stance':
            raise RuntimeError(f"[Core] control robot step failed, robot is not in stance state({self.state})!")
        
        com_height = self._rb_state.com_height
        # print(f"[Core] Current COM height: {com_height:.2f}m")
        # Check height limits based on current COM height
        MIN_COM_HEIGHT = 0.66  # Minimum allowed COM height in meters
        MAX_COM_HEIGHT = 0.86  # Maximum allowed COM height in meters

        # Validate COM height constraints
        if target_pose[2] < 0 and com_height < MIN_COM_HEIGHT:
            SDKLogger.warn(f"[Core] Cannot squat lower: COM height {com_height:.2f}m below minimum {MIN_COM_HEIGHT}m")
            return None
        
        if target_pose[2] > 0 and com_height > MAX_COM_HEIGHT:
            SDKLogger.warn(f"[Core] Cannot stand higher: COM height {com_height:.2f}m above maximum {MAX_COM_HEIGHT}m")
            return None

        # Ensure target height is within allowed range if height change requested
        if target_pose[2] != 0:
            target_height = com_height + target_pose[2]
            if target_height < MIN_COM_HEIGHT:
                SDKLogger.warn(f"[Core] Target height {target_height:.2f}m below minimum {MIN_COM_HEIGHT}m, limiting")
                target_pose[2] = MIN_COM_HEIGHT - com_height
            elif target_height > MAX_COM_HEIGHT:
                SDKLogger.warn(f"[Core] Target height {target_height:.2f}m above maximum {MAX_COM_HEIGHT}m, limiting") 
                target_pose[2] = MAX_COM_HEIGHT - com_height
        
        # TODO(kuavo): 根据实物测试来调整....
        if com_height > 0.82:
            max_x_step = 0.20
            max_y_step = 0.20
            max_yaw_step = 90
        else:
            max_x_step = 0.10
            max_y_step = 0.10
            max_yaw_step = 30
        
        body_poses = []
        
        # 计算目标点到原点的距离和朝向
        target_dist_x = abs(target_pose[0])
        target_dist_y = abs(target_pose[1])
        target_yaw = target_pose[3] * 180 / math.pi  # Convert yaw from radians to degrees
        
        # 计算需要的步数(考虑x位移、y位移和转角)
        steps_for_x = int(np.ceil(target_dist_x / max_x_step))
        steps_for_y = int(np.ceil(target_dist_y / max_y_step))
        steps_for_yaw = int(np.ceil(abs(target_yaw) / max_yaw_step))
        steps_needed = max(steps_for_x, steps_for_y, steps_for_yaw)
        # print(f"[Core] Steps needed - X: {steps_for_x}, Y: {steps_for_y}, Yaw: {steps_for_yaw}, Total: {steps_needed}")
        
        # 计算每一步的增量
        dx = target_pose[0] / steps_needed
        dy = target_pose[1] / steps_needed
        dyaw = target_yaw / steps_needed
        
        # 分解为多个小步,沿着直线路径前进并逐步调整朝向
        for i in range(steps_needed):
            x = dx * (i+1)
            y = dy * (i+1)
            z = target_pose[2]
            yaw = dyaw * (i+1)
            body_poses.append([x, y, 0.0, yaw])
        
        # print("target_pose:", target_pose)
        # print("body_poses:", body_poses)

        if not self._control.step_control(body_poses, dt, is_left_first_default, collision_check):
            return False
        
        # Wait for gait to switch to custom_gait
        start_time = time.time()
        while not self._rb_state.is_gait('custom_gait'):
            if time.time() - start_time > 1.0:  # 1.0s timeout
                SDKLogger.warn("[Core] Timeout waiting for gait to switch to custom_gait")
                return False
            time.sleep(0.01)

        return True

    def execute_gesture(self, gestures:list)->bool:
        return self._control.execute_gesture(gestures)
    
    def get_gesture_names(self)->list:
        return self._control.get_gesture_names()

    def control_robot_dexhand(self, left_position:list, right_position:list)->bool:
        return self._control.control_robot_dexhand(left_position, right_position)
    
    def control_leju_claw(self, postions:list, velocities:list=[90, 90], torques:list=[1.0, 1.0]) ->bool:
        return self._control.control_leju_claw(postions, velocities, torques)

    def control_robot_head(self, yaw:float, pitch:float)->bool:
        # Convert radians to degrees
        yaw_deg = yaw * 180 / math.pi
        pitch_deg = pitch * 180 / math.pi
        return self._control.control_robot_head(yaw_deg, pitch_deg)
    
    def control_robot_arm_traj(self, joint_data:list)->bool:
        if self.state != 'stance':
            raise RuntimeError(f"[Core] control_robot_arm_traj failed: robot must be in stance state, current state: {self.state}")
        
        # change to external control mode  
        if self._arm_ctrl_mode != KuavoArmCtrlMode.ExternalControl:
            SDKLogger.debug("[Core] control_robot_arm_traj, current arm mode != ExternalControl, change it.")
            if not self.change_robot_arm_ctrl_mode(KuavoArmCtrlMode.ExternalControl):
                SDKLogger.warn("[Core] control_robot_arm_traj failed, change robot arm ctrl mode failed!")
                return False
        return self._control.control_robot_arm_traj(joint_data)
    
    def control_robot_arm_target_poses(self, times:list, joint_q:list)->bool:
        if self.state != 'stance':
            raise RuntimeError("[Core] control_robot_arm_target_poses failed: robot must be in stance state")
        
        if self._arm_ctrl_mode != KuavoArmCtrlMode.ExternalControl:
            SDKLogger.debug("[Core] control_robot_arm_target_poses, current arm mode != ExternalControl, change it.")
            if not self.change_robot_arm_ctrl_mode(KuavoArmCtrlMode.ExternalControl):
                SDKLogger.warn("[Core] control_robot_arm_target_poses failed, change robot arm ctrl mode failed!")
                return False
            
        return self._control.control_robot_arm_target_poses(times, joint_q)

    def change_robot_arm_ctrl_mode(self, mode:KuavoArmCtrlMode)->bool:
        timeout = 1.0
        count = 0
        while self._rb_state.arm_control_mode != mode:
            SDKLogger.debug(f"[Core] Change robot arm control  from {self._rb_state.arm_control_mode} to {mode}, retry: {count}")
            self._control.change_robot_arm_ctrl_mode(mode)
            if self._rb_state.arm_control_mode == mode:
                break
            if timeout <= 0:
                SDKLogger.warn("[Core] Change robot arm control mode timeout!")
                return False
            timeout -= 0.1
            time.sleep(0.1)
            count += 1
        
        if not hasattr(self, '_arm_ctrl_mode_lock'):
            self._arm_ctrl_mode_lock = threading.Lock()
        with self._arm_ctrl_mode_lock:
            # 手臂控制模式切换成功，更新当前手臂控制模式
            self._arm_ctrl_mode = mode # update arm ctrl mode

        return True
    
    def robot_arm_reset(self)->bool:
        if self.state != 'stance':
            SDKLogger.warn("[Core] robot arm reset failed, robot is not in stance state!")
            return
        
        # init_pos = [0.0]*14
        # if not self.control_robot_arm_target_poses([1.5], [init_pos]):
        #     SDKLogger.warn("[Core] robot arm reset failed, control robot arm traj failed!")
        #     return False
        
        return self.change_robot_arm_ctrl_mode(KuavoArmCtrlMode.AutoSwing)
        
    """ ------------------------------------------------------------------------"""
    """ Arm Forward kinematics && Arm Inverse kinematics """
    def arm_ik(self, 
               l_eef_pose: KuavoPose, 
               r_eef_pose: KuavoPose, 
               l_elbow_pos_xyz: list = [0.0, 0.0, 0.0],
               r_elbow_pos_xyz: list = [0.0, 0.0, 0.0],
               arm_q0: list = None,
               params: KuavoIKParams=None) -> list:
        return self._control.arm_ik(l_eef_pose, r_eef_pose, l_elbow_pos_xyz, r_elbow_pos_xyz, arm_q0, params)
    

    def arm_fk(self, q: list) -> Tuple[KuavoPose, KuavoPose]:
        return self._control.arm_fk(q)
    
    """ Callbacks """
    def _humanoid_gait_changed(self, current_time: float, gait_name: str):
        if self.state != gait_name:
            # Check if to_$gait_name method exists
            to_method = f'to_{gait_name}'
            if hasattr(self, to_method):
                SDKLogger.debug(f"[Core] Received gait change notification: {gait_name} at time {current_time}")
                # Call the transition method if it exists
                getattr(self, to_method)()
