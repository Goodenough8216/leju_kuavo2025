
import rospy
import numpy as np
from typing import Tuple
from kuavo_humanoid_sdk.common.logger import SDKLogger
from kuavo_humanoid_sdk.interfaces.data_types import KuavoArmCtrlMode, KuavoIKParams, KuavoPose
from kuavo_humanoid_sdk.kuavo.core.ros.sat_utils import RotatingRectangle

from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState, Joy
from kuavo_msgs.msg import gestureTask,robotHandPosition, robotHeadMotionData, armTargetPoses, switchGaitByName
from kuavo_msgs.msg import footPose, footPoseTargetTrajectories
from kuavo_msgs.srv import (gestureExecute, gestureExecuteRequest,gestureList, gestureListRequest,
                            controlLejuClaw, controlLejuClawRequest, changeArmCtrlMode, changeArmCtrlModeRequest)
from motion_capture_ik.msg import twoArmHandPoseCmd, ikSolveParam
from motion_capture_ik.srv import twoArmHandPoseCmdSrv, fkSrv

class ControlEndEffector:
    def __init__(self, eef_type: str = "qiangnao"):
        self._pubs = []
        self._eef_type = eef_type
        if eef_type == "qiangnao":
            self._pub_ctrl_robot_hand = rospy.Publisher('/control_robot_hand_position', robotHandPosition, queue_size=10)                
            self._pubs.append(self._pub_ctrl_robot_hand)

    def connect(self, timeout:float=1.0)-> bool:
        start_time = rospy.Time.now()
        
        success = True
        for pub in self._pubs:
            while pub.get_num_connections() == 0:
                if (rospy.Time.now() - start_time).to_sec() > timeout:
                    SDKLogger.error(f"Timeout waiting for {pub.name} connection")
                    success = False
                    break
                rospy.sleep(0.1)

        return success

    """ Control Kuavo Robot Dexhand """
    def pub_control_robot_dexhand(self, left_position:list, right_position:list)->bool:
        if self._eef_type != 'qiangnao':
            SDKLogger.warning(f"{self._eef_type} not support control dexhand")
            return False
        try :
            hand_pose_msg = robotHandPosition()
            hand_pose_msg.left_hand_position = bytes(left_position)
            hand_pose_msg.right_hand_position = bytes(right_position)
            self._pub_ctrl_robot_hand.publish(hand_pose_msg)
            SDKLogger.debug(f"publish robot dexhand: {left_position}, {right_position}")
            return True
        except Exception as e:
            SDKLogger.error(f"publish robot dexhand: {e}")
            return False
        
    def srv_execute_gesture(self, gestures:list)->bool:
        if self._eef_type != 'qiangnao':
            SDKLogger.warning(f"{self._eef_type} not support control dexhand")
            return False
        try:
            service_name = 'gesture/execute'
            rospy.wait_for_service(service=service_name, timeout=2.0)
            # create service proxy
            gesture_service = rospy.ServiceProxy(service_name, gestureExecute)
            
            # request
            request = gestureExecuteRequest()
            for gs in gestures:
                gesture = gestureTask(gesture_name=gs["gesture_name"], hand_side=gs["hand_side"])
                request.gestures.append(gesture)
            
            # call service
            response = gesture_service(request)
            if not response.success:
                SDKLogger.error(f"Failed to execute gesture '{gestures}': {response.message}")
            return response.success
        except rospy.ServiceException as e:
            SDKLogger.error(f"Service call failed: {e}")
        except rospy.ROSException as e:
            SDKLogger.error(f"Service call failed: {e}")
        except Exception as e:
            SDKLogger.error(f"Service call failed: {e}")
        return False

    def srv_get_gesture_names(self)->list:
        if self._eef_type != 'qiangnao':
            SDKLogger.warning(f"{self._eef_type} not support control dexhand")
            return []
        try:
            service_name = 'gesture/list'
            rospy.wait_for_service(service=service_name, timeout=2.0)
            gesture_service = rospy.ServiceProxy(service_name, gestureList)
            request = gestureListRequest()
            response = gesture_service(request)
            gestures = []
            for gesture_info in response.gesture_infos:
                gestures.append(gesture_info.gesture_name)
                for alias in gesture_info.alias:
                    gestures.append(alias)
            return list(set(gestures))
        except rospy.ServiceException as e:
            SDKLogger.error(f"Service call failed: {e}")
        except Exception as e:
            SDKLogger.error(f"Service call failed: {e}")
        return []

    def srv_control_leju_claw(self, postions:list, velocities:list, torques:list) ->bool:
        if self._eef_type != 'lejuclaw':
            SDKLogger.warning(f"{self._eef_type} not support control lejuclaw.")
            return False
        try:
            # ros service
            service_name = 'control_robot_leju_claw'
            rospy.wait_for_service(service_name, timeout=2.0)
            control_lejucalw_srv = rospy.ServiceProxy(service_name, controlLejuClaw)
            
            # request
            request = controlLejuClawRequest()
            request.data.position = postions
            request.data.velocity = velocities
            request.data.effort = torques
            
            response = control_lejucalw_srv(request)
            if not response.success:
                SDKLogger.error(f"Failed to control leju claw: {response.message}")
            return response.success
        except rospy.ServiceException as e:
            SDKLogger.error(f"Service `control_robot_leju_claw` call failed: {e}")
        except rospy.ROSException as e:
            SDKLogger.error(f"Service `control_robot_leju_claw` call failed: {e}")
        except Exception as e:
            SDKLogger.error(f"Service `control_robot_leju_claw` call failed: {e}")
        return False

""" Control Robot Arm"""
class ControlRobotArm:
    def __init__(self):
        self._pub_ctrl_arm_traj = rospy.Publisher('/kuavo_arm_traj', JointState, queue_size=10)
        self._pub_ctrl_arm_target_poses = rospy.Publisher('/kuavo_arm_target_poses', armTargetPoses, queue_size=10)

    def connect(self, timeout:float=1.0)-> bool:
        start_time = rospy.Time.now()
        publishers = [
            (self._pub_ctrl_arm_traj, "arm trajectory publisher"),
            (self._pub_ctrl_arm_target_poses, "arm target poses publisher")
        ]
        
        success = True
        for pub, name in publishers:
            while pub.get_num_connections() == 0:
                if (rospy.Time.now() - start_time).to_sec() > timeout:
                    SDKLogger.error(f"Timeout waiting for {name} connection, '{pub.name}'")
                    success = False
                    break
                rospy.sleep(0.1)
        return success

    def pub_control_robot_arm_traj(self, joint_q: list)->bool:
        try:
            msg = JointState()
            msg.name = ["arm_joint_" + str(i) for i in range(0, 14)]
            msg.header.stamp = rospy.Time.now()
            msg.position = 180.0 / np.pi * np.array(joint_q) # convert to degree
            self._pub_ctrl_arm_traj.publish(msg)
            return True
        except Exception as e:
            SDKLogger.error(f"publish robot arm traj: {e}")
        return False
        
    def pub_arm_target_poses(self, times:list, joint_q:list):
        try:
            msg = armTargetPoses()
            msg.times = times
            for i in range(len(joint_q)):
                degs = [q for q in joint_q[i]]
                msg.values.extend(degs)
            self._pub_ctrl_arm_target_poses.publish(msg)
            return True
        except Exception as e:
            SDKLogger.error(f"publish arm target poses: {e}")
        return False

    def srv_change_arm_ctrl_mode(self, mode: KuavoArmCtrlMode)->bool:
        try:
            rospy.wait_for_service('/arm_traj_change_mode', timeout=2.0)
            change_arm_ctrl_mode_srv = rospy.ServiceProxy('/arm_traj_change_mode', changeArmCtrlMode)
            req = changeArmCtrlModeRequest()
            req.control_mode = mode.value
            resp = change_arm_ctrl_mode_srv(req)
            return resp.result
        except rospy.ServiceException as e:
            SDKLogger.error(f"Service call failed: {e}")
        except Exception as e:
            SDKLogger.error(f"[Error] change arm ctrl mode: {e}")
        return False
    
    def srv_get_arm_ctrl_mode(self)-> KuavoArmCtrlMode:
        try:
            rospy.wait_for_service('/humanoid_get_arm_ctrl_mode')
            get_arm_ctrl_mode_srv = rospy.ServiceProxy('/humanoid_get_arm_ctrl_mode', changeArmCtrlMode)
            req = changeArmCtrlModeRequest()
            resp = get_arm_ctrl_mode_srv(req)
            return KuavoArmCtrlMode(resp.control_mode)
        except rospy.ServiceException as e:
            SDKLogger.error(f"Service call failed: {e}")
        except Exception as e:
            SDKLogger.error(f"[Error] get arm ctrl mode: {e}")
        return None
        
""" Control Robot Head """
class ControlRobotHead:
    def __init__(self):
        self._pub_ctrl_robot_head = rospy.Publisher('/robot_head_motion_data', robotHeadMotionData, queue_size=10)

    def connect(self, timeout:float=1.0)->bool:
        start_time = rospy.Time.now()
        publishers = [
            # (self._pub_ctrl_robot_head, "robot head publisher") # not need check!
        ]
        for pub, name in publishers:
            while pub.get_num_connections() == 0:
                if (rospy.Time.now() - start_time).to_sec() > timeout:
                    SDKLogger.error(f"Timeout waiting for {name} connection, '{pub.name}'")
                    return False
                rospy.sleep(0.1)
        return True
    def pub_control_robot_head(self, yaw:float, pitch:float)->bool:
        try :
            msg = robotHeadMotionData()
            msg.joint_data = [yaw, pitch]
            self._pub_ctrl_robot_head.publish(msg)
            return True
        except Exception as e:
            SDKLogger.error(f"[Error] publish robot head: {e}")
            return False

""" Control Robot Motion """

# JoyButton constants
BUTTON_A = 0
BUTTON_B = 1
BUTTON_X = 2
BUTTON_Y = 3
BUTTON_LB = 4
BUTTON_RB = 5
BUTTON_BACK = 6
BUTTON_START = 7

# JoyAxis constants
AXIS_LEFT_STICK_Y = 0
AXIS_LEFT_STICK_X = 1
AXIS_LEFT_LT = 2  # 1 -> (-1)
AXIS_RIGHT_STICK_YAW = 3
AXIS_RIGHT_STICK_Z = 4
AXIS_RIGHT_RT = 5  # 1 -> (-1)
AXIS_LEFT_RIGHT_TRIGGER = 6
AXIS_FORWARD_BACK_TRIGGER = 7

class ControlRobotMotion:
    def __init__(self):
        self._pub_cmd_vel = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self._pub_cmd_pose = rospy.Publisher('/cmd_pose', Twist, queue_size=10)
        self._pub_joy = rospy.Publisher('/joy', Joy, queue_size=10)
        self._pub_switch_gait = rospy.Publisher('/humanoid_switch_gait_by_name', switchGaitByName, queue_size=10)
        self._pub_step_ctrl = rospy.Publisher('/humanoid_mpc_foot_pose_target_trajectories', footPoseTargetTrajectories, queue_size=10)

    def connect(self, timeout:float=1.0)-> bool:
        start_time = rospy.Time.now()
        publishers = [
            # (self._pub_joy, "joy publisher"),
            (self._pub_cmd_vel, "cmd_vel publisher"),
            (self._pub_cmd_pose, "cmd_pose publisher"),
            (self._pub_step_ctrl, "step_ctrl publisher"),
            (self._pub_switch_gait, "switch_gait publisher")
        ]
        
        success = True
        for pub, name in publishers:
            while pub.get_num_connections() == 0:
                if (rospy.Time.now() - start_time).to_sec() > timeout:
                    SDKLogger.error(f"Timeout waiting for {name} connection, '{pub.name}'")
                    success = False
                    break
                rospy.sleep(0.1)
        return success

    def pub_cmd_vel(self, linear_x: float, linear_y: float, angular_z: float)->bool:
        try:
            twist = Twist()
            twist.linear.x = linear_x
            twist.linear.y = linear_y
            twist.angular.z = angular_z
            self._pub_cmd_vel.publish(twist)
            return True
        except Exception as e:
            SDKLogger.error(f"[Error] publish cmd vel: {e}")
            return False

    def pub_cmd_pose(self, twist:Twist)->bool:
        try:
            self._pub_cmd_pose.publish(twist)
            return True
        except Exception as e:
            SDKLogger.error(f"[Error] publish cmd pose: {e}")
            return False    
    
    def _create_joy_msg(self)->Joy:
        joy_msg = Joy()
        joy_msg.axes = [0.0] * 8  # Initialize 8 axes
        joy_msg.buttons = [0] * 16  # Initialize 16 buttons
        return joy_msg

    def _pub_joy_command(self, button_index: int, command_name: str) -> bool:
        try:
            joy_msg = self._create_joy_msg()
            joy_msg.buttons[button_index] = 1
            self._pub_joy.publish(joy_msg)
            SDKLogger.debug(f"Published {command_name} command")
            return True
        except Exception as e:
            SDKLogger.error(f"[Error] publish {command_name}: {e}")
            return False

    def _pub_switch_gait_by_name(self, gait_name: str) -> bool:
        try:
            msg = switchGaitByName()
            msg.header.stamp = rospy.Time.now()
            msg.gait_name = gait_name
            self._pub_switch_gait.publish(msg)
            # print(f"Published {gait_name} gait")
            return True
        except Exception as e:
            SDKLogger.error(f"[Error] publish switch gait {gait_name}: {e}")
            return False

    def pub_walk_command(self) -> bool:
        # return self._pub_joy_command(BUTTON_Y, "walk")
        return self._pub_switch_gait_by_name("walk")

    def pub_stance_command(self) -> bool:
        try:
            self.pub_cmd_vel(linear_x=0.0, linear_y=0.0, angular_z=0.0)
            # return self._pub_joy_command(BUTTON_A, "stance") 
            return self._pub_switch_gait_by_name("stance")
        except Exception as e:
            SDKLogger.error(f"[Error] publish stance: {e}")
            return False

    def pub_trot_command(self) -> bool:
        # return self._pub_joy_command(BUTTON_B, "trot")
        return self._pub_switch_gait_by_name("walk")
    
    def pub_step_ctrl(self, msg:footPoseTargetTrajectories)->bool:
        try:
            self._pub_step_ctrl.publish(msg)
            return True
        except Exception as e:
            SDKLogger.error(f"[Error] publish step ctrl: {e}")
            return False
        
"""
Kuavo Robot Arm IK&FK
"""
class KuavoRobotArmIKFK:
    def __init__(self):
        pass
    def arm_ik(self, 
               left_pose: KuavoPose, 
               right_pose: KuavoPose, 
               left_elbow_pos_xyz: list = [0.0, 0.0, 0.0],
               right_elbow_pos_xyz: list = [0.0, 0.0, 0.0],
               arm_q0: list = None,
               params: KuavoIKParams=None) -> list:
        eef_pose_msg = twoArmHandPoseCmd()
        if arm_q0 is None:
            eef_pose_msg.joint_angles_as_q0 = False
        else:
            eef_pose_msg.joint_angles_as_q0 = True
            eef_pose_msg.joint_angles = arm_q0
        
        if params is None:
            eef_pose_msg.use_custom_ik_param = False
        else:
            eef_pose_msg.use_custom_ik_param = True
            eef_pose_msg.ik_param.major_optimality_tol = params.major_optimality_tol
            eef_pose_msg.ik_param.major_feasibility_tol = params.major_feasibility_tol
            eef_pose_msg.ik_param.minor_feasibility_tol = params.minor_feasibility_tol
            eef_pose_msg.ik_param.major_iterations_limit = params.major_iterations_limit
            eef_pose_msg.ik_param.oritation_constraint_tol= params.oritation_constraint_tol
            eef_pose_msg.ik_param.pos_constraint_tol = params.pos_constraint_tol 
            eef_pose_msg.ik_param.pos_cost_weight = params.pos_cost_weight

        # left hand
        eef_pose_msg.hand_poses.left_pose.pos_xyz =  left_pose.position
        eef_pose_msg.hand_poses.left_pose.quat_xyzw = left_pose.orientation
        eef_pose_msg.hand_poses.left_pose.elbow_pos_xyz = left_elbow_pos_xyz

        # right hand
        eef_pose_msg.hand_poses.right_pose.pos_xyz =  right_pose.position
        eef_pose_msg.hand_poses.right_pose.quat_xyzw = right_pose.orientation
        eef_pose_msg.hand_poses.right_pose.elbow_pos_xyz = right_elbow_pos_xyz

        return self._srv_arm_ik(eef_pose_msg)   

    def arm_fk(self, q: list) -> Tuple[KuavoPose, KuavoPose]:
        return self._srv_arm_fk(q)
    
    def _srv_arm_ik(self, eef_pose_msg:twoArmHandPoseCmd)->list:
        try:
            rospy.wait_for_service('/ik/two_arm_hand_pose_cmd_srv',timeout=1.0)
            ik_srv = rospy.ServiceProxy('/ik/two_arm_hand_pose_cmd_srv', twoArmHandPoseCmdSrv)
            res = ik_srv(eef_pose_msg)
            # print(eef_pose_msg)
            if res.success:
                return res.hand_poses.left_pose.joint_angles + res.hand_poses.right_pose.joint_angles
            else:
                return None
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)
            return None
        except Exception as e:
            print(f"Failed to call ik/fk_srv: {e}")
            return None
    def _srv_arm_fk(self, q: list) -> Tuple[KuavoPose, KuavoPose]:
        try:
            rospy.wait_for_service('/ik/fk_srv',timeout=1.0)
            ik_srv = rospy.ServiceProxy('/ik/fk_srv', fkSrv)
            res = ik_srv(q)
            if res.success:
                return KuavoPose(position=res.hand_poses.left_pose.pos_xyz, orientation=res.hand_poses.left_pose.quat_xyzw), \
                    KuavoPose(position=res.hand_poses.right_pose.pos_xyz, orientation=res.hand_poses.right_pose.quat_xyzw),
            else:
                return None
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)
            return None
        except Exception as e:
            print(f"Failed to call ik/fk_srv: {e}")
            return None
"""
    Kuavo Robot Control 
"""
class KuavoRobotControl:
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            if not rospy.core.is_initialized():
                if not rospy.get_node_uri():
                    rospy.init_node(f'kuavo_sdk_node_{os.getpid()}', anonymous=True, disable_signals=True)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self.kuavo_eef_control = None
            self.kuavo_head_control = ControlRobotHead()
            self.kuavo_arm_control = ControlRobotArm()
            self.kuavo_motion_control = ControlRobotMotion()
            self.kuavo_arm_ik_fk = KuavoRobotArmIKFK()
            # SDKLogger.debug("KuavoRobotControl initialized.")

    def initialize(self, eef_type:str=None, debug:bool=False, timeout:float=1.0)-> Tuple[bool, str]:
        # init eef control
        if eef_type is None:
            self.kuavo_eef_control = None
        else:
            self.kuavo_eef_control = ControlEndEffector(eef_type=eef_type)
        
        connect_success = True
        err_msg = ''
        if not self.kuavo_arm_control.connect(timeout):
            connect_success  = False
            err_msg = "Failed to connect to arm control topics, \n"
        if not self.kuavo_head_control.connect(timeout):
            connect_success  = False
            err_msg += "Failed to connect to head control topics, \n"
        if not self.kuavo_motion_control.connect(timeout):
            err_msg += "Failed to connect to motion control topics, \n"
            connect_success  = False

        if self.kuavo_eef_control is not None and not self.kuavo_eef_control.connect(timeout):
            connect_success  = False
            err_msg += "Failed to connect to end effector control topics."

        if connect_success:
            err_msg = 'success'
        return connect_success, err_msg
    
    """ End Effector Control"""
    def control_robot_dexhand(self, left_position:list, right_position:list)->bool:
        """
            Control robot dexhand
            Args:
                left_position: list of 6 floats between 0 and 100
                right_position: list of 6 floats between 0 and 100
        """
        if self.kuavo_eef_control is None:
            SDKLogger.error("End effector control is not initialized.")
            return False
        
        if len(left_position) != 6 or len(right_position) != 6:
            raise ValueError("Position lists must have a length of 6.")
        
        for position in left_position + right_position:
            if position < 0.0 or position > 100.0:
                raise ValueError("All position values must be in the range [0.0, 100.0].")    
        
        SDKLogger.debug(f"Control robot dexhand: {left_position}, {right_position}")
        return self.kuavo_eef_control.pub_control_robot_dexhand(left_position, right_position)

    def execute_gesture(self, gestures:list)->bool:
        """
            Execute gestures
            Arguments:
                - gestures: list of dicts with keys 'gesture_name' and 'hand_side'
                 e.g. [{'gesture_name': 'fist', 'hand_side': 0},]
        """
        if self.kuavo_eef_control is None:
            SDKLogger.warn("End effectors control is not initialized.")
            return False
        return self.kuavo_eef_control.srv_execute_gesture(gestures)

    def get_gesture_names(self)->list:
        """
            Get the names of all gestures.
        """
        if self.kuavo_eef_control is None:
            SDKLogger.warn("End effectors control is not initialized.")
            return []
        return self.kuavo_eef_control.srv_get_gesture_names()

    def control_leju_claw(self, postions:list, velocities:list=[90, 90], torques:list=[1.0, 1.0]) ->bool:
        """
            Control leju claw
            Arguments:
                - postions: list of positions for left and right claw
                - velocities: list of velocities for left and right claw
                - torques: list of torques for left and right claw
        """
        if self.kuavo_eef_control is None:
            SDKLogger.warn("End effectors control is not initialized.")
            return False
        SDKLogger.debug(f"Control leju claw: {postions}, {velocities}, {torques}")
        if len(postions) != 2 or len(velocities) != 2 or len(torques) != 2:
                raise ValueError("Position, velocity, and torque lists must have a length of 2.")
        return self.kuavo_eef_control.srv_control_leju_claw(postions, velocities, torques)
    """--------------------------------------------------------------------------------------------"""
    def control_robot_head(self, yaw:float, pitch:float)->bool:
        """
            Control robot head
            Arguments:
                - yaw: yaw angle, radian
                - pitch: pitch angle, radian
        """
        SDKLogger.debug(f"Control robot head: {yaw}, {pitch}")
        return self.kuavo_head_control.pub_control_robot_head(yaw, pitch)
    

    def control_robot_arm_traj(self, joint_data:list)->bool:
        """
            Control robot arm trajectory
            Arguments:
                - joint_data: list of joint data (degrees)
        """
        # SDKLogger.debug(f"[ROS] Control robot arm trajectory: {joint_data}")
        return self.kuavo_arm_control.pub_control_robot_arm_traj(joint_data)
    
    def control_robot_arm_target_poses(self, times:list, joint_q:list)->bool:
        """
            Control robot arm target poses
            Arguments:
                - times: list of times (seconds)
                - joint_q: list of joint data (degrees)
        """
        if len(times) != len(joint_q):
            raise ValueError("Times and joint_q must have the same length.")
        elif len(times) == 0:
            raise ValueError("Times and joint_q must not be empty.")
        
        return self.kuavo_arm_control.pub_arm_target_poses(times=times, joint_q=joint_q)
    
    def change_robot_arm_ctrl_mode(self, mode:KuavoArmCtrlMode)->bool:
        """
            Change robot arm control mode
            Arguments:
                - mode: arm control mode
        """
        SDKLogger.debug(f"[ROS] Change robot arm control mode: {mode}")
        return self.kuavo_arm_control.srv_change_arm_ctrl_mode(mode)
    
    def get_robot_arm_ctrl_mode(self)->int:
        """
            Get robot arm control mode
        """
        return self.kuavo_arm_control.srv_get_arm_ctrl_mode()
    
    def arm_ik(self, left_pose: KuavoPose, right_pose: KuavoPose, 
               left_elbow_pos_xyz: list = [0.0, 0.0, 0.0],  
               right_elbow_pos_xyz: list = [0.0, 0.0, 0.0],
               arm_q0: list = None, params: KuavoIKParams=None) -> list:
        return self.kuavo_arm_ik_fk.arm_ik(left_pose, right_pose, left_elbow_pos_xyz, right_elbow_pos_xyz, arm_q0, params)


    def arm_fk(self, q: list) -> Tuple[KuavoPose, KuavoPose]:
        return self.kuavo_arm_ik_fk.arm_fk(q)
    
    """ Motion """
    def robot_stance(self)->bool:
        return self.kuavo_motion_control.pub_stance_command()

    def robot_trot(self)->bool:
        return self.kuavo_motion_control.pub_trot_command()
    
    def robot_walk(self, linear_x:float, linear_y:float, angular_z:float)->bool:
        return self.kuavo_motion_control.pub_cmd_vel(linear_x, linear_y, angular_z)
    
    def control_torso_height(self, height:float, pitch:float=0.0)->bool:
        com_msg = Twist()
        com_msg.linear.z = height
        com_msg.angular.y = pitch
        return self.kuavo_motion_control.pub_cmd_pose(com_msg)

    def step_control(self, body_poses:list, dt:float, is_left_first_default:bool=True, collision_check:bool=True)->bool:
        """
            Step control
            Arguments:
                - body_poses: list of body poses (x, y, z, yaw), meters and degrees
                - dt: time step (seconds)
                - is_left_first_default: whether to start with left foot
                - collision_check: whether to check for collisions
        """
        if len(body_poses) == 0:
            raise ValueError("Body poses must not be empty.")
        if dt <= 0.0:
            raise ValueError("Time step must be greater than 0.0.")
        for bp in body_poses:
            if len(bp) != 4:
                raise ValueError("Body pose must have 4 elements: [x, y, z, yaw]")
        
        msg = get_multiple_steps_msg(body_poses, dt, is_left_first_default, collision_check)
        return self.kuavo_motion_control.pub_step_ctrl(msg)
""" ------------------------------------------------------------------------------"""    
def euler_to_rotation_matrix(yaw, pitch, roll):
    # 计算各轴的旋转矩阵
    R_yaw = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                      [np.sin(yaw), np.cos(yaw), 0],
                      [0, 0, 1]])

    R_pitch = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                        [0, 1, 0],
                        [-np.sin(pitch), 0, np.cos(pitch)]])

    R_roll = np.array([[1, 0, 0],
                       [0, np.cos(roll), -np.sin(roll)],
                       [0, np.sin(roll), np.cos(roll)]])

    # 按照 Yaw-Pitch-Roll 的顺序组合旋转矩阵
    R = np.dot(R_roll, np.dot(R_pitch, R_yaw))
    return R

def get_foot_pose_traj_msg(time_traj:list, foot_idx_traj:list, foot_traj:list, torso_traj:list):
    num = len(time_traj)

    # 创建消息实例
    msg = footPoseTargetTrajectories()
    msg.timeTrajectory = time_traj  # 设置时间轨迹
    msg.footIndexTrajectory = foot_idx_traj         # 设置脚索引
    msg.footPoseTrajectory = []  # 初始化脚姿态轨迹

    for i in range(num):
        foot_pose_msg = footPose()
        foot_pose_msg.footPose = foot_traj[i]      # 设置脚姿态
        foot_pose_msg.torsoPose = torso_traj[i]    # 设置躯干姿态
        msg.footPoseTrajectory.append(foot_pose_msg)

    return msg

def generate_steps(torso_pos, torso_yaw, foot_bias):
    l_foot_bias = np.array([0, foot_bias, -torso_pos[2]])
    r_foot_bias = np.array([0, -foot_bias, -torso_pos[2]])
    R_z = np.array([
        [np.cos(torso_yaw), -np.sin(torso_yaw), 0],
        [np.sin(torso_yaw), np.cos(torso_yaw), 0],
        [0, 0, 1]
    ])
    l_foot = torso_pos + R_z.dot(l_foot_bias)
    r_foot = torso_pos + R_z.dot(r_foot_bias)
    return l_foot, r_foot   

def get_multiple_steps_msg(body_poses:list, dt:float, is_left_first:bool=True, collision_check:bool=True):
    num_steps = 2*len(body_poses)
    time_traj = []
    foot_idx_traj = []
    foot_traj = []
    torso_traj = []
    l_foot_rect_last = RotatingRectangle(center=(0, 0.1), width=0.24, height=0.1, angle=0)
    r_foot_rect_last = RotatingRectangle(center=(0,-0.1), width=0.24, height=0.1, angle=0)
    torso_pose_last = np.array([0, 0, 0, 0])
    for i in range(num_steps):
        time_traj.append(dt * (i+1))
        body_pose = body_poses[i//2]
        torso_pos = np.asarray(body_pose[:3])
        torso_yaw = np.radians(body_pose[3])
        l_foot, r_foot = generate_steps(torso_pos, torso_yaw, 0.1)
        l_foot = [*l_foot[:3], torso_yaw]
        r_foot = [*r_foot[:3], torso_yaw]

        if(i%2 == 0):        
            torso_pose = np.array([*body_pose[:3], torso_yaw])
            R_wl = euler_to_rotation_matrix(torso_pose_last[3], 0, 0)
            delta_pos = R_wl.T @ (torso_pose[:3] - torso_pose_last[:3])
            # print("delta_pos:", delta_pos)
            if(torso_yaw > 0.0 or delta_pos[1] > 0.0):
                is_left_first = True
            else:
                is_left_first = False

        if(collision_check and i%2 == 0):
            l_foot_rect_next = RotatingRectangle(center=(l_foot[0],l_foot[1]), width=0.24, height=0.1, angle=torso_yaw)
            r_foot_rect_next = RotatingRectangle(center=(r_foot[0],r_foot[1]), width=0.24, height=0.1, angle=torso_yaw)
            l_collision = l_foot_rect_next.is_collision(r_foot_rect_last)
            r_collision = r_foot_rect_next.is_collision(l_foot_rect_last)
            if l_collision and r_collision:
                SDKLogger.error("[Control] Detect collision, Please adjust your body_poses input!!!")
                break
            elif l_collision:
                SDKLogger.warn("[Control] Left foot is in collision, switch to right foot")
                is_left_first = False
            elif r_collision:
                SDKLogger.warn("[Control] Right foot is in collision, switch to left foot")
                is_left_first = True
            l_foot_rect_last = l_foot_rect_next
            r_foot_rect_last = r_foot_rect_next
        if(i%2 == 0):
            torso_traj.append((torso_pose_last + torso_pose)/2.0)
            if is_left_first:
                foot_idx_traj.append(0)
                foot_traj.append(l_foot)
            else:
                foot_idx_traj.append(1)
                foot_traj.append(r_foot)
        else:
            torso_traj.append(torso_pose)
            if is_left_first:
                foot_idx_traj.append(1)
                foot_traj.append(r_foot)
            else:
                foot_idx_traj.append(0)
                foot_traj.append(l_foot)
        torso_pose_last = torso_traj[-1]
    # print("time_traj:", time_traj)
    # print("foot_idx_traj:", foot_idx_traj)
    # print("foot_traj:", foot_traj)
    # print("torso_traj:", torso_traj)
    return get_foot_pose_traj_msg(time_traj, foot_idx_traj, foot_traj, torso_traj)
""" ------------------------------------------------------------------------------"""