import rospy
import copy
import time
from typing import Tuple
from std_msgs.msg import Float64
from nav_msgs.msg import Odometry
from kuavo_msgs.msg import sensorsData, lejuClawState, gaitTimeName

from kuavo_msgs.srv import changeArmCtrlMode, changeArmCtrlModeRequest, getCurrentGaitName, getCurrentGaitNameRequest
from ocs2_msgs.msg import mpc_observation
from kuavo_humanoid_sdk.interfaces.data_types import KuavoImuData, KuavoJointData, KuavoOdometry, KuavoArmCtrlMode, EndEffectorState
from kuavo_humanoid_sdk.kuavo.core.ros.param import make_robot_param
from kuavo_humanoid_sdk.common.logger import SDKLogger

from collections import deque
from typing import Tuple, Optional

class GaitManager:
    def __init__(self, max_size: int = 20):
        self._max_size = max_size
        self._gait_time_names = deque(maxlen=max_size)

    @property
    def is_empty(self) -> bool:
        """Check if there are any gait time names stored.
        
        Returns:
            bool: True if no gait time names are stored, False otherwise
        """
        return len(self._gait_time_names) == 0
    def add(self, start_time: float, name: str) -> None:
        self._gait_time_names.append((start_time, name))

    def get_gait(self, current_time: float) -> str:
        if not self._gait_time_names:
            return "No_Gait"

        for start_time, name in reversed(self._gait_time_names):
            if current_time >= start_time:
                return name

        return self._gait_time_names[0].name

    def get_gait_name(self, current_time: float) -> str:
        return self.get_gait(current_time)[1]

    def get_last_gait_name(self) -> str:
        if not self._gait_time_names:
            return "No_Gait"
        return self._gait_time_names[-1][1]

class KuavoRobotStateCore:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, '_initialized'):
            rospy.Subscriber("/sensors_data_raw", sensorsData, self._sensors_data_raw_callback)
            rospy.Subscriber("/odom", Odometry, self._odom_callback)
            rospy.Subscriber("/humanoid/mpc/terrainHeight", Float64, self._terrain_height_callback)
            rospy.Subscriber("/humanoid_mpc_gait_time_name", gaitTimeName, self._humanoid_mpc_gait_changed_callback)
            rospy.Subscriber("/humanoid_mpc_observation", mpc_observation, self._humanoid_mpc_observation_callback)
            
            kuavo_info = make_robot_param()
            if kuavo_info['end_effector_type'] == "lejuclaw":
                rospy.Subscriber("/leju_claw_state", lejuClawState, self._lejuclaw_state_callback)
            elif kuavo_info['end_effector_type'] == "qiangnao":
                pass # TODO(kuavo): add qiangnao state subscriber
                # rospy.Subscriber("/robot_hand_state", robotHandState, self._dexterous_hand_state_callback)
            
            """ data """
            self._terrain_height = 0.0
            self._imu_data = KuavoImuData(
                gyro = (0.0, 0.0, 0.0),
                acc = (0.0, 0.0, 0.0),
                free_acc = (0.0, 0.0, 0.0),
                quat = (0.0, 0.0, 0.0, 0.0)
            )
            self._joint_data = KuavoJointData(
                position = [0.0] * 28,
                velocity = [0.0] * 28,
                torque = [0.0] * 28,
                acceleration = [0.0] * 28
            )
            self._odom_data = KuavoOdometry(
                position = (0.0, 0.0, 0.0),
                orientation = (0.0, 0.0, 0.0, 0.0),
                linear = (0.0, 0.0, 0.0),
                angular = (0.0, 0.0, 0.0)
            )
            self._eef_state = (EndEffectorState(
                position = 0.0,
                velocity = 0.0,
                effort = 0.0,
                state=EndEffectorState.GraspingState.UNKNOWN
            ), EndEffectorState(
                position = 0.0,
                velocity = 0.0,
                effort = 0.0,
                state=EndEffectorState.GraspingState.UNKNOWN
            ))

            # gait manager
            self._gait_manager = GaitManager()
            self._prev_gait_name = self.gait_name

            # Wait for first MPC observation data
            self._mpc_observation_data = None
            start_time = time.time()
            while self._mpc_observation_data is None:
                if time.time() - start_time > 1.0:  # 1.0s timeout
                    SDKLogger.warn("Timeout waiting for MPC observation data")
                    break
                SDKLogger.debug("Waiting for first MPC observation data...")
                time.sleep(0.1)
            # 如果 gait_manager 为空，则把当前的状态添加到其中
            if self._mpc_observation_data is not None:
                if self._gait_manager.is_empty:
                    self._prev_gait_name = self.gait_name()
                    SDKLogger.debug(f"[State] Adding initial gait state: {self._prev_gait_name} at time {self._mpc_observation_data.time}")
                    self._gait_manager.add(self._mpc_observation_data.time, self._prev_gait_name)

            # 获取当前手臂控制模式
            self._arm_ctrl_mode = self._srv_get_arm_ctrl_mode()
            self._initialized = True

    @property
    def com_height(self)->float:
        # odom.position.z - terrain_height = com_height
        return self._odom_data.position[2] - self._terrain_height
    
    @property
    def imu_data(self)->KuavoImuData:
        return self._imu_data
    
    @property
    def joint_data(self)->KuavoJointData:
        return self._joint_data
    
    @property
    def odom_data(self)->KuavoOdometry:
        return self._odom_data

    @property
    def arm_control_mode(self) -> KuavoArmCtrlMode:
        mode = self._srv_get_arm_ctrl_mode()
        if mode is not None:
            self._arm_ctrl_mode = mode
        return self._arm_ctrl_mode
    
    @property
    def eef_state(self)->Tuple[EndEffectorState, EndEffectorState]:
        return self._eef_state
    
    @property
    def current_observation_time(self) -> float:
        """Get the current observation time from MPC.
        
        Returns:
            float: Current observation time in seconds, or None if no observation data available
        """
        if self._mpc_observation_data is None:
            return None
        return self._mpc_observation_data.time
    
    @property
    def current_gait_mode(self) -> int:
        """
        Get the current gait mode from MPC observation.
        
        Notes: FF=0, FH=1, FT=2, FS=3, HF=4, HH=5, HT=6, HS=7, TF=8, TH=9, TT=10, TS=11, SF=12, SH=13, ST=14, SS=15
        
        Returns:
            int: Current gait mode, or None if no observation data available
        """
        if self._mpc_observation_data is None:
            return None
        return self._mpc_observation_data.mode

    def gait_name(self)->str:
        return self._srv_get_current_gait_name()
    
    def is_gait(self, gait_name: str) -> bool:
        """Check if current gait matches the given gait name.
        
        Args:
            gait_name (str): Name of gait to check
            
        Returns:
            bool: True if current gait matches given name, False otherwise
        """
        return self._gait_manager.get_gait(self._mpc_observation_data.time) == gait_name

    def register_gait_changed_callback(self, callback):
        """Register a callback function that will be called when the gait changes.
        
        Args:
            callback: A function that takes current time (float) and gait name (str) as arguments
        Raises:
            ValueError: If callback does not accept 2 parameters: current_time (float), gait_name (str)
        """
        if not hasattr(self, '_gait_changed_callbacks'):
            self._gait_changed_callbacks = []
        
        # Verify callback accepts current_time (float) and gait_name (str) parameters
        from inspect import signature
        sig = signature(callback)
        if len(sig.parameters) != 2:
            raise ValueError("Callback must accept 2 parameters: current_time (float), gait_name (str)")
        if callback not in self._gait_changed_callbacks:
            self._gait_changed_callbacks.append(callback)
        
    """ ------------------------------- callback ------------------------------- """
    def _terrain_height_callback(self, msg:Float64)->None:
        self._terrain_height = msg.data

    def _sensors_data_raw_callback(self, msg:sensorsData)->None:
        # update imu data
        self._imu_data = KuavoImuData(
            gyro = (msg.imu_data.gyro.x, msg.imu_data.gyro.y, msg.imu_data.gyro.z),
            acc = (msg.imu_data.acc.x, msg.imu_data.acc.y, msg.imu_data.acc.z),
            free_acc = (msg.imu_data.free_acc.x, msg.imu_data.free_acc.y, msg.imu_data.free_acc.z),
            quat = (msg.imu_data.quat.x, msg.imu_data.quat.y, msg.imu_data.quat.z, msg.imu_data.quat.w)
        )
        # update joint data
        self._joint_data = KuavoJointData(
            position = copy.deepcopy(msg.joint_data.joint_q),
            velocity = copy.deepcopy(msg.joint_data.joint_v),
            torque = copy.deepcopy(msg.joint_data.joint_vd),
            acceleration = copy.deepcopy(msg.joint_data.joint_current)
        )

    def _odom_callback(self, msg:Odometry)->None:
        # update odom data
        self._odom_data = KuavoOdometry(
            position = (msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z),
            orientation = (msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w),
            linear = (msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z),
            angular = (msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z)
        )

    def _lejuclaw_state_callback(self, msg:lejuClawState)->None:
        self._eef_state = (EndEffectorState(
            position = msg.data.position[0],
            velocity = msg.data.velocity[0],
            effort = msg.data.effort[0],
            state=EndEffectorState.GraspingState(msg.state[0])
        ), EndEffectorState(
            position = msg.data.position[1],
            velocity = msg.data.velocity[1],
            effort = msg.data.effort[1],
            state=EndEffectorState.GraspingState(msg.state[1])
        ))

    def _dexterous_hand_state_callback(self, msg)->None:
        self._eef_state = (EndEffectorState(
            position = msg.data.position[0],
            velocity = msg.data.velocity[0],
            effort = msg.data.effort[0],
            state=EndEffectorState.GraspingState(msg.state[0])
        ), EndEffectorState(
            position = msg.data.position[1],
            velocity = msg.data.velocity[1],
            effort = msg.data.effort[1],
            state=EndEffectorState.GraspingState(msg.state[1])
        ))
    def _humanoid_mpc_gait_changed_callback(self, msg: gaitTimeName):
        """
        Callback function for gait change messages.
        Updates the current gait name when a gait change occurs.
        """
        SDKLogger.debug(f"[State] Received gait change message: {msg.gait_name} at time {msg.start_time}")
        self._gait_manager.add(msg.start_time, msg.gait_name)
    
    def _humanoid_mpc_observation_callback(self, msg: mpc_observation) -> None:
        """
        Callback function for MPC observation messages.
        Updates the current MPC state and input data.
        """
        try:
            self._mpc_observation_data = msg
            # Only update gait info after initialization
            if hasattr(self, '_initialized'): 
                curr_time = self._mpc_observation_data.time
                current_gait = self._gait_manager.get_gait(curr_time)
                if self._prev_gait_name != current_gait:
                    SDKLogger.debug(f"[State] Gait changed to: {current_gait} at time {curr_time}")
                    # Update previous gait name and notify callback if registered
                    self._prev_gait_name = current_gait
                    if hasattr(self, '_gait_changed_callbacks') and self._gait_changed_callbacks is not None:
                        for callback in self._gait_changed_callbacks:
                            callback(curr_time, current_gait)
        except Exception as e:
            SDKLogger.error(f"Error processing MPC observation: {e}")

    def _srv_get_arm_ctrl_mode(self)-> KuavoArmCtrlMode:
        try:
            rospy.wait_for_service('/humanoid_get_arm_ctrl_mode')
            get_arm_ctrl_mode_srv = rospy.ServiceProxy('/humanoid_get_arm_ctrl_mode', changeArmCtrlMode)
            req = changeArmCtrlModeRequest()
            resp = get_arm_ctrl_mode_srv(req)
            return KuavoArmCtrlMode(resp.mode)
        except rospy.ServiceException as e:
            SDKLogger.error(f"Service call failed: {e}")
        except Exception as e:
            SDKLogger.error(f"[Error] get arm ctrl mode: {e}")
        return None
    
    def _srv_get_current_gait_name(self)->str:
        try:
            rospy.wait_for_service('/humanoid_get_current_gait_name', timeout=1.0)
            srv = rospy.ServiceProxy('/humanoid_get_current_gait_name', getCurrentGaitName)
            request = getCurrentGaitNameRequest()
            response = srv(request)
            if response.success:
                return response.gait_name
            else:
                return None
        except Exception as e:
            SDKLogger.error(f"Service call failed: {e}")
        return None