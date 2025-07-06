import rospy
from sensor_msgs.msg import JointState
import math

all_joint_names = [
    "l_leg_roll",
    "l_leg_yaw",
    "l_leg_pitch",
    "l_knee_pitch",
    "l_foot_pitch",
    "l_foot_roll",
    "l_l_bar_y",
    "l_l_tendon_y",
    "l_r_bar_y",
    "l_r_tendon_y",
    "r_leg_roll",
    "r_leg_yaw",
    "r_leg_pitch",
    "r_knee_pitch",
    "r_foot_pitch",
    "r_foot_roll",
    "r_r_bar_y",
    "r_r_tendon_y",
    "r_l_bar_y",
    "r_l_tendon_y",
    "l_arm_pitch",
    "l_arm_roll",
    "l_arm_yaw",
    "l_forearm_pitch",
    "l_forearm_yaw",
    "l_hand_roll",
    "l_hand_pitch",
    "l_l_arm_bar",
    "l_l_arm_tendon",
    "l_r_arm_bar",
    "l_r_arm_tendon",
    "r_arm_pitch",
    "r_arm_roll",
    "r_arm_yaw",
    "r_forearm_pitch",
    "r_forearm_yaw",
    "r_hand_roll",
    "r_hand_pitch",
    "r_r_arm_bar",
    "r_r_arm_tendon",
    "r_l_arm_bar",
    "r_l_arm_tendon",
]
left_joint_index = [20, 21, 22, 23, 24, 26, 25]
right_joint_index = [31, 32, 33, 34, 35, 37, 36]


def kuavo_traj_callback(msg):
    """
    Callback function for the joint_states topic.

    Args:
    - msg: Message from the joint_states topic (sensor_msgs.msg.JointState).
    """
    pub = rospy.Publisher("/joint_states", JointState, queue_size=10)
    # Print the joint names and positions
    joint_state_pos = [math.radians(pos) for pos in msg.position]
    new_msg = JointState()
    new_msg.header.frame_id = "torso"
    new_msg.header.stamp = rospy.Time.now()
    new_msg.name = all_joint_names
    new_msg.position = [0] * len(all_joint_names)
    new_msg.position[20:27] = joint_state_pos[:7]
    new_msg.position[31:38] = joint_state_pos[7:]
    print(new_msg)
    pub.publish(new_msg)


def pub_init_kuavo_joint_state():
    """
    Initialize the joint state of the Kuavo robot.
    """
    pub = rospy.Publisher("/joint_states", JointState, queue_size=10)
    new_msg = JointState()
    new_msg.header.frame_id = "torso"
    new_msg.header.stamp = rospy.Time.now()
    new_msg.name = all_joint_names
    new_msg.position = [0] * len(all_joint_names)
    pub.publish(new_msg)


def run_node():
    """
    Runs the joint_state node.
    """
    # Initialize the node
    rospy.init_node("robot_joint_state_node")

    # Subscribe to the joint_states topic
    rospy.Subscriber("/kuavo_arm_traj", JointState, kuavo_traj_callback)

    rate = rospy.Rate(10)  # 10hz

    pub_times = 5
    while not rospy.is_shutdown():
        if pub_times > 0:
            pub_init_kuavo_joint_state()
            pub_times -= 1
        rate.sleep()


if __name__ == "__main__":
    run_node()
