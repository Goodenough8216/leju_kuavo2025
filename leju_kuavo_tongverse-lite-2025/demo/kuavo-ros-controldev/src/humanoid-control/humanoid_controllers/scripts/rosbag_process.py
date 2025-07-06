#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os
import rosbag
import rospy
import ocs2_msgs.msg
from std_msgs.msg import Float32MultiArray


def get_latest_bag_file(folder_path):
    folder_path = os.path.expanduser(folder_path)
    # 获取文件夹下所有以.bag和.bag.active结尾的文件
    bag_files = [f for f in os.listdir(folder_path) if (f.endswith(".bag") or f.endswith(".bag.active"))and "expanded" not in f]
    bag_files.sort(key=lambda f: os.path.getmtime(os.path.join(folder_path, f)), reverse=True)
    
    if bag_files:
        return os.path.join(folder_path, bag_files[0])
    else:
        return None
    
class BagProcessor:
    def __init__(self, input_bag_path, output_bag_path=None):
        self.input_bag_path = input_bag_path
        self.output_bag_path = output_bag_path if output_bag_path else self._generate_default_output_path()
        print(f"Input bag path: {self.input_bag_path}\noutput bag path: {self.output_bag_path}")
        if not os.path.exists(input_bag_path):
            print(f"Input bag file not found: {input_bag_path}")
            exit()
        self.check_and_reindex_bag()
        self.messages = []
        self.policy_initial_ros_time = None
        self.num_policy_publishes = 100
        self.current_policy_publish_index = 0
        self.current_feet_state_publish_index = 0
    
    def check_and_reindex_bag(self):
        new_name, ext = os.path.splitext(self.input_bag_path)
        if ext == '.active':
            print("Input bag is not indexed, reindexing...")
            os.system(f"rosbag reindex {self.input_bag_path}")
            print('reindexing done.')
            os.rename(self.input_bag_path, new_name)
            self.input_bag_path = new_name
            print(f"rename file to {new_name}")

    def _generate_default_output_path(self):
        base, ext = os.path.splitext(self.input_bag_path)
        return base + '_expanded.bag'
    
    def process_mpc_policy(self, msg):
        new_topic = '/policy_expansions/mpc_policy/mpc_policy_' + str(self.current_policy_publish_index)
        for i, policy_t in enumerate(msg.timeTrajectory):
            policy_data = ocs2_msgs.msg.mpc_observation()
            policy_data.time = policy_t
            policy_data.mode = msg.initObservation.mode
            policy_data.state = msg.stateTrajectory[i]
            policy_data.input = msg.inputTrajectory[i]
            new_msg_time = self.policy_initial_ros_time + rospy.Duration(policy_t)
            self.messages.append((new_topic, policy_data, new_msg_time))

        self.current_policy_publish_index = (self.current_policy_publish_index + 1) if self.current_policy_publish_index < self.num_policy_publishes else 0
    
    def process_feet_state(self, msg):
        new_topic = '/policy_expansions/feet_target_policys/feet_target_policys_' + str(self.current_feet_state_publish_index)
        for i, policy_t in enumerate(msg.timeTrajectory):
            policy_data = ocs2_msgs.msg.mpc_target_trajectories()
            # print(type(msg.stateTrajectory[i]))
            policy_data.stateTrajectory = [msg.stateTrajectory[i]]
            policy_data.timeTrajectory = [policy_t]
            new_msg_time = self.policy_initial_ros_time + rospy.Duration(policy_t)
            self.messages.append((new_topic, policy_data, new_msg_time))
        self.current_feet_state_publish_index = (self.current_feet_state_publish_index + 1) if self.current_feet_state_publish_index < self.num_policy_publishes else 0
    
    def process_bag(self):
        count = 0
        try:
            print("Reading messages from input bag...")
            with rosbag.Bag(self.input_bag_path, 'r') as input_bag:
                for topic, msg, t in input_bag.read_messages():
                    if topic == '/humanoid_mpc_observation':
                        self.policy_initial_ros_time = t - rospy.Duration(msg.time)
                    if self.policy_initial_ros_time is None:
                        continue
                    if topic == '/humanoid_mpc_policy':
                        self.process_mpc_policy(msg)
                    if topic == '/humanoid_controller/feet_target_policys': # feets state
                        self.process_feet_state(msg)
                    self.messages.append((topic, msg, t))
                    if count % 4321 == 0:
                        print(f"Processing messages: {count}", end='\r')
                    count += 1
        except Exception as e:
            rospy.logerr(f"Error processing bag file: {e}")

    def write_output_bag(self):
        print(f"Messages length: {len(self.messages)}, sorting messages by time ...")
        self.messages.sort(key=lambda x: x[2])
        print(f"Writing messages to output bag: {self.output_bag_path} ...")
        length = len(self.messages)
        try:
            with rosbag.Bag(self.output_bag_path, 'w') as output_bag:
                for idx, (topic, msg, t) in enumerate(self.messages):
                    output_bag.write(topic, msg, t)
                    if idx % 4321 == 0:
                        print(f"Writing message {100*idx/length:.2f}%", end='\r')
        except Exception as e:
            rospy.logerr(f"Error writing bag file: {e}")

        print("Done!"+" "*100)

def main():
    parser = argparse.ArgumentParser(description="Process ROS bag files.")
    parser.add_argument('-i', '--input_path', type=str, help='Path to the input bag file.')
    parser.add_argument('-o', '--output_path', type=str, help='Path to the output bag file.')

    args = parser.parse_args()
    if args.input_path is None:
        print("No input bag file specified, using latest bag file in ~/.ros")
        args.input_path = get_latest_bag_file("~/.ros")
        print(f"Found latest bag file: {args.input_path}")
        assert args.input_path is not None, "No input bag file specified and no bag file found in ~/.ros"
    processor = BagProcessor(args.input_path, args.output_path)
    processor.process_bag()
    processor.write_output_bag()

if __name__ == "__main__":
    main()
