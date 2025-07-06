#!/bin/bash
export ROS_WS_PATH=/opt/ros/noetic

source $ROS_WS_PATH/setup.bash
source $KUAVO_ROS_CONTROL_WS_PATH/devel/setup.bash
H12PRO_CONTROLLER_NODE_DIR=$(rospack find h12pro_controller_node)
export PYTHONPATH=$PYTHONPATH:$H12PRO_CONTROLLER_NODE_DIR
cd $H12PRO_CONTROLLER_NODE_DIR

echo "current robot version: $ROBOT_VERSION"
echo "ROS_MASTER_URI: $ROS_MASTER_URI"
echo "ROS_IP: $ROS_IP"

# Check if both nodes are not running
if ! rosnode list | grep -q "/h12pro_channel_publisher" && ! rosnode list | grep -q "/joy_node"; then
    roslaunch h12pro_controller_node h12pro_autostart.launch
else
    echo "Node /h12pro_channel_publisher or /joy_node is already running"
    echo "Please check running nodes with 'rosnode list'"
fi
