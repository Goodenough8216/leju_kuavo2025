#include "gazebo-sim/joint_cmd_publisher.h"
int main(int argc, char **argv)
{
    ros::init(argc, argv, "joint_cmd_publisher");

    ros::NodeHandle nh;
    // 创建节点并启动
    JointCmdPublisher joint_cmd_publisher(nh);
    joint_cmd_publisher.start();
    // 循环等待并处理消息
    ros::spin();

    return 0;
}
