#include "gazebo-sim/sensor_data_publisher.h"
int main(int argc, char** argv) {
    ros::init(argc, argv, "sensor_data_publisher");

    ros::NodeHandle node_handle;
    // 创建并运行节点
    SensorDataPublisher sensor_data_publisher(node_handle);
    sensor_data_publisher.start();

    // 循环处理回调函数
    ros::spin();

    return 0;
}
