#ifndef SENSOR_DATA_PUBLISHER_H
#define SENSOR_DATA_PUBLISHER_H

#include <ros/ros.h>
#include <sensor_msgs/JointState.h>
#include <sensor_msgs/Imu.h>
#include <kuavo_msgs/jointData.h>
#include <kuavo_msgs/imuData.h>
#include <kuavo_msgs/sensorsData.h>

class SensorDataPublisher {
public:
    SensorDataPublisher(ros::NodeHandle &nh);

    // 回调函数
    void imuCallback(const sensor_msgs::Imu::ConstPtr &imu_msg);
    void jointStateCallback(const sensor_msgs::JointState::ConstPtr &joint_state_msg);

    // 定时器回调函数
    void publishSensorData(const ros::TimerEvent &event);
    void start();
    void waitForParams();

private:
    ros::NodeHandle nh_;

    // 订阅者
    ros::Subscriber imu_subscriber_;
    ros::Subscriber joint_state_subscriber_;

    // 发布者
    ros::Publisher sensor_data_publisher_;

    // 定时器
    ros::Timer timer_;
    ros::Time sensor_time_;

    // 存储最新的 IMU 和关节数据
    sensor_msgs::Imu::ConstPtr imu_msg_;
    sensor_msgs::JointState joint_state_msg_;

    std::vector<std::string> joint_names_;
    int active_joint_count_ = 0;

};

#endif // SENSOR_DATA_PUBLISHER_H
