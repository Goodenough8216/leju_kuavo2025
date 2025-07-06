#include <ros/ros.h>
#include "kuavo_monitor/kuavo_monitor.h"

int main(int argc, char **argv)
{
    ros::init(argc, argv, "kuavo_monitor_node");
    ros::NodeHandle nh;

    std::string taskFile;
    double controlFrequency = 500.0; // 1000Hz
    nh.getParam("/taskFile", taskFile);
    nh.getParam("/wbc_frequency", controlFrequency);
    HighlyDynamic::KuavoMonitor kuavo_monitor(nh, taskFile, controlFrequency);
    kuavo_monitor.run();
    return 0;
}