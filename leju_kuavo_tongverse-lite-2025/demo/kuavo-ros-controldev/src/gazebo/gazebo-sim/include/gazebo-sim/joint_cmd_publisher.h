#ifndef JOINT_CMD_PUBLISHER_H
#define JOINT_CMD_PUBLISHER_H

#include <ros/ros.h>
#include <std_msgs/Header.h>
#include <std_msgs/Float64MultiArray.h>
#include <kuavo_msgs/jointCmd.h>
#include <gazebo_msgs/SetModelConfiguration.h>
#include <gazebo_msgs/ModelState.h>
#include <std_msgs/Float64MultiArray.h>
#include <gazebo_msgs/GetModelState.h>
#include <gazebo_msgs/SetModelState.h>
#include <std_srvs/SetBool.h>

class JointCmdPublisher
{
public:
    JointCmdPublisher(ros::NodeHandle &nh);
    void waitForGazeboModelToLoad();
    bool handleSimStart(std_srvs::SetBool::Request &req, std_srvs::SetBool::Response &res);
    void waitingForSimStart();
    std::string extractRobotNameFromDescription(const std::string &robot_description);
    void jointCmdCallback(const kuavo_msgs::jointCmd::ConstPtr &joint_cmd_msg);
    void waitForParams();
    void setGazeboInitialState();
    void setTorsoPositionAndOrientation(const std::string &model_name, const geometry_msgs::Point &position, const geometry_msgs::Quaternion &orientation);
    void start();

private:
    ros::NodeHandle nh_;
    ros::ServiceClient set_model_config_client_;
    ros::ServiceClient set_model_state_client_;
    ros::Subscriber joint_cmd_subscriber_;
    ros::Publisher joints_effort_publisher_;
    std_msgs::Float64MultiArray joint_effort_msg_;
    std::vector<double> robot_init_state_param_;
    std::vector<std::string> joint_names_;
    int active_joint_count_;
    std::string robot_name_;
    bool sim_start_received_ = false;
};

#endif // JOINT_CMD_PUBLISHER_H
