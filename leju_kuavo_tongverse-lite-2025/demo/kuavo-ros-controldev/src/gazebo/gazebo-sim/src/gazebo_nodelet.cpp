#include <thread>
#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>
#include "gazebo-sim/joint_cmd_publisher.h"
#include "gazebo-sim/sensor_data_publisher.h"
#include <gazebo_ros_control/default_robot_hw_sim.h>
#include <controller_manager/controller_manager.h>

class GazeboNodelet : public nodelet::Nodelet
{
public:
    GazeboNodelet() {}
    ~GazeboNodelet() {}
    virtual void onInit()
    {
        NODELET_INFO("Initializing GazeboNodelet");
        nh_ = getNodeHandle();

        // Controller manager initialization
        // controller_manager_.reset(new controller_manager::ControllerManager(this, nh_));
        // controller_looper_thread_ = std::thread(&GazeboNodelet::controllerLooper, this);

        sensor_data_publisher_ = new SensorDataPublisher(nh_);
        joint_cmd_publisher_ = new JointCmdPublisher(nh_);
        control_thread_ = std::thread(&GazeboNodelet::controlLoop, this);
        
        NODELET_INFO("GazeboNodelet initialized");
    }

private:
    void controlLoop()
    {
        sensor_data_publisher_->start();
        joint_cmd_publisher_->start();
    }
    void controllerLooper()
    {
        double frequency = 500.0;  // Adjust the frequency as needed
        ros::Rate rate(frequency);  // Adjust the rate as needed
        while (ros::ok())
        {
            controller_manager_->update(ros::Time::now(), ros::Duration(1.0 / frequency));
            rate.sleep();
        }
    }

    std::shared_ptr<controller_manager::ControllerManager> controller_manager_;
    ros::NodeHandle nh_;
    std::thread control_thread_;
    std::thread controller_looper_thread_;
    JointCmdPublisher *joint_cmd_publisher_;
    SensorDataPublisher *sensor_data_publisher_;
};

PLUGINLIB_EXPORT_CLASS(GazeboNodelet, nodelet::Nodelet)
