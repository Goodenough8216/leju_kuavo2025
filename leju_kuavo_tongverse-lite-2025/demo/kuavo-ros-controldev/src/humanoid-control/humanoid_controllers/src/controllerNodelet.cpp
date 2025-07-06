
#include "humanoid_controllers/humanoidController.h"
#include <thread>
#include <ros/ros.h>
#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>

using Duration = std::chrono::duration<double>;
using Clock = std::chrono::high_resolution_clock;

class HumanoidControllerNodelet : public nodelet::Nodelet
{
public:
    virtual void onInit()
    {
        NODELET_INFO("Initializing HumanoidControllerNodelet nodelet...");
        nh = getNodeHandle();
        robot_hw = new humanoid_controller::HybridJointInterface(); // TODO:useless
        pause_sub = nh.subscribe<std_msgs::Bool>("pauseFlag", 1, &HumanoidControllerNodelet::pauseCallback, this);

        control_thread = std::thread(&HumanoidControllerNodelet::controlLoop, this);
        NODELET_INFO("HumanoidControllerNodelet nodelet initialized.");
    }

private:
    bool pause_flag{false};
    ros::NodeHandle nh;
    humanoid_controller::HybridJointInterface *robot_hw;
    ros::Subscriber pause_sub;
    humanoid_controller::humanoidController *controller_ptr_;
    std::chrono::high_resolution_clock::time_point lastTime;
    std::thread control_thread;
    void pauseCallback(const std_msgs::Bool::ConstPtr &msg)
    {
        pause_flag = msg->data;
        std::cerr << "pause_flag: " << pause_flag << std::endl;
    }

    void controlLoop()
    {
        int estimator_type = 1;
        bool with_estimation = false;
        if (nh.hasParam("/estimator_type"))
        {
            nh.getParam("/estimator_type", estimator_type);
        }
        else
        {
            ROS_INFO("estimator_type not found in parameter server");
        }
        if (estimator_type == 1)
        {
            std::cout << "Using nomal estimator" << std::endl;
            controller_ptr_ = new humanoid_controller::humanoidController();
        }
        else if (estimator_type == 2)
        {
#ifdef KUAVO_CONTROL_LIB_FOUND

            std::cout << "Using inEKF estimator" << std::endl;
            controller_ptr_ = new humanoid_controller::humanoidKuavoController();
#else
            ROS_ERROR("Kuavo control library not found. Please make sure to clone submodule kuavo-ros-control-lejulib into src/ and rebuild the workspace.");
            exit(1);
#endif
        }
        else
        {
            std::cout << "Using cheater estimator" << std::endl;
            controller_ptr_ = new humanoid_controller::humanoidCheaterController();
        }
        if (!controller_ptr_->init(robot_hw, nh, true))
        {
            ROS_ERROR("Failed to initialize the humanoid controller!");
            return;
        }

        auto startTime = std::chrono::high_resolution_clock::now();
        auto startTimeROS = ros::Time::now();
        controller_ptr_->starting(startTimeROS);
        lastTime = startTime;
        double controlFrequency = 500.0; // 1000Hz
        nh.getParam("/wbc_frequency", controlFrequency);
        ROS_INFO_STREAM("Wbc control frequency: " << controlFrequency);
        struct timespec next_time;
        clock_gettime(CLOCK_MONOTONIC, &next_time);
        ros::Rate rate(controlFrequency);
        uint64_t cycle_count = 0;
        while (ros::ok() && controller_ptr_->preUpdateComplete() != true)
        {
            controller_ptr_->preUpdate(ros::Time::now());
        }

        while (ros::ok())
        {
            // std::cout << "\n\nControlLoop: "<<cycle_count++ << std::endl;
            if (!pause_flag)
            {
                // std::cout << "controlLoop: Running control loop" << std::endl;
                const auto currentTime = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> time_span = currentTime - lastTime;
                ros::Duration elapsedTime(time_span.count());
                lastTime = currentTime;

                // Control
                controller_ptr_->update(ros::Time::now(), elapsedTime);

                // Sleep
                next_time.tv_sec += (next_time.tv_nsec + 1 / controlFrequency * 1e9) / 1e9;
                next_time.tv_nsec = (int)(next_time.tv_nsec + 1 / controlFrequency * 1e9) % (int)1e9;
                clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &next_time, NULL);

                // Add print statement if the cycle time exceeds 1 second
                const auto cycleTime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - currentTime).count();
                if (cycleTime > 1000)
                {
                    ROS_ERROR_STREAM("WBC Cycle time exceeded 1 second: " << cycleTime << "ms");
                }
            }
            rate.sleep();
        }
        std::cout << "controlLoop: Exiting control loop" << std::endl;
    }
};

PLUGINLIB_EXPORT_CLASS(HumanoidControllerNodelet, nodelet::Nodelet)
