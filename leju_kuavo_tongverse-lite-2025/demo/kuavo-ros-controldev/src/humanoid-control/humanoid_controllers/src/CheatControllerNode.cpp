
#include "humanoid_controllers/humanoidController.h"
#include <thread>

using Duration = std::chrono::duration<double>;
using Clock = std::chrono::high_resolution_clock;

bool pause_flag = false;

void pauseCallback(const std_msgs::Bool::ConstPtr &msg)
{
    pause_flag = msg->data;
    std::cerr << "pause_flag: " << pause_flag << std::endl;
}

int main(int argc, char **argv)
{
    ros::Duration elapsedTime_;

    ros::init(argc, argv, "humanoid_controller_node");
    ros::NodeHandle nh;
    humanoid_controller::HybridJointInterface *robot_hw;
    // create a subscriber to pauseFlag
    ros::Subscriber pause_sub = nh.subscribe<std_msgs::Bool>("pauseFlag", 1, pauseCallback);
    humanoid_controller::humanoidCheaterController controller;
    if (!controller.init(robot_hw, nh))
    {
        ROS_ERROR("Failed to initialize the humanoid controller!");
        return -1;
    }

    auto startTime = Clock::now();
    auto startTimeROS = ros::Time::now();
    controller.starting(startTimeROS);
    auto lastTime = startTime;

    // create a thread to spin the node
    std::thread spin_thread([]()
                            { ros::spin(); });
    spin_thread.detach();
    double controlFrequency = 500.0; // 1000Hz
    nh.getParam("/wbc_frequency", controlFrequency);
    std::cout << "Wbc control frequency: " << controlFrequency << std::endl;
    ros::Rate rate(controlFrequency);
    struct timespec next_time;
    clock_gettime(CLOCK_MONOTONIC, &next_time);
    while (ros::ok())
    {
        if (!pause_flag)
        {
            const auto currentTime = Clock::now();
            // Compute desired duration rounded to clock decimation
            // const Duration desiredDuration(1.0 / 1000);

            // Get change in time
            Duration time_span = std::chrono::duration_cast<Duration>(currentTime - lastTime);
            elapsedTime_ = ros::Duration(time_span.count());
            lastTime = currentTime;

            // Check cycle time for excess delay
            // ...

            // Control
            // let the controller compute the new command (via the controller manager)
            controller.update(ros::Time::now(), elapsedTime_);

            // Sleep
            const auto beforeSleep = Clock::now();
            // const auto sleepTill = currentTime + std::chrono::duration_cast<Clock::duration>(desiredDuration);
            // std::this_thread::sleep_until(sleepTill);
            // rate.sleep();
            next_time.tv_sec += (next_time.tv_nsec + 1 / controlFrequency * 1e9) / 1e9;
            next_time.tv_nsec = (int)(next_time.tv_nsec + 1 / controlFrequency * 1e9) % (int)1e9;

            clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &next_time, NULL);

            // Add print statement if the cycle time exceeds 1 second
            const auto cycleTime = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - currentTime).count();
            if (cycleTime > 1000)
            {
                ROS_INFO_STREAM("Cycle time exceeded 1 second: " << cycleTime << "ms");
                const auto sleepTime = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - beforeSleep).count();
                ROS_INFO_STREAM("Sleep time: " << sleepTime << "ms");
                const auto proccedTime = std::chrono::duration_cast<std::chrono::milliseconds>(beforeSleep - currentTime).count();

                ROS_INFO_STREAM("proccedTime duration: " << proccedTime << "s");
            }
        }
    }

    return 0;
}
