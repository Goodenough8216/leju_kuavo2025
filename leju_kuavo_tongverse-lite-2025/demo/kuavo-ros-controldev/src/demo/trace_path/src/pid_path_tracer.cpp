#include "pid_path_tracer.h"
#include "common/utils.h"

#include <cmath>
#include <iostream>
#include <thread>
#include <chrono>

#include <tf/tf.h>

namespace trace_path {
PidPathTracer::PidPathTracer(ros::NodeHandle &nh):PathTracerBase(nh)
{
}

void PidPathTracer::Follow(const PathGeneratorUniquePtr &path_generator)
{
    ros::Rate loop_rate(100); // 控制频率

    // reset
    flag_reset_start_point_ = true;
    while(ros::ok() && flag_reset_start_point_) {
        ros::spinOnce();
        loop_rate.sleep();
        std::cout << "Waitting for start point... \n";
    }

    auto path = path_generator->GeneratePath(start_point_odom_->pose.pose);
    if (path.poses.size() <= 0)  return;

    // 先发布路径提供查看
    PublishPath(path);

    // TODO 添加 按下 Enter 开始
    std::cout << "Following...\n";
    std::cout << "-------------------------------------------------------\n";
    std::cout << "Path Start Pose:\n" << path.poses.front().pose << "\n";
    std::cout << "-------------------------------------------------------\n";
    std::cout << "Path End Pose:\n" << path.poses.back().pose << std::endl;
    std::cout << "-------------------------------------------------------\n";
    std::cout << "Current yaw: " << utils::GetYawFromOrientation(current_pose_.orientation) << std::endl;

    int target_index = 0;    // 当前的目标点索引
    last_time = ros::Time::now(); // 初始化时间
    prev_error = 0.0;

    const double DISTANCE_THRESHOLD = 0.2;
    const double SLOW_DOWN_DISTANCE = 0.3;
    const double ANGLE_ERROR_THRESHOLD = M_PI / 6;

    while (ros::ok() && target_index < path.poses.size()) {
        ros::Time current_time = ros::Time::now();
        ros::Duration duration = current_time - last_time;
        last_time = current_time;

        // 获取目标点位置
        geometry_msgs::Point target_point = path.poses[target_index].pose.position;
        geometry_msgs::Point current_point = current_pose_.position;

        // 计算目标方向角
        double target_angle = utils::CalculateAngle(target_point, current_point);
        // 计算当前方向角
        double current_angle = tf::getYaw(current_pose_.orientation);

        // 计算误差
        double error = utils::NormalizeAngle(target_angle - current_angle);

        // 积分项
        integral_error += error * duration.toSec();

        // 微分项
        double delta_t = duration.toSec();
        double derivative = (error - prev_error) / (delta_t == 0 ? 1 : delta_t);
        prev_error = error;

        // PID控制
        double angular_speed = Kp * error + Ki * integral_error + Kd * derivative;

        // 计算线速度
        double distance_to_target = utils::CalculateDistance(target_point, current_point);
        double linear_speed = std::min(walk_velocity_, kMaxLinearVelocity);

        // 如果角度误差大于阈值，进一步降低线速度
        if (fabs(error) > ANGLE_ERROR_THRESHOLD) {
            linear_speed *= 0.3;
        }

        // 如果距离小于一定阈值，降低速度
        if (distance_to_target < SLOW_DOWN_DISTANCE) {
            linear_speed *= 0.3;
        }

        // 检查是否到达目标点
        if (distance_to_target < DISTANCE_THRESHOLD) {
            ++target_index;
            integral_error = 0.0; // 重置积分误差
        }

        // std::cout << "current_point:" << current_point << "\n";
        // std::cout << "target_point:" << target_point << "\n";
        // std::cout << "distance_to_target:" << distance_to_target << "\n";
        
        // 发布速度命令
        PublishCmdVel(linear_speed, std::min(angular_speed, kMaxAngularVelocity));
        PublishPath(path);

        ros::spinOnce();
        loop_rate.sleep();
    }

    // sleep 1.5s
    ros::Duration(1.5).sleep();
    ROS_INFO("Path Follower Finished!");
    std::cout << "current odom pose:\n" << current_pose_ << std::endl;
}

} // namespace trace_path 