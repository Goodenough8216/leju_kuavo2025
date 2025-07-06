#include "path_tracer.h"
#include "common/utils.h"

#include <cmath>
#include <iostream>
#include <thread>
#include <chrono>

#include <tf/tf.h>

namespace trace_path {
PathTracerBase::PathTracerBase(ros::NodeHandle &nh):nh_(nh)
{
    odom_sub_ = nh_.subscribe("/odom", 1, &PathTracerBase::odomCallback, this);
    cmd_vel_pub_ = nh_.advertise<geometry_msgs::Twist>("/cmd_vel", 10, true);
    path_pub_ = nh_.advertise<nav_msgs::Path>("trace_path/path", 10, true);
    realtime_path_pub_ = nh_.advertise<nav_msgs::Path>("trace_path/realtime_path", 10, true);

    path_.header.frame_id = "odom";
    path_.header.stamp = ros::Time::now();
}

void PathTracerBase::Follow(const PathGeneratorUniquePtr &path_generator)
{
    // Base case
    return;
}

void PathTracerBase::set_walk_velocity(double velocity)
{
    this->walk_velocity_ = velocity;
}

void PathTracerBase::set_max_linear_velocity(double v) 
{ 
    v_max_linear_ = v; 
}

void PathTracerBase::set_max_angular_velocity(double v) 
{
    v_max_angular_ = v;
}

void PathTracerBase::odomCallback(const nav_msgs::Odometry::ConstPtr& odom_msg)
{
    // std::cout << "odomCallback odom:\n" << odom_msg->pose.pose << std::endl;
    if (flag_reset_start_point_) {
        start_point_odom_ = odom_msg;
        flag_reset_start_point_ = false;
        std::cout << "start_point_odom:\n" << start_point_odom_->pose.pose << std::endl;
    }

    current_pose_ = odom_msg->pose.pose;
    
    // 发布实时运动轨迹
    geometry_msgs::PoseStamped pose_stamped;
    pose_stamped.header.frame_id = path_.header.frame_id;
    pose_stamped.header.stamp = ros::Time::now();
    pose_stamped.pose = current_pose_;
    pose_stamped.pose.position.z = 0.0;
    path_.poses.push_back(pose_stamped);
    realtime_path_pub_.publish(path_);
}

void PathTracerBase::PublishPath(const nav_msgs::Path &path)
{
    path_pub_.publish(path);
}

void PathTracerBase::PublishCmdVel(double linear_velocity, double angular_velocity)
{
    // 发布控制命令
    geometry_msgs::Twist cmd_vel;
    cmd_vel.linear.x = linear_velocity;
    cmd_vel.angular.z = angular_velocity;
    cmd_vel_pub_.publish(cmd_vel);
    // std::cout << "cmd_vel:\n" << cmd_vel << std::endl;
}

} // namespace trace_path 