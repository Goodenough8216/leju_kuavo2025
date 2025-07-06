#ifndef PID_PATH_TRACER_H
#define PID_PATH_TRACER_H
#include "path_tracer.h"
#include "path_generator.h"

#include "ros/ros.h"
#include "nav_msgs/Path.h"
#include "nav_msgs/Odometry.h"

namespace trace_path {

class PidPathTracer : public PathTracerBase
{
public:
    PidPathTracer(ros::NodeHandle &nh);
    
    /// @brief 基于 PID 控制算法的轨迹追踪
    /// @param path_generator 轨迹路径生成器
    void Follow(const PathGeneratorUniquePtr &path_generator) override;

private:
    /* PID */
    double Kp = 0.5; // 比例增益
    double Ki = 0.1; // 积分增益
    double Kd = 0.1; // 微分增益
    double integral_error = 0.0; // 积分误差
    double prev_error = 0.0;     // 上一时刻的误差
    ros::Time last_time;         // 上一次循环的时间
};
} // namespace  trace_path
# endif