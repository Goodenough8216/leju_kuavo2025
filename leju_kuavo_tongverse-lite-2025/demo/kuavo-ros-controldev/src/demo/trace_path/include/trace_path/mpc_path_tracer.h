#ifndef MPC_PATH_TRACER_H
#define MPC_PATH_TRACER_H
#include "path_tracer.h"
#include "path_generator.h"

#include <vector>

#include <Eigen/Dense>
#include <ros/ros.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/PoseStamped.h>

namespace trace_path {

class MpcPathTracer : public PathTracerBase
{
public:
    MpcPathTracer(ros::NodeHandle &nh);
    
    /// @brief 基于 MPC 控制算法的轨迹追踪
    /// @param path_generator 轨迹路径生成器
    void Follow(const PathGeneratorUniquePtr &path_generator) override;
    
private:
    void PublishMpcPath(const Eigen::VectorXd& x, const Eigen::VectorXd& y, const Eigen::VectorXd& yaw);

    ros::Publisher mpc_path_pub_;   // trace_path/mpc/path mpc 求解的局部路径
};
} // namespace  trace_path
# endif