#ifndef TRACE_PATH_BASE_H
#define TRACE_PATH_BASE_H
#include <memory>

#include "path_generator.h"

#include "ros/ros.h"
#include "nav_msgs/Path.h"
#include "nav_msgs/Odometry.h"

namespace trace_path {

class PathTracerBase
{
public:
    PathTracerBase(ros::NodeHandle &nh);
    
    /// @brief 跟随生成器返回的轨迹路径运动
    /// @param path_generator 轨迹路径生成器
    virtual void Follow(const PathGeneratorUniquePtr &path_generator);

    /// @brief 设置行走速度
    void set_walk_velocity(double velocity);
    
    /// @brief 设置最大线速度 m/s
    void set_max_linear_velocity(double v);
    /// @brief 设置最大角速度 
    void set_max_angular_velocity(double v);
protected:
    void PublishPath(const nav_msgs::Path &path);
    void PublishCmdVel(double linear_velocity, double angular_velocity);
    
private:
    void odomCallback(const nav_msgs::Odometry::ConstPtr& odom_msg);

protected:
    const double kMaxLinearVelocity = 0.5;   // 最大线速度
    const double kMaxAngularVelocity = 0.5;  // 最大角速度

    bool flag_reset_start_point_ = false;
    nav_msgs::Odometry::ConstPtr start_point_odom_ = nullptr;
    geometry_msgs::Pose current_pose_;  // 当前位置
    nav_msgs::Path path_;               // 实时运动轨迹
    double walk_velocity_ = 0.3;        // 行走速度 默认 0.3 m/s
    double v_max_linear_ = 0.75;        // 最大线速度
    double v_max_angular_ = 0.5;        // 最大角速度
    /* ROS */
    ros::NodeHandle nh_;
    ros::Subscriber odom_sub_;      // 订阅里程计信息 /odom
    ros::Publisher  cmd_vel_pub_;   // 控制速度运动   /cmd_vel
    ros::Publisher  path_pub_;      // 固定路径发布   trace_path/path
    ros::Publisher  realtime_path_pub_; // 实时轨迹发布 trace_path/realtime_path
};

class PathTracerFactory
{
public:
    template<typename GeneratorType, typename... Args>
    static std::unique_ptr<PathTracerBase> Create(Args... args)
    {
        return std::make_unique<GeneratorType>(args...);
    }
};

} // namespace  trace_path
# endif