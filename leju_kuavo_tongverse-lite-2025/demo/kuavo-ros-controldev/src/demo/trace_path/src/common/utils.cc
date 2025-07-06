#include "utils.h"
#include <cmath>
#include "tf2/LinearMath/Quaternion.h"
#include "tf2/LinearMath/Matrix3x3.h"

namespace trace_path {
namespace utils {

double GetPitchFromOrientation(const geometry_msgs::Quaternion &ori)
{
    // 从四元数中提取朝向角（yaw）
    tf2::Quaternion q(ori.x, ori.y, ori.z, ori.w);
    tf2::Matrix3x3 m(q);
    double roll, pitch, yaw;
    m.getRPY(roll, pitch, yaw);
    return NormalizeAngle(pitch);
}

double GetRollFromOrientation(const geometry_msgs::Quaternion &ori)
{   
    // 从四元数中提取朝向角（yaw）
    tf2::Quaternion q(ori.x, ori.y, ori.z, ori.w);
    tf2::Matrix3x3 m(q);
    double roll, pitch, yaw;
    m.getRPY(roll, pitch, yaw);
    return NormalizeAngle(roll);
}

double GetYawFromOrientation(const geometry_msgs::Quaternion &ori)
{
    // 从四元数中提取朝向角（yaw）
    tf2::Quaternion q(ori.x, ori.y, ori.z, ori.w);
    tf2::Matrix3x3 m(q);
    double roll, pitch, yaw;
    m.getRPY(roll, pitch, yaw);
    return NormalizeAngle(yaw);
}

geometry_msgs::Pose2D RotatePoint(double x, double y, double theta)
{
    geometry_msgs::Pose2D pose2d;

    pose2d.x = x * std::cos(theta) - y * std::sin(theta);
    pose2d.y = x * std::sin(theta) + y * std::cos(theta);
    
    return pose2d;
}

double CalculateAngle(geometry_msgs::Point goal, geometry_msgs::Point current)
{
    double dx = goal.x - current.x;
    double dy = goal.y - current.y;
    return std::atan2(dy, dx);
}

double CalculateDistance(geometry_msgs::Point goal, geometry_msgs::Point current)
{
    // std::cout << "CalculateDistance: " << std::pow(goal.x - current.x, 2) << std::endl;
    // std::cout << "CalculateDistance: " << std::pow(goal.y - current.y, 2) << std::endl;
    return std::sqrt(std::pow(goal.x - current.x, 2) + std::pow(goal.y - current.y, 2));
}

double NormalizeAngle(double angle) 
{
    return std::atan2(std::sin(angle), std::cos(angle));
}
} // namespace  utils
} // namespace trace_path