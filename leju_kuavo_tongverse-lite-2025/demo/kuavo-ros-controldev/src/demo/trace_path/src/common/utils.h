#ifndef TRACK_TRAJ_PATH_UTILS_H
#define TRACK_TRAJ_PATH_UTILS_H
#include <array>
#include "geometry_msgs/Pose2D.h"
#include "geometry_msgs/Point.h"
#include "geometry_msgs/Quaternion.h"

namespace trace_path {
namespace utils {

/// @brief 从四元数中获取Pitch角度
double GetPitchFromOrientation(const geometry_msgs::Quaternion &ori);

/// @brief 从四元数中获取Roll角度
double GetRollFromOrientation(const geometry_msgs::Quaternion &ori);

/// @brief 从四元数中获取yaw角度
double GetYawFromOrientation(const geometry_msgs::Quaternion &ori);

/// @brief 绕原点旋转指定角度
/// @param theta 旋转角度，以弧度表示
geometry_msgs::Pose2D RotatePoint(double x, double y, double theta);

/// @brief 计算两点之间的角度
double CalculateAngle(geometry_msgs::Point goal, geometry_msgs::Point current);

/// @brief 计算两点之间的距离
double CalculateDistance(geometry_msgs::Point goal, geometry_msgs::Point current);

// @brief 用于归一化角度到 -pi 到 pi 之间的辅助函数
double NormalizeAngle(double angle);
} // namespace  utils
} // namespace trace_path
#endif