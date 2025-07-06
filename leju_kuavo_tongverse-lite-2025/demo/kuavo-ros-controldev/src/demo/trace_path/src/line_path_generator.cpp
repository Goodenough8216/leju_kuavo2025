#include "line_path_generator.h"
#include "common/utils.h"

#include <cmath>

#include <ros/time.h>  // 或者其他包含 ros::Time 的头文件
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseStamped.h>

namespace trace_path {

nav_msgs::Path LinePathGenerator::GeneratePath(const geometry_msgs::Pose& start_pose)
{
    nav_msgs::Path path;
    path.header.frame_id = "odom";
    path.header.stamp = ros::Time::now();

    // 提前计算每一步的距离
    double step_distance = params_.speed * params_.dt;

    // 计算路径上的点数量
    int num_points = static_cast<int>(params_.length / step_distance);

    // 确保至少有一个点
    if (num_points <= 0) {
        num_points = 1;
    }

    // 从四元数中提取朝向角（yaw）
    double yaw = utils::GetYawFromOrientation(start_pose.orientation);

    // 提前计算 cos(yaw) 和 sin(yaw)
    double cos_yaw = cos(yaw);
    double sin_yaw = sin(yaw);

    // 生成路径上的点
    for (int i = 0; i <= num_points; ++i) {
        double t = i * params_.dt; // 时间间隔

        // 创建PoseStamped消息并加入到路径中
        geometry_msgs::PoseStamped pose_stamped;
        pose_stamped.header = path.header; // 使用相同的header
        pose_stamped.pose.position.x = start_pose.position.x + t * params_.speed * cos_yaw;
        pose_stamped.pose.position.y = start_pose.position.y + t * params_.speed * sin_yaw;
        // pose_stamped.pose.position.z = start_pose.position.z; // 忽略z轴，始终为0
        pose_stamped.pose.orientation = start_pose.orientation;

        path.poses.push_back(pose_stamped);
    }

    return path;
}

} // namespace trace_path