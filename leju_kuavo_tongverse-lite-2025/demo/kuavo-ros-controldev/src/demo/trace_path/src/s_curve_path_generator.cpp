#include "s_curve_path_generator.h"
#include "common/utils.h"
#include "nav_msgs/Odometry.h"
#include "nav_msgs/Path.h"
#include "geometry_msgs/PoseStamped.h"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2/LinearMath/Matrix3x3.h"
#include <cmath>
namespace trace_path {

nav_msgs::Path SCurvePathGenerator::GeneratePath(const geometry_msgs::Pose& start_pose)
{   
    nav_msgs::Path path;
    path.header.frame_id = "odom";
    path.header.stamp = ros::Time::now();

    // 当前位置
    double current_x = start_pose.position.x;
    double current_y = start_pose.position.y;

    // 从四元数中提取朝向角（yaw）
    double current_theta = utils::GetYawFromOrientation(start_pose.orientation);

    // 计算每一步的距离
    double length = params_.length;
    double amplitude = params_.amplitude; 
    double speed = params_.speed;
    double dt = params_.dt;
    double step_distance = speed * dt;

    // 总长度
    double total_length = length * 2; // S形曲线的实际长度为两倍的长度

    // 插入路径中的点
    double x = current_x;
    double y = current_y;
    double t = 0.0;

    while (t < total_length) {
        // 计算"S"形曲线的Y坐标偏移量
        double y_offset = amplitude * std::sin(M_PI * t / length) * std::sin(M_PI * t / length);
        double dy_dt = (2 * amplitude * M_PI / length) * std::cos(M_PI * t / length) * std::sin(M_PI * t / length);

        // 创建PoseStamped消息并加入到路径中
        double yaw = std::atan2(dy_dt, 1.0); // 计算该点的朝向
        yaw = utils::NormalizeAngle(yaw + current_theta);
        geometry_msgs::PoseStamped pose_stamped = this->CreatePoseStamped(x, y + y_offset, yaw);
        path.poses.push_back(pose_stamped);

        // 更新位置
        x += step_distance;
        t += step_distance;
    }

    // 根据机器人的朝向旋转路径中的所有点
    for (auto& pose : path.poses) {
        double x = pose.pose.position.x - current_x;
        double y = pose.pose.position.y - current_y;
        auto pose2d = utils::RotatePoint(x, y, current_theta);
        pose.pose.position.x = pose2d.x + current_x;
        pose.pose.position.y = pose2d.y + current_y;
    }

    return path;
}

} // namespace trace_path