#include "square_path_generator.h"
#include "common/utils.h"

#include <cmath>
#include <cstdint>

#include "nav_msgs/Odometry.h"
#include "nav_msgs/Path.h"
#include "geometry_msgs/PoseStamped.h"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2/LinearMath/Matrix3x3.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.h"
namespace trace_path {

nav_msgs::Path SquarePathGenerator::GeneratePath(const geometry_msgs::Pose& start_pose)
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
    double step_distance = params_.speed * params_.dt;

    // 正方形的四个顶点
    std::vector<std::pair<double, double>> square_points = {
        {current_x, current_y},
        {current_x + params_.side_length, current_y},
        {current_x + params_.side_length, current_y + params_.side_length},
        {current_x, current_y + params_.side_length},
        {current_x, current_y} // 返回起点
    };

    // 插入路径中的点
    for (size_t i = 1; i < square_points.size(); ++i) {
        const auto& prev_point = square_points[i - 1];
        const auto& next_point = square_points[i];

        // 在两个顶点之间插入点
        double dx = next_point.first - prev_point.first;
        double dy = next_point.second - prev_point.second;
        double distance = sqrt(dx * dx + dy * dy);
        double yaw = atan2(dy, dx); // 计算当前边的朝向

        int num_steps = static_cast<int>(distance / step_distance);

        if (num_steps <= 0) {
            num_steps = 1;
        }

        // 生成路径点
        for (int j = 0; j <= num_steps; ++j) {
            double t = static_cast<double>(j) / num_steps;
            double x = prev_point.first + t * dx;
            double y = prev_point.second + t * dy;

            // 创建PoseStamped消息并加入到路径中
            path.poses.push_back(CreatePoseStamped(x, y, yaw + current_theta));
        }
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