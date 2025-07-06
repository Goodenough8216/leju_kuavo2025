#include "circle_path_generator.h"
#include "common/utils.h"

#include <cmath>
#include "nav_msgs/Odometry.h"
#include "nav_msgs/Path.h"
#include "geometry_msgs/PoseStamped.h"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.h"

namespace trace_path {

nav_msgs::Path CirclePathGenerator::GeneratePath(const geometry_msgs::Pose& start_pose)
{   
    nav_msgs::Path path;
    path.header.frame_id = "odom";
    path.header.stamp = ros::Time::now();

    // 当前位置和朝向 当前点为圆形路径上一点，朝向即该点的切线方向
    double current_x = start_pose.position.x;
    double current_y = start_pose.position.y;
    double current_theta = utils::GetYawFromOrientation(start_pose.orientation);
    
    // 逆时针方向: 1, 顺时针方向: -1    
    // 圆心位置和朝向 圆在当前点切线垂直线上
    int direction = clockwise_? -1 : 1;
    double center_x = current_x + params_.radius * cos(current_theta + direction * M_PI_2);
    double center_y = current_y + params_.radius * sin(current_theta + direction * M_PI_2);
    double center_theta = current_theta + direction * M_PI_2; // 朝向为当前yaw的垂直向

    // 计算每一步的角度变化
    double angular_speed = params_.speed / params_.radius; // 角速度
    double angle_step = direction * angular_speed * params_.dt; // 每一步的角度变化

    // 半个圆周角
    double half_pi_direction = direction * M_PI;

    // 插入路径中的点
    double angle = 0.0;
    while (fabs(direction * angle) < 2 * M_PI) {
        auto theta = utils::NormalizeAngle(angle + center_theta + half_pi_direction);
        double x = center_x + params_.radius * cos(theta);
        double y = center_y + params_.radius * sin(theta);

        // 创建PoseStamped消息并加入到路径中
        double yaw = utils::NormalizeAngle(angle + center_theta - direction * M_PI_2);
        geometry_msgs::PoseStamped pose_stamped = this->CreatePoseStamped(x, y, yaw);
        path.poses.push_back(pose_stamped);

        // 更新角度
        angle += angle_step;
    }

    // 确保最后一个点与起始点重合，形成封闭的圆
    if (fabs(angle - 2 * M_PI) > 1e-6) { // 使用一个小的容差值
        double x = center_x + params_.radius * cos(2 * M_PI + center_theta + half_pi_direction);
        double y = center_y + params_.radius * sin(2 * M_PI + center_theta + half_pi_direction);
        double yaw = utils::NormalizeAngle(angle + center_theta - direction * M_PI_2);
        geometry_msgs::PoseStamped last_pose_stamped = this->CreatePoseStamped(x, y, yaw);
        path.poses.push_back(last_pose_stamped);
    }

    return path;
}

void CirclePathGenerator::set_clockwise(bool clockwise)
{
    clockwise_ = clockwise;
}

} // namespace trace_path