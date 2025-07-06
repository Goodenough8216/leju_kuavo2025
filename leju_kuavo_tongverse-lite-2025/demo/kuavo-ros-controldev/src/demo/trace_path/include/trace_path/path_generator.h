#ifndef GENERATOR_PATH_H
#define GENERATOR_PATH_H

#include <memory>
#include "nav_msgs/Path.h"
#include <geometry_msgs/Pose.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

namespace trace_path {

class PathGeneratorBase;
using PathGeneratorPtr = std::shared_ptr<PathGeneratorBase>;
using PathGeneratorUniquePtr = std::unique_ptr<PathGeneratorBase>;

struct BaseParams
{
    double speed = 0.4; // 速度
    double dt = 1.0;    // 时间步长

    BaseParams(): speed(0.4), dt(1.0){}
    BaseParams(double speed_, double dt_)
        : speed(speed_), dt(dt_) {}
};

class PathGeneratorBase
{
public: /* Interface */
    /// @brief 根据起始位置生成一条轨迹路径
    /// @param start_pose 起始位置的Pose对象
    virtual nav_msgs::Path GeneratePath(const geometry_msgs::Pose& start_pose) = 0;

public: /* Helper Functions */
    /**
     * 创建并返回一个geometry_msgs::PoseStamped对象。
     * 
     * 该函数用于根据给定的位置坐标x, y和可选的参考帧frame_id，
     * 创建一个时间戳并设置其位置信息，同时设置四元数表示的朝向为无旋转状态。
     * 
     * @param x 位置的x坐标。
     * @param y 位置的y坐标。
     * @param frame_id 参考帧的标识符，默认为"odom"。
     * 
     * @return 创建的PoseStamped对象，包含设置的位置和时间信息。
     */
    geometry_msgs::PoseStamped CreatePoseStamped(double x, double y, double yaw, const std::string& frame_id = "odom")
    {
        geometry_msgs::PoseStamped pose_stamped;
        pose_stamped.header.frame_id = frame_id;
        pose_stamped.header.stamp = ros::Time::now();
        pose_stamped.pose.position.x = x;
        pose_stamped.pose.position.y = y;
        // yaw
        tf2::Quaternion q;
        q.setRPY(0, 0, yaw);
        pose_stamped.pose.orientation = tf2::toMsg(q);
        return pose_stamped;
    }
};

// 工厂类，用于创建不同类型的路径生成器
class PathGeneratorFactory
{
public:
    template<typename GeneratorType, typename... Args>
    static std::unique_ptr<PathGeneratorBase> Create(Args... args)
    {
        return std::make_unique<GeneratorType>(args...);
    }
};

} // namespace trace_path
#endif