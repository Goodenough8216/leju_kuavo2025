#ifndef CIRCLE_PATH_GENERATOR_H
#define CIRCLE_PATH_GENERATOR_H
#include "path_generator.h"

namespace trace_path {

/// @brief 生成圆形路径所需的参数
struct CircleParams : public BaseParams
{
    double radius; // 半径

    CircleParams():BaseParams(), radius(0.0) {}

    CircleParams(double radius_)
        :BaseParams(), radius(radius_) {}
    CircleParams(double radius_, double speed_, double dt_)
        :BaseParams(speed_, dt_), radius(radius_) {}
};

/// @brief 圆形路径生成器
class CirclePathGenerator : public PathGeneratorBase {
public:
    CirclePathGenerator(const CircleParams& parameters) : params_(parameters) {}
    virtual nav_msgs::Path GeneratePath(const geometry_msgs::Pose& start_pose) override;

    /// @brief 设置生成圆形路径的方向
    /// @param clockwise true 表示顺时针，false 表示逆时针
    void set_clockwise(bool clockwise);

    /// @brief  获取生成圆形路径的方向
    /// @return 
    bool get_clockwise() const {return clockwise_;}
private:
    CircleParams params_;
    bool clockwise_ = false; // true 表示顺时针，false 表示逆时针 
};

} // namespace trace_path

#endif