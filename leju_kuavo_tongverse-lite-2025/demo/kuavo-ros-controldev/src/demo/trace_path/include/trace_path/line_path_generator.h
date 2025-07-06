#ifndef LINE_PATH_GENERATOR_H
#define LINE_PATH_GENERATOR_H
#include "path_generator.h"

namespace trace_path {

/// @brief 生成直线路径所需的参数
struct LineParams : public BaseParams
{
    double length; // 长度

    LineParams(): BaseParams(), length(0.0) {}
    LineParams(double length_)
        :BaseParams(), length(length_) {}
    LineParams(double length_, double speed_, double dt_)
        :BaseParams(speed_, dt_), length(length_) {}
};

/// @brief 直线轨迹路径生成器
class LinePathGenerator : public PathGeneratorBase {
public:
    LinePathGenerator(const LineParams& parameters) : params_(parameters) {}
    virtual nav_msgs::Path GeneratePath(const geometry_msgs::Pose& start_pose) override;

private:
    LineParams params_;
};

} // namespace trace_path

#endif