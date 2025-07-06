#ifndef S_CURVE_PATH_GENERATOR_H
#define S_CURVE_PATH_GENERATOR_H
#include "path_generator.h"

namespace trace_path {

/// @brief 用于生成"S"形曲线路径所需的参数
struct SCurveParams : public BaseParams
{
    double length;    // 长度
    double amplitude; // 振幅

    SCurveParams(): BaseParams(), length(0.0), amplitude(0.0) {}
    SCurveParams(double length_, double amplitude_)
        : BaseParams(), length(length_), amplitude(amplitude_) {}
    SCurveParams(double length_, double amplitude_, double speed_, double dt_)
        : BaseParams(speed_, dt_), length(length_), amplitude(amplitude_) {}
};

/// @brief "S"形曲线路径生成器
class SCurvePathGenerator : public PathGeneratorBase {
public:
    SCurvePathGenerator(const SCurveParams& parameters) : params_(parameters) {}
    virtual nav_msgs::Path GeneratePath(const geometry_msgs::Pose& start_pose) override;

private:
    SCurveParams params_;    
};
} // namespace trace_path

#endif