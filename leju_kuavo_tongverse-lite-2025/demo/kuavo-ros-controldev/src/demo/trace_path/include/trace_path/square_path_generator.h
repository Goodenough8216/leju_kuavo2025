#ifndef SQUARE_PATH_GENERATOR_H
#define SQUARE_PATH_GENERATOR_H
#include "path_generator.h"

namespace trace_path {

///@brief 生成正方形路径所需的参数
struct SquareParams : public BaseParams
{
    double side_length; // 边长

    SquareParams(): BaseParams(), side_length(0.0) {}
    SquareParams(double side_length_)
        :BaseParams(), side_length(side_length_) {}
    SquareParams(double side_length_, double speed_, double dt_)
        :BaseParams(speed_, dt_), side_length(side_length_) {}
};

/// @brief  正方形轨迹路径生成器
class SquarePathGenerator : public PathGeneratorBase {
public:
    SquarePathGenerator(const SquareParams& parameters) : params_(parameters) {}
    virtual nav_msgs::Path GeneratePath(const geometry_msgs::Pose& start_pose) override;

private:
    SquareParams params_;  
};

} // namespace trace_path

#endif