#include <ros/ros.h>
#include <iostream>

#include "path_generator.h"
#include "line_path_generator.h"
#include "square_path_generator.h"
#include "circle_path_generator.h"
#include "s_curve_path_generator.h"

#include "path_tracer.h"
#include "pid_path_tracer.h"
#include "mpc_path_tracer.h"

using namespace trace_path;
int main (int argc, char **argv)
{
    ros::init (argc, argv, "trace_path_demo_node");
    ros::NodeHandle nh;
    
    // PID 
    {
        // 路径中点的间隙 = speed * dt  (单位：m)
        LineParams   line_params = {.length = 2.5, .speed = 0.2, .dt = 1.0};
        SquareParams square_params = {.side_length = 2.0, .speed = 0.2, .dt = 1.0}; 
        CircleParams circle_params = {.radius = 1.5, .speed = 0.25,.dt = 1.0};
        SCurveParams s_curve_params = {.length = 2.5, .amplitude = 1.15, .speed = 0.25, .dt = 1.0};

        // 路径生成器
        auto square_gen = PathGeneratorFactory::Create<SquarePathGenerator>(square_params);
        auto circle_gen = PathGeneratorFactory::Create<CirclePathGenerator>(circle_params);
        auto line_gen = PathGeneratorFactory::Create<LinePathGenerator>(line_params);
        auto s_curve_gen = PathGeneratorFactory::Create<SCurvePathGenerator>(s_curve_params);

        // 创建路径追踪器
        auto pid_path_tracer = PathTracerFactory::Create<PidPathTracer>(nh);

        pid_path_tracer->set_walk_velocity(0.3); // 跟随速度 0.3 m/s
        // pid_path_tracer->Follow(line_gen);     // 直线
        // pid_path_tracer->Follow(square_gen);   // 正方形
        // pid_path_tracer->set_walk_velocity(0.3); // 圆形跟随速度 0.3 m/s
        // pid_path_tracer->Follow(circle_gen);   // 圆形 
        // pid_path_tracer->Follow(s_curve_gen);  // "S" 曲线 
    }

    // MPC 
    {
        // 参考路径中, 点的间隙 = speed * dt  (单位：m)
        LineParams   line_params = {.length = 2.0, .speed = 0.25, .dt = 0.1};
        SquareParams square_params = {.side_length = 2.0, .speed = 0.25, .dt = 0.1}; 
        CircleParams circle_params = {.radius = 1.0, .speed = 0.25,.dt = 0.1};
        SCurveParams s_curve_params = {.length = 2.5, .amplitude = 1.25, .speed = 0.25, .dt = 0.1};
        // SCurveParams s_curve_params = {.length = 3.5, .amplitude = 1.25, .speed = 0.25, .dt = 0.1};

        // 创建路径生成器
        auto square_gen = PathGeneratorFactory::Create<SquarePathGenerator>(square_params);
        auto circle_gen = PathGeneratorFactory::Create<CirclePathGenerator>(circle_params);
        auto line_gen = PathGeneratorFactory::Create<LinePathGenerator>(line_params);
        auto s_curve_gen = PathGeneratorFactory::Create<SCurvePathGenerator>(s_curve_params);

        auto mpc_path_tracer = PathTracerFactory::Create<MpcPathTracer>(nh);
        mpc_path_tracer->set_max_linear_velocity(0.20); // 降低速度 0.2 m/s
        mpc_path_tracer->set_max_angular_velocity(M_PI/8);

        /* 直线 */ 
        mpc_path_tracer->Follow(line_gen);     // 直线
        
        /* 正方形 */
        // mpc_path_tracer->Follow(square_gen);   // 正方形

        /* 圆形 */
        // mpc_path_tracer->Follow(circle_gen); // 圆形 
        
        /* "S" 曲线  */
        // mpc_path_tracer->Follow(s_curve_gen); // "S" 曲线 
    }


    return 0;
}