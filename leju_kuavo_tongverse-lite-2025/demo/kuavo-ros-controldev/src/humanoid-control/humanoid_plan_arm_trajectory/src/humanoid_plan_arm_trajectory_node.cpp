#include <ros/ros.h>
#include "bezier_curve_interpolator.h"
#include "cubic_spline_interpolator.h"
// Include other interpolator headers as needed

int main(int argc, char** argv)
{
    ros::init(argc, argv, "humanoid_plan_arm_trajectory");
    ros::NodeHandle nh;
    ros::NodeHandle private_nh("~");

    std::vector<std::unique_ptr<ocs2::humanoid::HumanoidPlanArmTrajectory>> interpolators;

    // Initialize all interpolators
    interpolators.push_back(std::make_unique<ocs2::humanoid::BezierCurveInterpolator>("BezierCurveInterpolator", 0, "bezier"));
    interpolators.push_back(std::make_unique<ocs2::humanoid::CubicSplineInterpolator>("CubicSplineInterpolator", 0, "cubic_spline"));
    // Add other interpolators here as needed

    // Initialize all interpolators
    for (auto& interpolator : interpolators) {
        interpolator->initialize(nh, private_nh);
    }

    ros::spin();

    return 0;
}
