#include <pluginlib/class_list_macros.h>
#include <nodelet/nodelet.h>
#include "bezier_curve_interpolator.h"
#include "cubic_spline_interpolator.h"
// Include other interpolator headers as needed

namespace ocs2 {
namespace humanoid {

class HumanoidPlanArmTrajectoryNodelet : public nodelet::Nodelet
{
public:
    virtual void onInit()
    {
        ros::NodeHandle& nh = getNodeHandle();
        ros::NodeHandle& private_nh = getPrivateNodeHandle();

        // Initialize all interpolators
        interpolators_.push_back(std::make_unique<BezierCurveInterpolator>("BezierCurveInterpolator"));
        interpolators_.push_back(std::make_unique<CubicSplineInterpolator>("CubicSplineInterpolator"));
        // Add other interpolators here as needed

        // Initialize all interpolators
        for (auto& interpolator : interpolators_) {
            interpolator->initialize(nh, private_nh);
        }
    }

private:
    std::vector<std::unique_ptr<HumanoidPlanArmTrajectory>> interpolators_;
};

} // namespace humanoid
} // namespace ocs2

PLUGINLIB_EXPORT_CLASS(ocs2::humanoid::HumanoidPlanArmTrajectoryNodelet, nodelet::Nodelet)