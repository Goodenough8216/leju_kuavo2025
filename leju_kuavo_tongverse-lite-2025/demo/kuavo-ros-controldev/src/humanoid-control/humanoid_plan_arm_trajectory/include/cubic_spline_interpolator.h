#ifndef CUBIC_SPLINE_INTERPOLATOR_H
#define CUBIC_SPLINE_INTERPOLATOR_H

#include <vector>
#include <list>
#include <cmath>
#include <iostream>
#include <drake/common/trajectories/piecewise_polynomial.h>
#include "humanoid_plan_arm_trajectory.h"
#include "humanoid_plan_arm_trajectory/planArmTrajectoryCubicSpline.h"

namespace ocs2 {
namespace humanoid {
class CubicSplineInterpolator : public HumanoidPlanArmTrajectory {
      public:
        CubicSplineInterpolator(const std::string& name = "CubicSplineInterpolator", int joint_num = 0, const std::string& interpolate_type = "cubic_spline");
        ~CubicSplineInterpolator()=default;
        virtual void initialize(ros::NodeHandle& nh, ros::NodeHandle& private_nh);
      private:
        void initializeSpecific() override;
        void interpolate() override;
        void update() override;
        void reset() override;

        bool planArmTrajectoryCubicSpline(humanoid_plan_arm_trajectory::planArmTrajectoryCubicSpline::Request& req, 
                                                   humanoid_plan_arm_trajectory::planArmTrajectoryCubicSpline::Response& res);
        double total_time_ = 0.0;

        drake::trajectories::PiecewisePolynomial<double> pos_traj_;
        drake::trajectories::PiecewisePolynomial<double> vel_traj_;
        drake::trajectories::PiecewisePolynomial<double> acc_traj_;

        std::vector<double> times_;
        std::vector<std::vector<double>> positions_;
        std::vector<std::vector<double>> velocities_;
        std::vector<std::vector<double>> accelerations_;
    };
  }
}

#endif // CUBIC_SPLINE_INTERPOLATOR_H
