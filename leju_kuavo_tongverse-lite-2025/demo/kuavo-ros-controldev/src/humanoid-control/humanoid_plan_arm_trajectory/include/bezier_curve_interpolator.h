#ifndef BEZIER_CURVE_INTERPOLATOR_H
#define BEZIER_CURVE_INTERPOLATOR_H
#pragma once

#include "humanoid_plan_arm_trajectory.h"
#include <drake/common/trajectories/trajectory.h>
#include <drake/common/trajectories/bezier_curve.h>
#include <memory>
#include <list>
#include <algorithm>

// msg
#include <humanoid_plan_arm_trajectory/bezierCurveCubicPoint.h>
#include <humanoid_plan_arm_trajectory/jointBezierTrajectory.h>

// srv
#include <humanoid_plan_arm_trajectory/planArmTrajectoryBezierCurve.h>

namespace ocs2 {
  namespace humanoid {
    using ControlPointsType = std::vector<std::vector<std::vector<Eigen::VectorXd>>>;

    class BezierCurve {
      public:
          BezierCurve(double start_time, double end_time, 
                      drake::trajectories::BezierCurve<double> pos_traj,
                      std::unique_ptr<drake::trajectories::Trajectory<double>> vel_traj,
                      std::unique_ptr<drake::trajectories::Trajectory<double>> acc_traj);

          // Use default move constructor and move assignment operator
          BezierCurve(BezierCurve&&) = default;
          BezierCurve& operator=(BezierCurve&&) = default;

          // Delete copy constructor and copy assignment operator
          BezierCurve(const BezierCurve&) = delete;
          BezierCurve& operator=(const BezierCurve&) = delete;

          double start_time;
          double end_time;
          drake::trajectories::BezierCurve<double> pos_traj;
          std::unique_ptr<drake::trajectories::Trajectory<double>> vel_traj;
          std::unique_ptr<drake::trajectories::Trajectory<double>> acc_traj;
    };
    
    class BezierCurveInterpolator : public HumanoidPlanArmTrajectory {
      public:
        BezierCurveInterpolator(const std::string& name = "BezierCurveInterpolator", int joint_num = 0, const std::string& interpolate_type = "bezier");
        ~BezierCurveInterpolator()=default;
        virtual void initialize(ros::NodeHandle& nh, ros::NodeHandle& private_nh);
      private:
        void initializeSpecific() override;
        void interpolate() override;
        void update() override;
        void reset() override;

        void evaluate(const int index, const std::list<BezierCurve>& curve_list, double current_step, std::vector<double>& positions, std::vector<double>& velocities, std::vector<double>& accelerations);
        void createSingleBezierCurve(size_t i, size_t j, std::list<BezierCurve>& curve_list);
        bool planArmTrajectoryBezierCurveCallback(humanoid_plan_arm_trajectory::planArmTrajectoryBezierCurve::Request& req, 
                                                   humanoid_plan_arm_trajectory::planArmTrajectoryBezierCurve::Response& res);

        double total_time_ = 0.0;
        ControlPointsType control_points_;
        std::vector<std::list<BezierCurve>> bezier_curves_; 
    };
  }
}

#endif // BEZIER_CURVE_INTERPOLATOR_H
