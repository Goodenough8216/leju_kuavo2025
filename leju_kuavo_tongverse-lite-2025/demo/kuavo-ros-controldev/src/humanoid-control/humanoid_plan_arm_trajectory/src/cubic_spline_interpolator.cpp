#include "cubic_spline_interpolator.h"

namespace ocs2 {
namespace humanoid {

CubicSplineInterpolator::CubicSplineInterpolator(const std::string& name, int joint_num, const std::string& interpolate_type)
    : HumanoidPlanArmTrajectory(name, joint_num, interpolate_type) {
}

void CubicSplineInterpolator::initializeSpecific() {
  plan_arm_traj_srv_ = nh_->advertiseService(interpolate_type_ + "/plan_arm_trajectory", &CubicSplineInterpolator::planArmTrajectoryCubicSpline, this);
}

void CubicSplineInterpolator::initialize(ros::NodeHandle& nh, ros::NodeHandle& private_nh) {
  nh_ = &nh;
  private_nh_ = &private_nh;

  initializeCommon();
  initializeSpecific();
}

void CubicSplineInterpolator::interpolate() {
  std::vector<Eigen::MatrixXd> pos_samples;
  for (const auto& position : positions_) {
    pos_samples.push_back(Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 1>>(position.data(), position.size()));
  }
  pos_traj_ = drake::trajectories::PiecewisePolynomial<double>::CubicShapePreserving(times_, pos_samples, true);
  vel_traj_ = pos_traj_.derivative();
  acc_traj_ = vel_traj_.derivative();
  interpolate_finished_ = true;
}

void CubicSplineInterpolator::update() {
  if (!interpolate_finished_ || joint_num_ == 0) {
    return;
  }
  step_ += dt_;
  double current_step = is_stopped_ ? stop_step_ : step_;

  Eigen::VectorXd positions_vectorXd = pos_traj_.value(current_step);
  Eigen::VectorXd velocities_vectorXd = vel_traj_.value(current_step);
  Eigen::VectorXd accelerations_vectorXd = acc_traj_.value(current_step);

  std::vector<double> positions(positions_vectorXd.data(), positions_vectorXd.data() + positions_vectorXd.size());
  std::vector<double> velocities(velocities_vectorXd.data(), velocities_vectorXd.data() + velocities_vectorXd.size());
  std::vector<double> accelerations(accelerations_vectorXd.data(), accelerations_vectorXd.data() + accelerations_vectorXd.size());

  is_finished_ = current_step >= total_time_;
  int progress = static_cast<int>(current_step * 1000);  // ms
  updateTrajectoryPoint(positions, velocities, accelerations);
  // updateJointState(positions, velocities);
  updateTrajectoryState(progress, is_finished_);
}

void CubicSplineInterpolator::reset() {
  step_ = 0.0;
  is_stopped_ = false;
  is_finished_ = false;
  stop_step_ = 0.0;
  total_time_ = 0.0;
  interpolate_finished_ = false;
  times_.clear();
  positions_.clear();
  velocities_.clear();
  accelerations_.clear();
  joint_names_.clear();
}

bool CubicSplineInterpolator::planArmTrajectoryCubicSpline(humanoid_plan_arm_trajectory::planArmTrajectoryCubicSpline::Request& req, 
                                                humanoid_plan_arm_trajectory::planArmTrajectoryCubicSpline::Response& res) {
  current_interpolate_type_ = interpolate_type_;
  reset();
  
  if (req.joint_trajectory.points.empty()) {
    ROS_ERROR("Received empty trajectory");
    res.success = false;
    return false;
  }

  size_t expected_size = req.joint_trajectory.points[0].positions.size();
  
  times_.reserve(req.joint_trajectory.points.size());
  positions_.reserve(req.joint_trajectory.points.size());
  
  for (const auto& point : req.joint_trajectory.points) {
    if (point.positions.size() != expected_size) {
      ROS_ERROR_STREAM("Inconsistent number of joints. Expected " << expected_size << ", got " << point.positions.size());
      res.success = false;
      return false;
    }
    times_.push_back(point.time_from_start.toSec());
    positions_.push_back(point.positions);
  }

  auto checkTimeIncreasing = [](const std::vector<double>& times) {
    for (size_t i = 1; i < times.size(); ++i) {
      if (times[i] <= times[i - 1]) {
        return false;
      }
    }
    return true;
  };
  if (!checkTimeIncreasing(times_)) {
    ROS_ERROR("Time must be increasing");
    res.success = false;
    return false;
  }
  joint_names_ = req.joint_trajectory.joint_names;
  joint_num_ = expected_size;
  total_time_ = times_.back();
  interpolate();
  res.success = true;
  return true;
}

} // namespace humanoid
} // namespace ocs2
