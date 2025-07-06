#include "bezier_curve_interpolator.h"
#include <algorithm>

namespace ocs2 {
namespace humanoid {

BezierCurve::BezierCurve(double start_time, double end_time, 
            drake::trajectories::BezierCurve<double> pos_traj,
            std::unique_ptr<drake::trajectories::Trajectory<double>> vel_traj,
            std::unique_ptr<drake::trajectories::Trajectory<double>> acc_traj)
    : start_time(start_time), end_time(end_time), pos_traj(std::move(pos_traj)),
      vel_traj(std::move(vel_traj)), acc_traj(std::move(acc_traj)) {}

BezierCurveInterpolator::BezierCurveInterpolator(const std::string& name, int joint_num, const std::string& interpolate_type)
    : HumanoidPlanArmTrajectory(name, joint_num, interpolate_type) {
}

void BezierCurveInterpolator::initialize(ros::NodeHandle& nh, ros::NodeHandle& private_nh) {
    nh_ = &nh;
    private_nh_ = &private_nh;

    initializeCommon();
    initializeSpecific();
}

void BezierCurveInterpolator::reset() {
    step_ = 0.0;
    is_stopped_ = false;
    is_finished_ = false;
    stop_step_ = 0.0;
    total_time_ = 0.0;
    interpolate_finished_ = false;
    bezier_curves_.clear();
    traj_.points.clear();
    joint_names_.clear();
}

void BezierCurveInterpolator::initializeSpecific() {
    std::cout << "interpolate_type_: " << interpolate_type_ << std::endl;
    
    plan_arm_traj_srv_ = nh_->advertiseService(interpolate_type_ + "/plan_arm_trajectory", 
                                              &BezierCurveInterpolator::planArmTrajectoryBezierCurveCallback, this);
}

void BezierCurveInterpolator::update() {
    if (!interpolate_finished_ || bezier_curves_.empty() || joint_num_ == 0) {
        return;
    }
    step_ += dt_;
    double current_step = is_stopped_ ? stop_step_ : step_;
    ROS_DEBUG("[BezierCurveInterpolator] Trajectory progress: %f / %f", current_step, total_time_);

    std::vector<double> positions(joint_num_, 0.0);
    std::vector<double> velocities(joint_num_, 0.0);
    std::vector<double> accelerations(joint_num_, 0.0);

    if (!bezier_curves_.empty()) {
        for (size_t i = 0; i < bezier_curves_.size(); ++i) {
            evaluate(i, bezier_curves_[i], current_step, positions, velocities, accelerations);
        }
    }
    
    is_finished_ = current_step >= total_time_;
    int progress = static_cast<int>(current_step * 1000);  // ms
    updateTrajectoryPoint(positions, velocities, accelerations);
    // updateJointState(positions, velocities);
    updateTrajectoryState(progress, is_finished_);
}

void BezierCurveInterpolator::evaluate(const int index, const std::list<BezierCurve>& curve_list, double current_step, 
                                       std::vector<double>& positions, std::vector<double>& velocities, std::vector<double>& accelerations) {
    auto it = std::find_if(curve_list.begin(), curve_list.end(),
                           [current_step](const BezierCurve& curve) { return current_step <= curve.end_time; });

    if (it == curve_list.end()) {
        it = std::prev(curve_list.end());
    }

    const auto& curve = *it;
    double t = std::clamp((std::min(current_step, total_time_) - curve.start_time) / (curve.end_time - curve.start_time), 0.0, 1.0);

    Eigen::Vector2d pos = curve.pos_traj.value(t);
    Eigen::Vector2d vel = curve.vel_traj->value(t);
    Eigen::Vector2d acc = curve.acc_traj->value(t);

    positions[index] = pos.y();
    velocities[index] = vel.y();
    accelerations[index] = acc.y();
}

void BezierCurveInterpolator::interpolate() {
    bezier_curves_.clear();
    bezier_curves_.reserve(control_points_.size());
    joint_num_ = control_points_.size();
    for (size_t i = 0; i < control_points_.size(); i++) {
        auto curve_list = std::list<BezierCurve>();
        for (size_t j = 1; j < control_points_[i].size(); j++) {
            createSingleBezierCurve(i, j, curve_list);
        }
        bezier_curves_.emplace_back(std::move(curve_list));
    }
    interpolate_finished_ = true;
}

void BezierCurveInterpolator::createSingleBezierCurve(size_t i, size_t j, std::list<BezierCurve>& curve_list) {
    double start_time = control_points_[i][j-1][0][0];
    double end_time = control_points_[i][j][0][0];

    Eigen::Matrix<double, 2, 4> control_points_matrix;
    control_points_matrix.col(0) = control_points_[i][j-1][0];
    control_points_matrix.col(1) = control_points_[i][j-1][2];
    control_points_matrix.col(2) = control_points_[i][j][1];
    control_points_matrix.col(3) = control_points_[i][j][0];
    drake::trajectories::BezierCurve<double> pos_traj(0, 1, control_points_matrix);
    auto vel_traj = pos_traj.MakeDerivative();
    auto acc_traj = pos_traj.MakeDerivative(2);

    curve_list.emplace_back(start_time, end_time, std::move(pos_traj), std::move(vel_traj), std::move(acc_traj));

}

bool BezierCurveInterpolator::planArmTrajectoryBezierCurveCallback(humanoid_plan_arm_trajectory::planArmTrajectoryBezierCurve::Request& req, 
                                                                    humanoid_plan_arm_trajectory::planArmTrajectoryBezierCurve::Response& res) {
    
    std::cout << "BezierCurveInterpolator::planArmTrajectoryBezierCurveCallback" << std::endl;
    current_interpolate_type_ = interpolate_type_;
    std::cout << "current_interpolate_type_: " << current_interpolate_type_ << std::endl;
    reset();

    total_time_ = req.end_frame_time - req.start_frame_time;
    ControlPointsType control_points;
    for (const auto& frames : req.multi_joint_bezier_trajectory) {
      auto curve_points = frames.bezier_curve_points;
      std::vector<std::vector<Eigen::VectorXd>> joint_control_points;

      for (size_t i = 0; i < curve_points.size(); ++i) {
          const auto& curve_point = curve_points[i];
          double curve_start_time = curve_point.end_point[0];
          double curve_end_time = (i + 1 < curve_points.size()) ? curve_points[i+1].end_point[0] : curve_start_time;
          
          std::vector<Eigen::VectorXd> frame_points;
          Eigen::VectorXd end_point(2);
          Eigen::VectorXd left_control_point(2);
          Eigen::VectorXd right_control_point(2);
          end_point << curve_point.end_point[0], curve_point.end_point[1];
          left_control_point << curve_point.left_control_point[0], curve_point.left_control_point[1];
          right_control_point << curve_point.right_control_point[0], curve_point.right_control_point[1];
          frame_points.push_back(end_point);
          frame_points.push_back(left_control_point);
          frame_points.push_back(right_control_point);
          joint_control_points.push_back(frame_points);
      }
      control_points.push_back(joint_control_points);
    }
    control_points_ = control_points;
    joint_names_ = req.joint_names;
    interpolate();

    res.success = true;
    return true;
}

} // namespace humanoid
} // namespace ocs2
