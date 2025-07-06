#include <Eigen/Dense>

#pragma once

namespace autoHeadChase {

  class PoseTransformer {
  public:
    // Converts a pose from a local frame to the world frame
    static Eigen::VectorXd transformPoseToWorld(const Eigen::VectorXd& pose_in_local, const Eigen::VectorXd& frame_pose_in_world);

    // Converts a pose from the world frame to a local frame
    static Eigen::VectorXd transformPoseToLocal(const Eigen::VectorXd& pose_in_world, const Eigen::VectorXd& frame_pose_in_world);

    static Eigen::Matrix4d eulerToTransformationMatrix(double roll, double pitch, double yaw);

    // Helper function to convert pose to a transformation matrix
    static Eigen::Matrix4d poseToTransform(const Eigen::VectorXd& pose);

    // Helper function to convert transformation matrix back to pose
    static Eigen::VectorXd transformToPose(const Eigen::Matrix4d& transform_matrix);
  };

}  // namespace autoHeadChase
