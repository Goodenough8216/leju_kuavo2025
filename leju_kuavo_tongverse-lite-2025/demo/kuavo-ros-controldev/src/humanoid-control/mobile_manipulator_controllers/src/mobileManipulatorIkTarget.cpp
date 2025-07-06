#include <ros/ros.h>
#include <ocs2_mpc/SystemObservation.h>
#include <ocs2_ros_interfaces/command/TargetTrajectoriesRosPublisher.h>
#include "motion_capture_ik/twoArmHandPoseCmd.h"
#include <std_msgs/Float64MultiArray.h>

namespace mobile_manipulator_controller
{
using namespace ocs2;
struct IkCmd
{
    Eigen::Vector3d pos;     // hand pos
    Eigen::Quaterniond quat; // hand quaternion
};

struct BasePoseCmd
{
  int dim;
  Eigen::Matrix<double, 6, 1> pose;
};

class MobileManipulatorIkTarget {
  public:
    MobileManipulatorIkTarget(ros::NodeHandle& nodeHandle, const std::string& robotName)
    : nodeHandle_(nodeHandle)
    {
      targetTrajectoriesPublisherPtr_.reset(new TargetTrajectoriesRosPublisher(nodeHandle_, robotName));
      // observation subscriber
      auto observationCallback = [this](const ocs2_msgs::mpc_observation::ConstPtr& msg) {
        latestObservation_ = ros_msg_conversions::readObservationMsg(*msg);
        if(!observationReceived_)
          observationReceived_ = true;
      };
      observationSubscriber_ = nodeHandle_.subscribe<ocs2_msgs::mpc_observation>(robotName + "_mpc_observation", 1, observationCallback);
      ikCmdSubscriber_ = nodeHandle_.subscribe<motion_capture_ik::twoArmHandPoseCmd>("/ik/two_arm_hand_pose_cmd", 10, &MobileManipulatorIkTarget::ikCmdCallback, this);
      basePoseCmdSubscriber_ = nodeHandle_.subscribe<std_msgs::Float64MultiArray>(robotName + "/base_pose_cmd", 10, &MobileManipulatorIkTarget::basePoseCmdCallback, this);
    }

    void run()
    {
      ros::spin();
    }

  private:
    void basePoseCmdCallback(const std_msgs::Float64MultiArray::ConstPtr& msg)
    {
      newBasePoseReceived_ = true;
      const auto& pose = msg->data;
      if(pose.size() > 6)
      {
        ROS_ERROR_STREAM("Invalid base pose command size: " << pose.size());
        return;
      }
      basePoseCmd_.dim = pose.size();
      for(int i = 0; i < pose.size(); i++)
        basePoseCmd_.pose(i) = pose[i];
    }

    void ikCmdCallback(const motion_capture_ik::twoArmHandPoseCmd::ConstPtr& msg)
    {
      if(!observationReceived_)
      {
        ROS_WARN_STREAM("No observation received yet, cannot compute target trajectories");
        return;
      }
      auto &hand_poses = msg->hand_poses;
      IkCmd ik_cmd_l, ik_cmd_r;
      // left
      ik_cmd_l.pos << hand_poses.left_pose.pos_xyz[0], hand_poses.left_pose.pos_xyz[1], hand_poses.left_pose.pos_xyz[2];
      ik_cmd_l.quat = Eigen::Quaterniond(hand_poses.left_pose.quat_xyzw[3],
                              hand_poses.left_pose.quat_xyzw[0], hand_poses.left_pose.quat_xyzw[1], hand_poses.left_pose.quat_xyzw[2]);
      // right
      ik_cmd_r.pos << hand_poses.right_pose.pos_xyz[0], hand_poses.right_pose.pos_xyz[1], hand_poses.right_pose.pos_xyz[2];
      ik_cmd_r.quat = Eigen::Quaterniond(hand_poses.right_pose.quat_xyzw[3],
                              hand_poses.right_pose.quat_xyzw[0], hand_poses.right_pose.quat_xyzw[1], hand_poses.right_pose.quat_xyzw[2]);
      auto target = goalPoseToTargetTrajectories(ik_cmd_l, ik_cmd_r, latestObservation_);
      // const auto mpcTargetTrajectoriesMsg = ocs2::ros_msg_conversions::createTargetTrajectoriesMsg(targetTrajectories);
      targetTrajectoriesPublisherPtr_->publishTargetTrajectories(target);
    }

    TargetTrajectories goalPoseToTargetTrajectories(const IkCmd& cmd_l, const IkCmd& cmd_r, const SystemObservation& observation) {
      // time trajectory
      const scalar_array_t timeTrajectory{observation.time};
      // state trajectory: 3 + 4 for desired position vector and orientation quaternion
      auto l_pose = (vector_t(7) << cmd_l.pos, cmd_l.quat.coeffs()).finished(); 
      auto r_pose = (vector_t(7) << cmd_r.pos, cmd_r.quat.coeffs()).finished();

      vector_t target;
      if(newBasePoseReceived_)
      {
        target = (vector_t(14 + basePoseCmd_.dim) << basePoseCmd_.pose, l_pose, r_pose).finished();
        newBasePoseReceived_ = false;
      }
      else
      {
        target = (vector_t(14) << l_pose, r_pose).finished();        
      }
      const vector_array_t stateTrajectory{target};
      // input trajectory
      const vector_array_t inputTrajectory{vector_t::Zero(observation.input.size())};

      return {timeTrajectory, stateTrajectory, inputTrajectory};
    }

  private:
    ros::NodeHandle& nodeHandle_;
    std::unique_ptr<TargetTrajectoriesRosPublisher> targetTrajectoriesPublisherPtr_;
    ::ros::Subscriber observationSubscriber_;
    ::ros::Subscriber ikCmdSubscriber_;
    ::ros::Subscriber basePoseCmdSubscriber_;
    SystemObservation latestObservation_;
    bool observationReceived_ = false;
    bool newBasePoseReceived_ = false;
    BasePoseCmd basePoseCmd_;
};
}  // namespace mobile_manipulator_controller


int main(int argc, char* argv[]) {
  const std::string robotName = "mobile_manipulator";
  ::ros::init(argc, argv, robotName + "_target");
  ::ros::NodeHandle nodeHandle;

  mobile_manipulator_controller::MobileManipulatorIkTarget ikTarget(nodeHandle, robotName);
  ikTarget.run();
  // Successful exit
  return 0;
}
