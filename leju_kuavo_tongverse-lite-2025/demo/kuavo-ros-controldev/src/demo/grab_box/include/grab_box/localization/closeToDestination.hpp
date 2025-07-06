#include <ros/ros.h>
#include <behaviortree_cpp_v3/bt_factory.h>
#include <grab_box/utils/customIoPort.h>
#include <ocs2_mpc/SystemObservation.h>
#include <ocs2_msgs/mpc_observation.h>
#include <ocs2_ros_interfaces/common/RosMsgConversions.h>
#include <ocs2_robotic_tools/common/RotationTransforms.h>
#include "grab_box/common/ocs2_ros_interface.hpp"

namespace GrabBox
{
  class CloseToDestination : public BT::ConditionNode
  {
  public:
    CloseToDestination(const std::string &name, const BT::NodeConfiguration &config) 
      : BT::ConditionNode(name, config)
    {
      // ros::NodeHandle nh = ros::NodeHandle("~");
      // subObservation_ = nh.subscribe("/humanoid_mpc_observation", 10, &CloseToDestination::observationCallback, this);
    }

    static BT::PortsList providedPorts()
    {
      return {BT::InputPort<double>("pos_error_threshold"), BT::InputPort<double>("yaw_error_threshold"),
        BT::InputPort<Eigen::Vector4d>("destination"), BT::OutputPort<Eigen::Vector4d>("pose_delta")}; // 提供输入端口
    }

    BT::NodeStatus tick() override final
    {
      Eigen::Vector4d destination = Eigen::Vector4d::Zero();
      double pos_error_threshold, yaw_error_threshold;
      getInput("destination", destination);
      getInput("pos_error_threshold", pos_error_threshold);
      getInput("yaw_error_threshold", yaw_error_threshold);
      std::cout << "pos_error_threshold: " << pos_error_threshold << " m, yaw_error_threshold: " << yaw_error_threshold << " deg." << std::endl;

      Eigen::Vector4d currentPose;// x, y, z, yaw
      auto res = config().blackboard->get<Eigen::Vector4d>("torso_pose", currentPose);
      if (res)
      {
        // Eigen::Vector4d currentPose = latestObservation_.state.segment<4>(6);// x, y, z, yaw
        Eigen::Matrix3d R_ws = ocs2::getRotationMatrixFromZyxEulerAngles(Eigen::Vector3d(currentPose(3), 0, 0));
        currentPose(3) *= 180.0 / M_PI; // convert to degrees
        Eigen::Vector4d poseDelta = destination - currentPose;
        poseDelta(2) = 0;
        poseDelta(3) = normalizedYaw(poseDelta(3));
        poseDelta.head<3>() = R_ws.transpose() * poseDelta.head<3>();
        setOutput("pose_delta", poseDelta);
        ROS_INFO_STREAM("pose_delta: " << poseDelta.transpose());
        if (poseDelta.head<3>().norm() < pos_error_threshold && std::abs(poseDelta(3)) < yaw_error_threshold)
        {
          return BT::NodeStatus::SUCCESS;
        }
        // updated_ = false;
      }
      return BT::NodeStatus::FAILURE;
    }
  private:
    void observationCallback(const ocs2_msgs::mpc_observation::ConstPtr &msg)
    {
      latestObservation_ = ocs2::ros_msg_conversions::readObservationMsg(*msg);
      // updated_ = true;
    }
  private:
    ros::Subscriber subObservation_;
    ocs2::SystemObservation latestObservation_;
    // bool updated_ = false;
  };
} // namespace GrabBox