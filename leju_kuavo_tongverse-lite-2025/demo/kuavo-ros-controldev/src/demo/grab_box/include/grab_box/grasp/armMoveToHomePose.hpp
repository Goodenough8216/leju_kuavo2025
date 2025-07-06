#include <ros/ros.h>
#include <behaviortree_cpp_v3/bt_factory.h>
#include <Eigen/Core>
#include <cmath>
#include <std_msgs/Bool.h>
#include <std_srvs/SetBool.h>
#include <grab_box/utils/customIoPort.h>
#include <sensor_msgs/JointState.h>
#include <visualization_msgs/Marker.h>
#include <ocs2_robotic_tools/common/RotationTransforms.h>
#include "motion_capture_ik/changeArmCtrlMode.h"
#include "motion_capture_ik/twoArmHandPoseCmdSrv.h"
#include "motion_capture_ik/ikSolveParam.h"
#include "kuavo_msgs/headBodyPose.h"
#include "kuavo_msgs/changeArmCtrlMode.h"
#include "grab_box/common/ocs2_ros_interface.hpp"

namespace GrabBox
{
  typedef std::pair<Eigen::Vector3d, Eigen::Quaterniond> HandPose;
  typedef std::vector<std::pair<HandPose, HandPose>> TwoHandPoseTrajectory;

  using namespace std::chrono;
  // Example of Asynchronous node that uses StatefulActionNode as base class
  class ArmMoveToHomePose : public BT::StatefulActionNode
  {
  public:
    ArmMoveToHomePose(const std::string& name, const BT::NodeConfiguration& config)
    : BT::StatefulActionNode(name, config)
    {

      // ros
      ros::NodeHandle nh;
      pubArmTraj_ = nh.advertise<sensor_msgs::JointState>("/kuavo_arm_traj", 10);
    }

    static BT::PortsList providedPorts()
    {
      return{ BT::InputPort<Eigen::Vector3d>("box_pos"), BT::InputPort<Eigen::Vector4d>("box_quat"), BT::InputPort<Eigen::Vector3d>("box_size")};
    }

    BT::NodeStatus onStart() override
    {
      changeArmCtrlModeSrv(2);//using external controller

      armMoveToReadyJoints();
      is_runing_ = true;
      // ros::Duration(1).sleep();
      // return BT::NodeStatus::RUNNING;
      return BT::NodeStatus::SUCCESS;
    }

    /// method invoked by an action in the RUNNING state.
    BT::NodeStatus onRunning() override
    {
      return BT::NodeStatus::SUCCESS;
    }

    void onHalted() override
    {
      // nothing to do here...
      std::cout << "Arm Ready Pose Movement interrupted" << std::endl;
      is_runing_ = false;
    }
  private:


    void armMoveToReadyJoints() {
        std::vector<double> joints(14,0.0);
        pubArmTraj_.publish(getJointStatesMsg(joints));
    }

    ros::Publisher pubArmTraj_;
    bool is_runing_ = false;
  };
} // namespace GrabBox