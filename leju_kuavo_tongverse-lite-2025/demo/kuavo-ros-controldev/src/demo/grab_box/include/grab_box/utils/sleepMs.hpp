#include <ros/ros.h>
#include <behaviortree_cpp_v3/bt_factory.h>
#include <ocs2_mpc/SystemObservation.h>
#include <ocs2_msgs/mpc_observation.h>
#include <ocs2_ros_interfaces/common/RosMsgConversions.h>
#include <std_srvs/SetBool.h> 
#include "grab_box/common/ocs2_ros_interface.hpp"
#include "kuavo_msgs/endEffectorData.h"

namespace GrabBox
{
  class SleepMs : public BT::ConditionNode
  {
  public:
    SleepMs(const std::string &name, const BT::NodeConfiguration &config) 
      : BT::ConditionNode(name, config)
    {
    }

    static BT::PortsList providedPorts()
    {
      return {BT::InputPort<int>("sleep_ms")}; // 提供输入端口
    }

    BT::NodeStatus tick() override final
    {
      int sleep_ms = 0;
      getInput<int>("sleep_ms", sleep_ms);
      ROS_INFO_STREAM("Sleeping for " << sleep_ms << " milliseconds.");
      ros::Duration(sleep_ms/1000.0).sleep();
      ROS_INFO_STREAM("Done sleeping.");
      return BT::NodeStatus::SUCCESS;
    }
  };
} // namespace GrabBox
