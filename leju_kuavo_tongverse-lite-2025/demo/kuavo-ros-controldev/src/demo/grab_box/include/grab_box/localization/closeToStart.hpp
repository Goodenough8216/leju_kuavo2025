#include <ros/ros.h>
#include <behaviortree_cpp_v3/bt_factory.h>

namespace GrabBox
{
  class CloseToStart : public BT::ConditionNode
  {
  public:
    CloseToStart(const std::string &name, const BT::NodeConfiguration &config) 
      : BT::ConditionNode(name, config)
    {
    }

    static BT::PortsList providedPorts()
    {
      return {BT::InputPort<int>("value_input")}; // 提供输入端口
    }

    BT::NodeStatus tick() override final
    {
      return BT::NodeStatus::FAILURE;
    }
  private:
    // ros::NodeHandle nh_;
  };
} // namespace GrabBox