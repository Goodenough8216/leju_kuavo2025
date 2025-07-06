#include <ros/ros.h>
#include <behaviortree_cpp_v3/bt_factory.h>
#include <string>

namespace GrabBox
{
  class Echo : public BT::SyncActionNode
  {
  public:
    Echo(const std::string &name, const BT::NodeConfiguration &config)
        : BT::SyncActionNode(name, config)
    {
    }
    static BT::PortsList providedPorts()
    {
      return {BT::InputPort<std::string>("echo_input")};
    }
    BT::NodeStatus tick() override final
    {
      std::string val = "Noting to echo";
      auto res = getInput<std::string>("echo_input", val);
      std::cout << "Echo: " << val << std::endl;
      return BT::NodeStatus::SUCCESS;
    }
  };
} // namespace GrabBox