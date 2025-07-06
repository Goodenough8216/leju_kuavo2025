#include <ros/ros.h>
#include <behaviortree_cpp_v3/bt_factory.h>
#include <grab_box/utils/customIoPort.h>


namespace GrabBox
{
  using namespace std::chrono;

  // Example of Asynchronous node that uses StatefulActionNode as base class
  
  // A simple action node that prints "World"
  class MakePlan : public BT::SyncActionNode
  {
  public:
    MakePlan(const std::string &name, const BT::NodeConfiguration &config)
      : BT::SyncActionNode(name, config)
    {
    }
    static BT::PortsList providedPorts()
    {
      return {BT::OutputPort<torsoPoseTraj>("planed_trajectory"), BT::InputPort<torsoPose>("target_pose")};
    }
    // You must override the virtual function tick()
    BT::NodeStatus tick() override
    {
      torsoPose target_pose;
      auto res = getInput<torsoPose>("target_pose", target_pose);
      torsoPoseTraj torso_pose_traj;
      // TODO: Implement your logic here
      torso_pose_traj.push_back(target_pose);
      // ROS_INFO_STREAM("square_output: " << output);
      res = setOutput<torsoPoseTraj>("planed_trajectory", torso_pose_traj);
      // ROS_ASSERT_MSG(res, "%s", res.error().c_str());
      return BT::NodeStatus::SUCCESS;
    }
  };
} // namespace GrabBox