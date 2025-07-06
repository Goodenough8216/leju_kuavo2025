#include <ros/ros.h>
#include <behaviortree_cpp_v3/bt_factory.h>


namespace GrabBox
{
  using namespace std::chrono;

  // Example of Asynchronous node that uses StatefulActionNode as base class
  class NavToStart : public BT::StatefulActionNode
  {
  public:
    NavToStart(const std::string& name, const BT::NodeConfiguration& config)
    : BT::StatefulActionNode(name, config)
    {}

    static BT::PortsList providedPorts()
    {
      // amount of milliseconds that we want to sleep
      return{ BT::InputPort<int>("max_duration") };
    }

    BT::NodeStatus onStart() override
    {
      int max_duration = 0;
      getInput("max_duration", max_duration);

      if( max_duration <= 0 ) {
        // No need to go into the RUNNING state
        return BT::NodeStatus::SUCCESS;
    }
    else {
      // once the deadline is reached, we will return SUCCESS.
      deadline_ = system_clock::now() + milliseconds(1000*max_duration);
      return BT::NodeStatus::RUNNING;
    }
    }

    /// method invoked by an action in the RUNNING state.
    BT::NodeStatus onRunning() override
    {
      if ( system_clock::now() >= deadline_ ) {
        std::cout << "NavToStart reached deadline" << std::endl;
        return BT::NodeStatus::SUCCESS;
      }
      else {
        return BT::NodeStatus::RUNNING;
      }
    }

    void onHalted() override
    {
      // nothing to do here...
      std::cout << "NavToStart interrupted" << std::endl;
    }

  private:
    system_clock::time_point deadline_;
  };
} // namespace GrabBox