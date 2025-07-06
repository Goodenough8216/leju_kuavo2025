#include <ros/ros.h>
#include <behaviortree_cpp_v3/bt_factory.h>
#include <behaviortree_cpp_v3/loggers/bt_cout_logger.h>
#include "grab_box/package_path.h"

// A simple action node that prints "Hello"
class SayHello : public BT::SyncActionNode
{
public:
    SayHello(const std::string &name)
        : BT::SyncActionNode(name, {})
    {
    }
    BT::NodeStatus tick() override final
    {
        std::cout << "Hello" << std::endl;
        return BT::NodeStatus::SUCCESS;
    }
};
class CheckValue : public BT::ConditionNode
{
public:
    CheckValue(const std::string &name, const BT::NodeConfiguration &config)
        : BT::ConditionNode(name, config)
    {
    }

    static BT::PortsList providedPorts()
    {
        return {BT::InputPort<int>("value_input")}; // 提供输入端口
    }

    BT::NodeStatus tick() override final
    {
        int value;
        if (!getInput<int>("value_input", value))
        {
            return BT::NodeStatus::FAILURE; // 输入值未提供
        }
        return (value > 256) ? BT::NodeStatus::SUCCESS : BT::NodeStatus::FAILURE; // 判断条件
    }
};

class Echo : public BT::SyncActionNode
{
public:
    Echo(const std::string &name, const BT::NodeConfiguration &config)
        : BT::SyncActionNode(name, config)
    {
    }
    static BT::PortsList providedPorts()
    {
        return {BT::InputPort<int>("echo_input")};
    }
    BT::NodeStatus tick() override final
    {
        int val = -1;
        auto res = getInput<int>("echo_input", val);
        std::cout << "Echo: " << val << std::endl;
        return BT::NodeStatus::SUCCESS;
    }
};

// A simple action node that prints "World"
class Square : public BT::SyncActionNode
{
public:
    Square(const std::string &name, const BT::NodeConfiguration &config)
        : BT::SyncActionNode(name, config)
    {
    }
    static BT::PortsList providedPorts()
    {
        return {BT::OutputPort<int>("square_output"), BT::InputPort<int>("square_input")};
    }
    // You must override the virtual function tick()
    BT::NodeStatus tick() override
    {
        int val = -1;
        auto res = getInput<int>("square_input", val);
        ROS_ASSERT_MSG(res, "%s", res.error().c_str());
        int output = val*val;
        ROS_INFO_STREAM("square_output: " << output);
        res = setOutput<int>("square_output", output);
        ROS_ASSERT_MSG(res, "%s", res.error().c_str());
        return BT::NodeStatus::SUCCESS;
    }
};

// Register the nodes to the factory

int main(int argc, char **argv)
{
    ros::init(argc, argv, "hello_BT");
    ros::NodeHandle nh("~");

    // Create a behavior tree factory and register the custom nodes
    BT::BehaviorTreeFactory factory;
    factory.registerNodeType<SayHello>("SayHello");
    factory.registerNodeType<Square>("Square");
    factory.registerNodeType<Echo>("Echo");
    factory.registerNodeType<CheckValue>("CheckValue");

    // Create a behavior tree from a string
    std::string path = GrabBox::getPath() + "/cfg/test/test1.xml";
    BT::Blackboard::Ptr blackboard = BT::Blackboard::create();
    auto tree = factory.createTreeFromFile(path, blackboard);
    // Create a logger to print the status of the nodes
    BT::StdCoutLogger logger(tree);
    ros::Rate rate(1);
    // Run the behavior tree until it returns SUCCESS or FAILURE
    while (ros::ok())
    {
        tree.tickRoot();
        rate.sleep();
    }

    return 0;
}
