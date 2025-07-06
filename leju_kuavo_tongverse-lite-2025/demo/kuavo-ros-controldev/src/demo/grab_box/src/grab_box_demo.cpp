#include <ros/ros.h>
#include <behaviortree_cpp_v3/bt_factory.h>
#include <behaviortree_cpp_v3/loggers/bt_cout_logger.h>
#include <yaml-cpp/yaml.h>
#include "grab_box/package_path.h"
#include "grab_box/localization/closeToStart.hpp"
#include "grab_box/localization/checkStatusOK.hpp"
#include "grab_box/localization/closeToDestination.hpp"
#include "grab_box/localization/boxTagOK.hpp"
#include "grab_box/localization/computeTargetPose.hpp"
#include "grab_box/localization/computeBackTurnPoseFromBoxToTarget.hpp"
#include "grab_box/navigation/navToStart.hpp"
#include "grab_box/navigation/singleStepToDestination.hpp"
#include "grab_box/navigation/moveToDestination.hpp"
#include "grab_box/navigation/makePlan.hpp"
#include "grab_box/navigation/cmdPoseWorldMoveToDestination.hpp"
#include "grab_box/utils/echo.hpp"
#include "grab_box/grasp/graspBox.hpp"
#include "grab_box/grasp/armMoveToReadyPose.hpp"
#include "grab_box/grasp/armMoveToHomePose.hpp"
#include "grab_box/utils/forEachTag.hpp"
#include "grab_box/utils/sleepMs.hpp"
#include "humanoid_interface_drake/humanoid_interface_drake.h"
#include "grab_box/utils/forceCheck.hpp"
#include "grab_box/utils/timingDecorator.hpp"
#include "grab_box/navigation/cmdPoseMoveToDestination.hpp"
#include "grab_box/navigation/planAndMoveToDestination.hpp"

// #include "grab_box/utils/CSVLogger.hpp"
using namespace GrabBox;
int main(int argc, char **argv)
{
  ros::init(argc, argv, "grab_box_demo");
  ros::NodeHandle nh("~");

  // Create a behavior tree factory and register the custom nodes
  BT::BehaviorTreeFactory factory;
  factory.registerNodeType<CloseToStart>("CloseToStart");
  factory.registerNodeType<CheckStatusOK>("CheckStatusOK");
  factory.registerNodeType<NavToStart>("NavToStart");
  factory.registerNodeType<Echo>("Echo");
  factory.registerNodeType<SingleStepToDestination>("SingleStepToDestination");
  factory.registerNodeType<CloseToDestination>("CloseToDestination");
  factory.registerNodeType<MakePlan>("MakePlan");
  factory.registerNodeType<MoveToDestination>("MoveToDestination");
  factory.registerNodeType<PlanAndMoveToDestination>("PlanAndMoveToDestination");
  factory.registerNodeType<BoxTagOK>("BoxTagOK");
  factory.registerNodeType<GraspBox>("GraspBox");
  factory.registerNodeType<ComputeTargetPose>("ComputeTargetPose");
  factory.registerNodeType<ComputeBackTurnPoseFromBoxToTarget>("ComputeBackTurnPoseFromBoxToTarget");
  factory.registerNodeType<ArmMoveToReadyPose>("ArmMoveToReadyPose");
  factory.registerNodeType<ArmMoveToHomePose>("ArmMoveToHomePose");
  factory.registerNodeType<SleepMs>("SleepMs");
  factory.registerNodeType<ForEachTag>("ForEachTag");
  factory.registerNodeType<TimingDecorator>("TimingDecorator");
  factory.registerNodeType<ForceCheck>("ForceCheck");
  factory.registerNodeType<CmdPoseMoveToDestination>("CmdPoseMoveToDestination");
  factory.registerNodeType<CmdPoseWorldMoveToDestination>("CmdPoseWorldMoveToDestination");

  RobotVersion robot_version(4, 2);
  if (nh.hasParam("/robot_version"))
  {
    int robot_version_int;
    nh.getParam("/robot_version", robot_version_int);
    int major = robot_version_int / 10;
    int minor = robot_version_int % 10;
    robot_version = RobotVersion(major, minor);
  }
  auto humanoid_drake_interface = HighlyDynamic::HumanoidInterfaceDrake::getInstancePtr(robot_version, true, 2e-3);

  // Register the custom logger
  CSVLogger::getInstance().initialize("behavior_tree_log.csv");
  // Create a behavior tree from a string
  YAML::Node config = YAML::LoadFile(GrabBox::getPath() + "/cfg/bt_config.yaml");
  std::string bt_xml_string = config["bt_xml_file"].as<std::string>();
  std::string path = GrabBox::getPath() + "/cfg/" + bt_xml_string;
  std::cout << "Loading behavior tree from " << path << std::endl;
  // std::string path = GrabBox::getPath() + "/cfg/grab_box_demo.xml";
  BT::Blackboard::Ptr blackboard = BT::Blackboard::create();
  blackboard->set<YAML::Node>("config", config);
  blackboard->set<HighlyDynamic::HumanoidInterfaceDrake*>("humanoid_drake_interface", humanoid_drake_interface);

  auto tree = factory.createTreeFromFile(path, blackboard);
  // Create a logger to print the status of the nodes
  BT::StdCoutLogger logger(tree);
  ros::Rate rate(config["tick_rate"].as<int>());
  // Run the behavior tree until it returns SUCCESS or FAILURE
  while (ros::ok())
  {
    // Tick the behavior tree
    BT::NodeStatus status = tree.tickRoot();
    // Check the status of the behavior tree
    if (status == BT::NodeStatus::SUCCESS || status == BT::NodeStatus::FAILURE)
    {
      // If the behavior tree returns SUCCESS or FAILURE, break the loop
      ROS_INFO("Behavior tree finished with status: %s",
               status == BT::NodeStatus::SUCCESS? "SUCCESS" : "FAILURE");
      break;
    }
    rate.sleep();
    ros::spinOnce();
  }

  CSVLogger::getInstance().close();

  return 0;
}
