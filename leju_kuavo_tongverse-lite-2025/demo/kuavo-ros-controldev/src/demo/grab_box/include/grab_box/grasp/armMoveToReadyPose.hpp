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
#include "kuavo_msgs/jointCmd.h"

namespace GrabBox
{
  typedef std::pair<Eigen::Vector3d, Eigen::Quaterniond> HandPose;
  typedef std::vector<std::pair<HandPose, HandPose>> TwoHandPoseTrajectory;

  using namespace std::chrono;
  // Example of Asynchronous node that uses StatefulActionNode as base class
  class ArmMoveToReadyPose : public BT::StatefulActionNode
  {
  public:
    ArmMoveToReadyPose(const std::string& name, const BT::NodeConfiguration& config)
    : BT::StatefulActionNode(name, config)
    {
      // ros
      ros::NodeHandle nh;
      pubArmTraj_ = nh.advertise<sensor_msgs::JointState>("/kuavo_arm_traj", 10);
      joint_sub_ = nh.subscribe<kuavo_msgs::jointCmd>("/joint_cmd", 10, &ArmMoveToReadyPose::jointCmdCallback, this);
      while (!nh.hasParam("/legRealDof") || !nh.hasParam("/armRealDof"))
      {
        sleep(1);
      }
      ros::param::get("/armRealDof", num_arm_joints_);
      ros::param::get("/legRealDof", num_leg_joints_);

    }
    void jointCmdCallback(const kuavo_msgs::jointCmd::ConstPtr &msg)
    {
      // std::cout << "Received joint command" << std::endl;
      Eigen::VectorXd joint_q(msg->joint_q.size());
      for (size_t i = 0; i < msg->joint_q.size(); i++)
      {
          joint_q[i] = msg->joint_q[i];
      }
      config().blackboard->set<Eigen::VectorXd>("command_joint_positions", joint_q);
      update_command_joints_ = true;
    }
    
    static BT::PortsList providedPorts()
    {
      return { 
        BT::InputPort<Eigen::Vector3d>("box_pos"), 
        BT::InputPort<Eigen::Vector4d>("box_quat"), 
        BT::InputPort<Eigen::Vector3d>("box_size"),  
        BT::InputPort<std::string>("action_name"),
        BT::InputPort<double>("move_speed")
      };
    }

    BT::NodeStatus onStart() override
    {


      // Get the action name from the input port (use default if not provided)
      std::string action_name;
      if (!getInput<std::string>("action_name", action_name)) {
        ROS_WARN("No action_name provided, using default: ready_pose");
        action_name = "ready_joints"; // Default action name
      }
      ready_joints_ = getParamsFromBlackboard<std::vector<double>>(config(), action_name);

      // Get move speed from the input port (use default if not provided)
      double move_speed;
      if (!getInput<double>("move_speed", move_speed)) {
        ROS_WARN("No move_speed provided, using default: 0.5");
        move_speed = 0.5; // Default move speed
      }
      while (!update_command_joints_)
      {
        ros::spinOnce();
        ros::Duration(0.01).sleep();
      }
      
      changeArmCtrlModeSrv(2);//using external controller

      is_running_ = true;
      // Get current joint positions from the blackboard
      Eigen::VectorXd command_joint_positions;
      if (!config().blackboard->get("command_joint_positions", command_joint_positions)) {
        ROS_ERROR("Cannot get command_joint_positions from blackboard");
        return BT::NodeStatus::FAILURE;
      }
      auto arms_joints_command = command_joint_positions.segment(num_leg_joints_, num_arm_joints_);

      // Convert Eigen::VectorXd to std::vector<double>
      std::vector<double> start_joints(arms_joints_command.data(), arms_joints_command.data() + arms_joints_command.size());

      // Generate interpolated trajectory
      const double dt = 0.01; // Time step for interpolation
      std::vector<std::vector<double>> trajectory = generateInterpolatedTrajectory(start_joints, ready_joints_, move_speed, dt);

      // Publish interpolated trajectory
      for (const auto& joints : trajectory) {
        pubArmTraj_.publish(getJointStatesMsg(joints));
        ros::Duration(dt).sleep();
      }

      return BT::NodeStatus::SUCCESS;
    }

    BT::NodeStatus onRunning() override
    {
      return BT::NodeStatus::SUCCESS;
    }

    void onHalted() override
    {
      ROS_WARN("Arm Ready Pose Movement interrupted");
      is_running_ = false;
    }

  private:
    std::vector<std::vector<double>> generateInterpolatedTrajectory(
      const std::vector<double>& start_joints,
      const std::vector<double>& target_joints,
      double move_speed,
      double dt)
    {
      std::vector<std::vector<double>> trajectory;

      // Calculate the maximum joint displacement
      double max_displacement = 0.0;
      for (size_t i = 0; i < start_joints.size(); ++i) {
        double displacement = std::abs(target_joints[i] - start_joints[i]);
        if (displacement > max_displacement) {
          max_displacement = displacement;
        }
      }

      // Calculate the total time required for the movement
      double total_time = max_displacement / move_speed;
      int num_steps = static_cast<int>(total_time / dt);
      ROS_INFO_STREAM("Total time: " << total_time << " num_steps: " << num_steps);

      // Generate interpolated trajectory
      for (int i = 0; i <= num_steps; ++i) {
        double alpha = static_cast<double>(i) / num_steps;
        std::vector<double> interpolated_joints;
        for (size_t j = 0; j < start_joints.size(); ++j) {
          interpolated_joints.push_back(
            (1.0 - alpha) * start_joints[j] + alpha * target_joints[j]
          );
        }
        trajectory.push_back(interpolated_joints);
      }

      return trajectory;
    }

  private:
    ros::Publisher pubArmTraj_;
    ros::Subscriber joint_sub_;
    bool is_running_ = false;
    bool update_command_joints_ = false;
    std::vector<double> ready_joints_;
    int num_arm_joints_, num_leg_joints_;
  };
} // namespace GrabBox
