#include <ros/ros.h>
#include <behaviortree_cpp_v3/bt_factory.h>
#include <grab_box/utils/singleStepControl.hpp>
#include <Eigen/Core>
#include <cmath>
#include <std_msgs/Bool.h>
#include <std_srvs/SetBool.h>
#include <grab_box/utils/customIoPort.h>
#include <ocs2_robotic_tools/common/RotationTransforms.h>
#include "grab_box/common/ocs2_ros_interface.hpp"
#include <kuavo_msgs/footPoseTargetTrajectoriesSrv.h>

namespace GrabBox
{
  enum StepType
  {
    FOUR_DOF = 0, // x, y, z, yaw
    ONE_DOF = 1,  // yaw
  };
  using namespace std::chrono;
  // Example of Asynchronous node that uses StatefulActionNode as base class
  class SingleStepToDestination : public BT::StatefulActionNode
  {
  public:
    SingleStepToDestination(const std::string& name, const BT::NodeConfiguration& config)
    : BT::StatefulActionNode(name, config)
    {
      ros::NodeHandle nh;
      pubFoot_ = nh.advertise<kuavo_msgs::footPoseTargetTrajectories>("/humanoid_mpc_foot_pose_target_trajectories", 10);
      // subIsCustomGait_ = nh.subscribe("/monitor/is_custom_gait", 10, &SingleStepToDestination::isCustomGaitCallback, this);
      get_current_gait_client_ = nh.serviceClient<std_srvs::SetBool>("humanoid_get_current_gait");
      torso_spd_threashold_ = getParamsFromBlackboard<double>(config, "singleStepToDestination.torso_spd_threashold");
      auto max_delta_pose_vec = getParamsFromBlackboard<vector<double>>(config, "singleStepToDestination.max_delta_pose");
      max_delta_pose_ = Eigen::Vector4d(max_delta_pose_vec[0], max_delta_pose_vec[1], max_delta_pose_vec[2], max_delta_pose_vec[3]);
      step_dt_ = getParamsFromBlackboard<double>(config, "singleStepToDestination.step_dt");
      pub_wait_time_ = getParamsFromBlackboard<double>(config, "singleStepToDestination.pub_wait_time");
      foot_bias_ = getParamsFromBlackboard<double>(config, "singleStepToDestination.foot_bias");
      foot_pose_target_trajectories_client_ = nh.serviceClient<kuavo_msgs::footPoseTargetTrajectoriesSrv>("/humanoid_mpc_foot_pose_target_trajectories_srv");
    }

    static BT::PortsList providedPorts()
    {
      // amount of milliseconds that we want to sleep
      return{ BT::InputPort<Eigen::Vector4d>("destination"), BT::InputPort<int>("step_type") };
    }

    BT::NodeStatus onStart() override
    {
      is_custom_gait_published_ = false;
      getInput("step_type", step_type_);
      // limitArmControlMode();
      std::cout << "[SingleStepToDestination] step_type = " << ((step_type_==StepType::FOUR_DOF) ? "FOUR_DOF" : "ONE_DOF") << std::endl;
      return BT::NodeStatus::RUNNING;
    }

    vector<Vector4d> interplate_poses(const Vector4d& destination)
    {
      vector<Vector4d> body_poses;
      // const Vector4d max_delta_pose{0.15, 0.05, 0.1, 30};
      int num_steps = 1;
      for (int i = 0; i < 3; i++) 
      {
        num_steps = std::max(num_steps, static_cast<int>(std::ceil(destination(i) / max_delta_pose_(i))));
      }
      int degree_steps = static_cast<int>(std::ceil(abs(destination(3) / max_delta_pose_(3))));
      num_steps = std::max(num_steps, degree_steps);
      num_steps = std::max(num_steps, static_cast<int>(std::ceil(destination.head(3).norm() / max_delta_pose_.head(3).norm())));
      std::cout << "[SingleStepToDestination] num_steps = " << num_steps << std::endl;
      for (int i = 0; i < num_steps; i++)
      {
        Vector4d pose = Vector4d::Zero();
        pose =  (i+1) / static_cast<double>(num_steps) * destination;
        body_poses.push_back(pose);
      }
      return body_poses;
    }

    /// method invoked by an action in the RUNNING state.
    BT::NodeStatus onRunning() override
    {
      Eigen::VectorXd ocs2_state;
      if(!config().blackboard->get("ocs2_state", ocs2_state))
      {
        ROS_ERROR("Failed to retrieve ocs2_state from blackboard");
        return BT::NodeStatus::FAILURE;
      }
      if(ros::param::has("exit_non_stance_mode"))
      {
        bool exit_non_stance_mode = false;
        ros::param::get("/exit_non_stance_mode", exit_non_stance_mode);
        if(exit_non_stance_mode)
        {
          ROS_WARN_STREAM("Exit non-stance mode, waiting for the robot to stop moving.");
          return BT::NodeStatus::RUNNING;
        }
      }
      if(ocs2_state.head(3).norm() > torso_spd_threashold_)
      {
        // ROS_WARN_STREAM("The spd of the robot is too high, waiting for the robot to stop moving. spd = " << ocs2_state.head(3).norm());
        return BT::NodeStatus::RUNNING;
      }
      if(!is_custom_gait_published_)
      {
        is_custom_gait_published_ = true;
        Eigen::Vector4d destination = Eigen::Vector4d::Zero();
        getInput("destination", destination);
        std::cout << "[SingleStepToDestination] destination = " << destination.transpose() << std::endl;
        if(step_type_ == StepType::ONE_DOF)
        {
          Eigen::Vector3d p_wd = destination.head(3) ;

          destination.head(3).setZero();
          destination(3) = atan2(p_wd.y(), p_wd.x()) * 180 / M_PI;//deg
          std::cout << "[SingleStepToDestination] step_type = ONE_DOF, destination = " << destination.transpose() << std::endl;
        }
        bool collision_check = true;
        // double dt = 0.8; // 迈一步的时间间隔
        vector<Vector4d> body_poses = interplate_poses(destination);
        kuavo_msgs::footPoseTargetTrajectoriesSrv srv;
        // auto msg = get_multiple_steps_msg(body_poses, step_dt_, collision_check);
        auto msg = get_multiple_steps_msg(body_poses, step_dt_, foot_bias_, collision_check);
        srv.request.foot_pose_target_trajectories = msg;
        int retry_count = 0;
        int max_retries = 5;
        bool success = false;
        if(getCurrentGait() && !is_custom_gait_)
        {
          while (retry_count < max_retries)
          {
            std::cout << "[SingleStepToDestination] publish MSG to foot pose target trajectories." << std::endl;
            // pubFoot_.publish(msg);
            if (foot_pose_target_trajectories_client_.call(srv)) {
                if (srv.response.success) {
                  success = true;
                  break;
                    ROS_INFO("Successfully received the foot pose trajectory.");
                } else {
                    ROS_WARN_STREAM("[SingleStepToDestination] Service request failed, retrying...");
                }
            } else {
                ROS_WARN_STREAM("[SingleStepToDestination] Service call failed, retrying...");
            }
            retry_count++;
            ros::Duration(0.2).sleep();
          }
          // pubFoot_.publish(msg);
          // if(!changeArmCtrlModeSrv(1))//return home first
          //   return BT::NodeStatus::FAILURE;
          // if(!changeArmCtrlModeSrv(0))//keep arm in home
          // // if(!changeArmCtrlModeSrv(2))//keep external arm control
          //   return BT::NodeStatus::FAILURE;
          ros::Duration(pub_wait_time_).sleep();
          return BT::NodeStatus::RUNNING;
        }
        else
        {
          std::cout << "[SingleStepToDestination] failed to publish foot pose target trajectories." << std::endl;
          return BT::NodeStatus::FAILURE;
        }
      }
      getCurrentGait();
      if (!is_custom_gait_) {
        std::cout << "[SingleStepToDestination] reached end." << std::endl;
        return BT::NodeStatus::SUCCESS;
      }
      else {
        // std::cout << "[SingleStepToDestination] still running." << std::endl;
        return BT::NodeStatus::RUNNING;
      }
    }

    void onHalted() override
    {
      // nothing to do here...
      std::cout << "NavToStart interrupted" << std::endl;
    }
  private:
    void isCustomGaitCallback(const std_msgs::Bool::ConstPtr& msg)
    {
      is_custom_gait_ = msg->data;
    }
    
    bool getCurrentGait()
    {
      std_srvs::SetBool srv;
      srv.request.data = true;
      if (get_current_gait_client_.call(srv)) 
      {
        is_custom_gait_ = srv.response.success;
        return true;
      }
      else
        ROS_ERROR("Failed to call service humanoid_get_current_gait");
      return false;
    }

  private:
    system_clock::time_point deadline_;
    ros::Publisher pubFoot_;
    ros::Subscriber subIsCustomGait_;
    ros::ServiceClient get_current_gait_client_;
    ros::ServiceClient foot_pose_target_trajectories_client_;

    bool is_custom_gait_ = false;
    bool is_custom_gait_published_ = false;
    int step_type_ = StepType::FOUR_DOF;
    double torso_spd_threashold_ = 0.02; // m/s
    double step_dt_ = 0.8; // 迈一步的时间间隔
    double pub_wait_time_ = 0.5; // 发布foot pose target trajectories后等待的时间
    Eigen::Vector4d max_delta_pose_;
    double foot_bias_ = 0.1;//双脚间距/2
  };
} // namespace GrabBox