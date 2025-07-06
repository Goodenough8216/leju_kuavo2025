#ifndef HUMANOID_PLAN_ARM_TRAJECTORY_H
#define HUMANOID_PLAN_ARM_TRAJECTORY_H
#pragma once

#include <ros/ros.h>
#include <sensor_msgs/JointState.h>
#include <trajectory_msgs/JointTrajectory.h>
#include <humanoid_plan_arm_trajectory/planArmState.h>
#include <std_srvs/Trigger.h>

namespace ocs2 {
  namespace humanoid {
    class HumanoidPlanArmTrajectory {
      public:
        HumanoidPlanArmTrajectory(const std::string& name = "HumanoidPlanArmTrajectory", int joint_num = 0, const std::string& interpolate_type = "");
        virtual ~HumanoidPlanArmTrajectory() = default;
        void timerCallback(const ros::TimerEvent& event);
        virtual void initialize(ros::NodeHandle& nh, ros::NodeHandle& private_nh) = 0;
        bool stopPlanArmTrajectoryCallback(std_srvs::Trigger::Request &req, std_srvs::Trigger::Response &res);
        inline void stop() { is_stopped_ = true; stop_step_ = step_; }
        void updateTrajectoryPoint(const std::vector<double>& positions, 
                                   const std::vector<double>& velocities, 
                                   const std::vector<double>& accelerations);
        void updateTrajectoryState(const int progress, const bool is_finished);
        void updateJointState(const std::vector<double>& positions, const std::vector<double>& velocities);
        static std::string current_interpolate_type_;

      protected:
        virtual void interpolate() = 0; // interpolate the trajectory
        virtual void update() = 0; // update the joint state from the trajectory
        virtual void reset() = 0; // reset the trajectory
        virtual void initializeSpecific() = 0;
        void initializeCommon();
        
        std::vector<std::string> joint_names_;
        std::string name_;
        std::string interpolate_type_;
        std::string joint_state_topic_;
        std::string joint_state_unit_;
        int joint_num_ = 0;
        double step_ = 0.0;
        double stop_step_ = 0.0;
        constexpr static double dt_ = 1.0e-3; 
        constexpr static int rate_ = 1000; // default 1000 Hz
        bool is_finished_ = false;
        bool is_stopped_ = false;

        ros::NodeHandle* nh_;
        ros::NodeHandle* private_nh_;
        ros::Publisher arm_traj_pub_;
        ros::Publisher arm_traj_state_pub_;
        ros::Publisher joint_state_pub_;
        ros::ServiceServer plan_arm_traj_srv_; // need to initialize in the inherited class
        ros::ServiceServer stop_arm_traj_srv_; 
        ros::Timer timer_;
        
        bool interpolate_finished_ = false;
        sensor_msgs::JointState joint_state_;
        trajectory_msgs::JointTrajectory traj_;
        humanoid_plan_arm_trajectory::planArmState arm_traj_state_;
    };
  }
}

#endif // HUMANOID_PLAN_ARM_TRAJECTORY_H
