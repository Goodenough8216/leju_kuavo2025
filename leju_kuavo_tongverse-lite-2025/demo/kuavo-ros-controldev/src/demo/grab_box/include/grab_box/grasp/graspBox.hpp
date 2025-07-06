#include <ros/ros.h>
#include <behaviortree_cpp_v3/bt_factory.h>
#include <Eigen/Core>
#include <cmath>
#include <std_msgs/Bool.h>
#include <std_srvs/SetBool.h>
#include <grab_box/utils/customIoPort.h>
#include <sensor_msgs/JointState.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <ocs2_robotic_tools/common/RotationTransforms.h>
#include "motion_capture_ik/changeArmCtrlMode.h"
#include "motion_capture_ik/twoArmHandPoseCmdSrv.h"
#include "motion_capture_ik/ikSolveParam.h"
#include "motion_capture_ik/fkSrv.h"
#include "kuavo_msgs/headBodyPose.h"
#include "kuavo_msgs/changeArmCtrlMode.h"
#include "grab_box/common/ocs2_ros_interface.hpp"
#include "grab_box/common/drake_interface.hpp"
#include "grab_box/common/math.hpp"
#include "grab_box/utils/cubic_interpolator.hpp"
#include "humanoid_interface_drake/humanoid_interface_drake.h"

namespace GrabBox
{
  enum GraspType
  {
    GraspUp = 0,
    PutDown = 1,
    Move = 2
  };

  using namespace std::chrono;
  // Example of Asynchronous node that uses StatefulActionNode as base class
  class GraspBox : public BT::StatefulActionNode
  {
  public:
    GraspBox(const std::string& name, const BT::NodeConfiguration& config)
    : BT::StatefulActionNode(name, config)
    {
      // 初始化参数
      // snopt params
      ikParam_.major_optimality_tol = 1e-3;
      ikParam_.major_feasibility_tol = 1e-3;
      ikParam_.minor_feasibility_tol = 1e-3;
      ikParam_.major_iterations_limit = 100;
      // constraint and cost params
      ikParam_.oritation_constraint_tol= 1e-3;
      ikParam_.pos_constraint_tol = 1e-3;// work when pos_cost_weight==0.0
      ikParam_.pos_cost_weight = 0.0;// If U need high accuracy, set this to 0.0 !!!

      // ikParam_.torso_ref_type = 0;
      // ikParam_.torso_weight = 10;
      // ikParam_.torso_ref_tol = {1e-3, 1e-3, 0.02, 0, 3*M_PI/180, 0};
      // ros
      ros::NodeHandle nh;
      pubArmTraj_ = nh.advertise<sensor_msgs::JointState>("/kuavo_arm_traj", 10);
      pubBodyPose_ = nh.advertise<kuavo_msgs::headBodyPose>("/kuavo_head_body_orientation", 10);
      pubBoxMarker_ = nh.advertise<visualization_msgs::Marker>("/grasp_box/box_marker", 10);
      pubEefMarker_ = nh.advertise<visualization_msgs::Marker>("/grasp_box/eef_marker", 10);
      enable_wbc_arm_trajectory_control_srv_ = nh.serviceClient<kuavo_msgs::changeArmCtrlMode>("/enable_wbc_arm_trajectory_control");
      exec_traj_pub_ = nh.advertise<visualization_msgs::MarkerArray>("/grasp_box/exec_two_hand_trajectory", 10);
      planed_traj_pub_ = nh.advertise<visualization_msgs::MarkerArray>("/grasp_box/planed_two_hand_trajectory", 10);
      solved_traj_pub_ = nh.advertise<visualization_msgs::MarkerArray>("/grasp_box/solved_two_hand_trajectory", 10);
      eef_wrench_pub_ = nh.advertise<std_msgs::Float64MultiArray>("/hand_wrench_cmd", 10);
      ik_cmd_pub_ = nh.advertise<motion_capture_ik::twoArmHandPoseCmd>("ik/two_arm_hand_pose_cmd", 10);
      base_pose_cmd_pub_ = nh.advertise<std_msgs::Float64MultiArray>("/base_pose_cmd", 10);

      fk_srv_ = nh.serviceClient<motion_capture_ik::fkSrv>("/ik/fk_srv");
      while(!nh.hasParam("/com_height"))
      {
        ROS_ERROR_STREAM("com_height parameter is NOT found, waiting for 0.1s.");
        ros::Duration(0.1).sleep();
      }
      ROS_INFO_STREAM("com_height parameter is founded.");
      nh.getParam("/com_height", com_height_);
      std::cout << "[GraspBox] comHeight: " << com_height_<<std::endl;
      max_hand_dis_ = getParamsFromBlackboard<double>(config, "grasp_box.max_hand_dis");
      hand_move_spd_ = getParamsFromBlackboard<double>(config, "grasp_box.hand_move_spd");
      std::cout << "max_hand_dis: " << max_hand_dis_ << std::endl;
      std::cout << "hand_move_spd: " << hand_move_spd_ << std::endl;

      pre_x_ = getParamsFromBlackboard<double>(config, "grasp_box.pre_x");
      pre_y_ = getParamsFromBlackboard<double>(config, "grasp_box.pre_y");
      bias_y_ = getParamsFromBlackboard<double>(config, "grasp_box.bias_y");
      grab_up_height_ = getParamsFromBlackboard<double>(config, "grasp_box.grab_up_height");
      grasp_force_ = getParamsFromBlackboard<double>(config, "grasp_box.grasp_force");
      auto grasp_zyx = getParamsFromBlackboard<std::vector<double>>(config, "grasp_box.grasp_zyx_agl");
      grasp_zyx_agl_ = Eigen::Vector3d(grasp_zyx[0], grasp_zyx[1], grasp_zyx[2]) * M_PI / 180.0;//rad
      auto torso_move_spd = getParamsFromBlackboard<std::vector<double>>(config, "grasp_box.torso_move_spd");
      torso_z_spd_ = torso_move_spd[0];
      torso_pitch_spd_ = torso_move_spd[1] * M_PI / 180.0;//rad/s
      ik_to_wbc_ = getParamsFromBlackboard<bool>(config, "grasp_box.ik_to_wbc");
      int tick_rate = getParamsFromBlackboard<int>(config, "tick_rate");
      auto grab_traj = getParamsFromBlackboard<std::vector<double>>(config, "grasp_box.grab_traj");
      auto put_traj = getParamsFromBlackboard<std::vector<double>>(config, "grasp_box.put_traj");
      if(grab_traj.size() % 6 != 0 || put_traj.size() % 6 != 0)
      {
        ROS_ERROR("Invalid grab_traj or put_traj size.");
        abort();
      }
      for(auto g : grab_traj)
        std::cout << "grab_traj: " << g << std::endl;
      for(int i = 0; i < grab_traj.size()/6; i++){
        grab_trajectory_.push_back(Eigen::Map<Vector6d>(grab_traj.data() + i*6, 6));
        std::cout << "grab_trajectory_[" << i << "]: " << grab_trajectory_[i].transpose() << std::endl;
      }
      for(int i = 0; i < put_traj.size()/6; i++){
        put_trajectory_.push_back(Eigen::Map<Vector6d>(put_traj.data() + i*6, 6));
        std::cout << "put_trajectory_[" << i << "]: " << put_trajectory_[i].transpose() << std::endl;
      }
      dt_ = 1 / static_cast<double>(tick_rate);
      std::cout << "dt: " << dt_ << std::endl;
      // drake
      HighlyDynamic::HumanoidInterfaceDrake* humanoid_drake_interface;
      config.blackboard->get("humanoid_drake_interface", humanoid_drake_interface);
      auto [plant, context] = humanoid_drake_interface->getPlantWithArmAndContext();
      drake_interface_ = DrakeInterface(&plant, &context);
    }

    static BT::PortsList providedPorts()
    {
      return{ BT::InputPort<Eigen::Vector3d>("box_pos"), BT::InputPort<Eigen::Vector4d>("box_quat")
        , BT::InputPort<Eigen::Vector3d>("box_size"), BT::InputPort<Eigen::Vector3d>("box_offset"), BT::InputPort<int>("grasp_type")};
    }

    BT::NodeStatus onStart() override
    {
      if(current_time_ > 0.0)
      {
        // current_time_ += 5*dt_;
        std::cout << "GraspBox restarted." << std::endl;
        return BT::NodeStatus::RUNNING;
      }
      // if(!changeKinematicMpcControlMode(1))//Arm only control mode
      //   return BT::NodeStatus::FAILURE;
      changeArmCtrlModeSrv(2);//using external controller
      if(!enableWbcArmCtrl(ik_to_wbc_)) // 手臂关节是否直接传递给wbc
      {
        ROS_ERROR("Failed to set enable_wbc_arm_trajectory control mode.");
        return BT::NodeStatus::FAILURE;
      }
      Eigen::Vector3d box_pos, box_size, box_offset_;
      Eigen::Vector4d box_quat;//xyzw
      getInput<Eigen::Vector3d>("box_pos", box_pos);
      getInput<Eigen::Vector3d>("box_size", box_size);
      getInput<Eigen::Vector4d>("box_quat", box_quat);
      getInput<Eigen::Vector3d>("box_offset", box_offset_);

      if (!config().blackboard->get("torso_pose", initial_torso_pose_)) 
      {
        ROS_ERROR("Cannot get initial_torso_pose_ from blackboard");
        return BT::NodeStatus::FAILURE;
      }
      std::cout << "initial torso pose: " << initial_torso_pose_.transpose() << std::endl;
      auto box_pos_in_robot = transBoxPos2RobotFrame(initial_torso_pose_, box_pos, box_quat).first;
      box_pos_in_robot(0) += box_size(0)/2.0 + box_offset_(0);
      box_pos_in_robot(1) += box_offset_(1);
      box_pos_in_robot(2) += box_offset_(2);

      std::cout << ">>>>>>>>>>>>>>>>>" << std::endl;
      std::cout << "box pos:" << box_pos.transpose() << std::endl;
      std::cout << "box pos in robot frame:" << box_pos_in_robot.transpose() << std::endl;
      std::cout << "box size:" << box_size.transpose() << std::endl;
      std::cout << "box quat:" << box_quat.transpose() << std::endl;
      std::cout << "<<<<<<<<<<<<<<<<<" << std::endl;
      HandPose pose_wb = transBoxPos2WorldFrame(initial_torso_pose_, box_pos_in_robot);
      pubBoxMarker_.publish(constructBoxMarker(pose_wb.first, box_size, pose_wb.second.coeffs()));

      // 当前eef位置插值到两个手的位置
      TwoHandPose two_hand_pose;
      config().blackboard->get("real_hand_poses", two_hand_pose);
      if(abs(two_hand_pose.first.first(2)) < 0.1)
      {
        return BT::NodeStatus::FAILURE;
      }
      // TwoHandPose two_hand_pose = two_hand_pose_w;
      std::cout << "l hand pos w: " << two_hand_pose.first.first.transpose() << std::endl;
      std::cout << "r hand pos w: " << two_hand_pose.second.first.transpose() << std::endl;
      two_hand_pose.first = transBoxPos2RobotFrame(initial_torso_pose_, two_hand_pose.first.first, two_hand_pose.first.second.coeffs());
      two_hand_pose.second = transBoxPos2RobotFrame(initial_torso_pose_, two_hand_pose.second.first, two_hand_pose.second.second.coeffs());
      std::cout << "l hand pos r: " << two_hand_pose.first.first.transpose() << std::endl;
      std::cout << "r hand pos r: " << two_hand_pose.second.first.transpose() << std::endl;
      // if(!getTwoHandPose(two_hand_pose))
      // {
      //   return BT::NodeStatus::FAILURE;
      // }
      // auto l_pos = transBoxPos2WorldFrame(two_hand_pose.first.first);
      // pubEefMarker_.publish(constructBoxMarker(l_pos, 0.02*Eigen::Vector3d::Ones(), Eigen::Vector4d::UnitW()));
      twoHandPoseTrajectory_.clear();
      int grasp_type;
      getInput<int>("grasp_type", grasp_type);
      // if(grasp_type == GraspType::PutDown)
      // {
      //   Eigen::Vector3d virtual_box_pos_in_robot = (two_hand_pose.first.first + two_hand_pose.second.first) / 2.0;
      //   twoHandPoseTrajectory_ = generateTwoHandPoseTrajectory(virtual_box_pos_in_robot, box_size, box_quat);
      // }
      // else
      {
        twoHandPoseTrajectory_ = generateTwoHandPoseTrajectory(box_pos_in_robot, box_size, box_quat);
      }
      twoHandPoseTrajectory_.insert(twoHandPoseTrajectory_.begin(), std::make_pair(two_hand_pose, 0.0));//插入当前eef位置+force
      // twoHandPoseTrajectory_ = interpolateTwoHandPoseTrajectory(twoHandPoseTrajectory_, max_hand_dis_);
      end_time_ = interpolateTwoHandPoseTrajectoryByCubicSpline(twoHandPoseTrajectory_, hand_move_spd_);
      // auto interp_torso = [&]()
      {
        std::vector<double> t_values = {0, end_time_};
        Vector6d torso_ref_dummy = Vector6d::Zero();
        std::vector<Eigen::VectorXd> pose_values = {torso_ref_dummy, torso_ref_dummy};
        cubic_interp_torso_ = CubicInterpolator(t_values, pose_values);
      }
      // torsoRefTrajectory_.resize(twoHandPoseTrajectory_.size(), Eigen::Vector4d::Zero());
      // visualizeTrajectory(twoHandPoseTrajectory_);
      is_runing_ = true;
      is_planed_torso_ref_traj_ = false;
      // ikParam_.torso_ref_type = 0; // 躯干cost
      // current_time_ = 0.0;
      return BT::NodeStatus::RUNNING;
    }

    /// method invoked by an action in the RUNNING state.
    BT::NodeStatus onRunning() override
    {
      // std::cout << "GraspBox running." << std::endl;
      planed_traj_pub_.publish(getVisualizeTrajectoryMsg(twoHandPoseTrajectory_));

      TwoHandPose two_hand_pose;
      config().blackboard->get("real_hand_poses", two_hand_pose);
      two_hand_pose.first = transBoxPos2RobotFrame(initial_torso_pose_, two_hand_pose.first.first, two_hand_pose.first.second.coeffs());
      two_hand_pose.second = transBoxPos2RobotFrame(initial_torso_pose_, two_hand_pose.second.first, two_hand_pose.second.second.coeffs());
      // getTwoHandPoseFromOcs2State(two_hand_pose);
      execTwoHandPoseTrajectory_.push_back(std::make_pair(two_hand_pose, 0.0));
      exec_traj_pub_.publish(getVisualizeTrajectoryMsg(execTwoHandPoseTrajectory_, {0,0,1,0.8}));
      // auto l_pos = transBoxPos2WorldFrame(two_hand_pose.first.first);
      // pubEefMarker_.publish(constructBoxMarker(l_pos, 0.02*Eigen::Vector3d::Ones(), Eigen::Vector4d::UnitW()));
      // visualizeTrajectory(twoHandPoseTrajectory_);
      int grasp_type;
      getInput<int>("grasp_type", grasp_type);
      // if(current_step_ >= twoHandPoseTrajectory_.size())
      // if(current_time_ > end_time_)
      // {
      //   if(!enableKinematicMpc(false))
      //     return BT::NodeStatus::FAILURE;
      //   // ROS_ERROR("current_step >= twoHandPoseTrajectory_.size().");
      //   initial_torso_pose_ = Eigen::VectorXd::Zero(4);
      //   return BT::NodeStatus::SUCCESS;
      // }
      // auto [l_pose, r_pose] = twoHandPoseTrajectory_[current_step_].first;
      if(current_time_ > end_time_)
      {
        // if(!is_planed_torso_ref_traj_ && grasp_type != GraspType::Move)
        // if(!is_planed_torso_ref_traj_)
        // {
        //   std::cout << "Plan torso ref trajectory." << std::endl;
        //   changeKinematicMpcControlMode(3);// control base and arm
        //   smoothlyMoveToHomeWithHandKeep();
        //   is_planed_torso_ref_traj_ = true;
        // }
        // else
        {
          std::cout << "GraspBox finished." << std::endl;
          if(!changeKinematicMpcControlMode(0))// 不接入运动学mpc控制
            return BT::NodeStatus::FAILURE;
          // changeArmCtrlModeSrv(0); // keep current control position
          return BT::NodeStatus::SUCCESS;
        }
      }
      std::cout << "current_time_: " << current_time_ << std::endl;
      HandPose l_pose, r_pose;
      l_pose.first = cubic_interp_lh_.getPos(current_time_);
      l_pose.second = cubic_interp_lh_.getQuat(current_time_);
      r_pose.first = cubic_interp_rh_.getPos(current_time_);
      r_pose.second = cubic_interp_rh_.getQuat(current_time_);
      l_pose = transBoxPos2WorldFrame(initial_torso_pose_, l_pose.first, l_pose.second.coeffs());
      r_pose = transBoxPos2WorldFrame(initial_torso_pose_, r_pose.first, r_pose.second.coeffs());
      double terrain_height=0.0;
      config().blackboard->get("terrain_height", terrain_height);
      l_pose.first(2) -= (com_height_ + terrain_height);
      r_pose.first(2) -= (com_height_ + terrain_height);
      std::cout << "l hand pos: " << l_pose.first.transpose() << std::endl;
      std::cout << "l hand quat: " << l_pose.second.coeffs().transpose() << std::endl;
      auto l_vel = cubic_interp_lh_.getVel(current_time_);
      auto r_vel = cubic_interp_rh_.getVel(current_time_);

      solvedTwoHandPoseTrajectory_.push_back(std::make_pair(std::make_pair(l_pose, r_pose), 0.0));
      solved_traj_pub_.publish(getVisualizeTrajectoryMsg(solvedTwoHandPoseTrajectory_, {0,1,0,0.6}));
      // double force = twoHandPoseTrajectory_[current_step_].second;
      double force = cubic_interp_lh_.getForce(current_time_)(1);
      // const auto& torso_ref = torsoRefTrajectory_[current_step_];
      Eigen::VectorXd torso_ref = Eigen::VectorXd::Zero(6);
      torso_ref = cubic_interp_torso_.getPos(current_time_);
      // else
      //   getTorsoRef(torso_ref);
      std::cout << "torsor ref: " << torso_ref.transpose() << std::endl;
      controlEefForce(force);
      if(is_planed_torso_ref_traj_)
      {
        std::cout << "l hand quat: " << l_pose.second.coeffs().transpose() << std::endl;
        controlTorso(torso_ref);
      }
      bool ret = controlArm(l_pose, r_pose, l_vel, r_vel);

      if(current_time_ <= 2*dt_){
        std::cout << "send!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
      if(!changeKinematicMpcControlMode(1))//Arm only control mode
        return BT::NodeStatus::FAILURE;
    }
      current_time_ += dt_;
      if(ret){
        // ++current_step_;
        // std::cout << "Process: " << current_step_/static_cast<double>(twoHandPoseTrajectory_.size())*100.0 << "%." << std::endl;
        std::cout << "Process: " << current_time_/end_time_*100.0 << "%." << std::endl;
        // ros::Duration(max_hand_dis_/hand_move_spd_).sleep();//TODO: sleep time should be adjusted according to delta pos
        return BT::NodeStatus::RUNNING;
      }
      else
      {
        ROS_ERROR("Failed to control arm.");
        changeArmCtrlModeSrv(0); // keep current control position
        return BT::NodeStatus::FAILURE;
      }
      // if(current_step_ >= twoHandPoseTrajectory_.size())
      // {
      //   // ROS_ERROR("current_step >= twoHandPoseTrajectory_.size().");
      //   return BT::NodeStatus::SUCCESS;
      // }
    }

    void onHalted() override
    {
      // nothing to do here...
      std::cout << "GraspBox interrupted" << std::endl;
      is_runing_ = false;
    }
  private:
    /*
    ** @brief 生成两个手的运动轨迹
    * @param box_pos 盒子的位置（箱子几何中心）
    * @param box_size 盒子的尺寸（xyz）
    * @param box_quat 盒子的姿态（xyzw）
    */
    TwoHandPoseWithForceTrajectory generateTwoHandPoseTrajectory(const Eigen::Vector3d& box_pos, const Eigen::Vector3d& box_size, const Eigen::Vector4d& box_quat)
    {
      TwoHandPoseWithForceTrajectory traj;
      traj.reserve(4);
      HandPose l_hand, r_hand;
      double r = box_size(1)/2.0;

      auto add_trajectory_point = [&traj, this](const Eigen::Vector3d& box_pos,
                          const Eigen::Vector3d& l_bias, 
                          const Eigen::Vector3d& r_bias,
                          double force)
      {
        Eigen::Matrix3d R_hl = ocs2::getRotationMatrixFromZyxEulerAngles(grasp_zyx_agl_);
        Eigen::Vector3d grasp_zyx_agl_r_tmp = (-1)*grasp_zyx_agl_;
        grasp_zyx_agl_r_tmp(1) = grasp_zyx_agl_r_tmp(1);
        Eigen::Matrix3d R_hr = ocs2::getRotationMatrixFromZyxEulerAngles(grasp_zyx_agl_r_tmp);

        Eigen::Quaterniond hand_quat(0.707, 0, -0.707, 0);//wxyz
        const auto R_wh = hand_quat.toRotationMatrix();
        // auto 

        HandPose l_hand, r_hand;
        l_hand.first = box_pos + l_bias;
        r_hand.first = box_pos + r_bias;
        l_hand.second = Eigen::Quaterniond(R_wh * R_hl);
        r_hand.second = Eigen::Quaterniond(R_wh * R_hr);
        auto two_hand_pose = std::make_pair(l_hand, r_hand);
        traj.push_back(std::make_pair(two_hand_pose, force));
      };
      // 定义偏移量并生成轨迹点
      std::vector<std::pair<Eigen::Vector3d, double>> biases_with_force;
      int grasp_type;
      getInput<int>("grasp_type", grasp_type);
      double f = grasp_force_;
      if(grasp_type == GraspType::GraspUp)
      {
        for(const auto& g : grab_trajectory_)
          biases_with_force.push_back({{g(0), r + g(1), g(2)}, g(4)});
      }
      else if(grasp_type == GraspType::PutDown)
      {
        for(const auto& g : put_trajectory_)
          biases_with_force.push_back({{g(0), r + g(1), g(2)}, g(4)});
      }
      else if (grasp_type == GraspType::Move) {
        biases_with_force = {
          // {-pre_x_, (r - bias_y_), 0},                   // pre-move1          
          {{0, (r - bias_y_), 0}, f},                       // pre-move2
          // {0, (r - bias_y_), 0.1},                       // move
          // {0, (r - bias_y_), 0.1},                       // move
          // {-pre_x_, (r - bias_y_), 0.1}                   // take back
        };
      }
      else
      {
        ROS_ERROR("Invalid grasp type.");
        return {};
      }
      for(const auto& b : biases_with_force)
        std::cout << "bias: " << b.first.transpose() << ", force: " << b.second << std::endl;
      Eigen::Vector3d l_bias, r_bias;
      for (const auto& b : biases_with_force) {
        const auto& bias = b.first;
        double force = b.second;
        l_bias << bias.x(), bias.y(), bias.z();
        r_bias << bias.x(), -bias.y(), bias.z();
        add_trajectory_point(box_pos, l_bias, r_bias, force);
      }
      return std::move(traj);
    }

    TwoHandPoseWithForceTrajectory interpolateTwoHandPoseTrajectory(const TwoHandPoseWithForceTrajectory& traj_with_force, double max_dis=0.02)
    {
      TwoHandPoseWithForceTrajectory interpolated_traj;
      interpolated_traj.push_back(traj_with_force[0]);
      for(int i=0; i<traj_with_force.size()-1; i++)
      {
        const auto &traj = traj_with_force[i].first;
        const auto &traj_next = traj_with_force[i+1].first;
        double force = traj_with_force[i].second;
        double force_next = traj_with_force[i+1].second;

        const auto &l_pose = traj.first;
        const auto &r_pose = traj.second;
        const auto &l_pose_next = traj_next.first;
        const auto &r_pose_next = traj_next.second;
        double l_step = (l_pose_next.first - l_pose.first).norm() / max_dis;
        double r_step = (r_pose_next.first - r_pose.first).norm() / max_dis;
        int step_num = std::ceil(std::max(l_step, r_step));
        for(int j=1; j<=step_num; j++)
        {
          double ratio = j / static_cast<double>(step_num);
          Eigen::Vector3d l_pos_inter = l_pose.first + (l_pose_next.first - l_pose.first) * ratio;
          Eigen::Vector3d r_pos_inter = r_pose.first + (r_pose_next.first - r_pose.first) * ratio;

          Eigen::Quaterniond l_quat_inter = l_pose.second.slerp(ratio, l_pose_next.second);
          Eigen::Quaterniond r_quat_inter = r_pose.second.slerp(ratio, r_pose_next.second);
          
          double force_inter = force + (force_next - force) * ratio;

          auto two_hand_pose = std::make_pair(std::make_pair(l_pos_inter, l_quat_inter), std::make_pair(r_pos_inter, r_quat_inter));
          interpolated_traj.push_back(std::make_pair(two_hand_pose, force_inter));
        }
      }

      return std::move(interpolated_traj);
    }

    /**
     * @brief 三次样条插值
     * @param traj_with_force 带有力的轨迹
     * @param vel 速度
     * @return 终止时间
     */
    double interpolateTwoHandPoseTrajectoryByCubicSpline(const TwoHandPoseWithForceTrajectory& traj_with_force, double vel=0.5)
    {
      std::vector<double> t_values_l, t_values_r;
      std::vector<Eigen::VectorXd> pos_values_l, pos_values_r;
      std::vector<Eigen::Quaterniond> quat_values_l, quat_values_r;
      std::vector<Eigen::Vector3d> force_values_l, force_values_r;

      t_values_l.push_back(0);
      t_values_r.push_back(0);
      pos_values_l.push_back(traj_with_force[0].first.first.first);
      pos_values_r.push_back(traj_with_force[0].first.second.first);
      quat_values_l.push_back(traj_with_force[0].first.first.second);
      quat_values_r.push_back(traj_with_force[0].first.second.second);
      force_values_l.push_back({0, traj_with_force[0].second, 0});
      force_values_r.push_back({0, traj_with_force[0].second, 0});
      for(int i=0; i<traj_with_force.size()-1; i++)
      {
        const auto &traj = traj_with_force[i].first;
        const auto &traj_next = traj_with_force[i+1].first;
        const double force_next = traj_with_force[i+1].second;

        const auto &l_pose = traj.first;
        const auto &l_pose_next = traj_next.first;
        const auto &r_pose = traj.second;
        const auto &r_pose_next = traj_next.second;

        double time_cost_l = std::max(0.05, (l_pose_next.first - l_pose.first).norm() / vel);
        double time_cost_r = std::max(0.05, (r_pose_next.first - r_pose.first).norm() / vel);
        // std::cout << "time_cost_l: " << time_cost_l << ", time_cost_r: " << time_cost_r << std::endl;

        t_values_l.push_back(t_values_l.back() + time_cost_l);
        t_values_r.push_back(t_values_r.back() + time_cost_r);
        pos_values_l.push_back(l_pose_next.first);
        pos_values_r.push_back(r_pose_next.first);
        quat_values_l.push_back(l_pose_next.second);
        quat_values_r.push_back(r_pose_next.second);
        force_values_l.push_back({0, force_next, 0});
        force_values_r.push_back({0, force_next, 0});
      }
      cubic_interp_lh_ = CubicInterpolator(t_values_l, pos_values_l);
      cubic_interp_lh_.addQuaternion(quat_values_l);
      cubic_interp_lh_.addForce(force_values_l);
      cubic_interp_rh_ = CubicInterpolator(t_values_r, pos_values_r);
      cubic_interp_rh_.addQuaternion(quat_values_r);
      cubic_interp_rh_.addForce(force_values_r);
      return std::max(t_values_l.back(), t_values_r.back());
    }

    std::vector<Eigen::Vector4d> interpolateTorsoRefTrajectory(Eigen::Vector4d& start, const Eigen::Vector4d& end, double z_dis=0.02, double agl_dis=2*M_PI/180)
    {
      std::vector<Eigen::Vector4d> interpolated_traj;
      int step_num = static_cast<int>(std::ceil((end(0) - start(0))/z_dis));
      for(int i=1; i<4; i++)
        step_num = std::max(step_num, static_cast<int>(std::ceil((end(i) - start(i))/agl_dis)));
      for(int j=1; j<=step_num; j++)
      {
        double ratio = j / static_cast<double>(step_num);
        Eigen::Vector4d torso_ref_inter = start + (end - start) * ratio;        
        interpolated_traj.push_back(torso_ref_inter);
      }
      return std::move(interpolated_traj);
    }

    double interpolateTorsoRefTrajectoryByCubicSpline(Vector6d& start, const Vector6d& end, double z_vel=0.02, double agl_vel=30*M_PI/180)
    {
      std::vector<double> t_values;
      std::vector<Eigen::VectorXd> pose_values;
      t_values.push_back(0);
      pose_values.push_back(start);
      double time_cost = abs(end(2) - start(2)) / z_vel;
      for(int i=3; i<6; i++)
        time_cost = std::max(time_cost, (std::abs((end(i) - start(i))/agl_vel)));
      t_values.push_back(t_values.back() + time_cost);
      pose_values.push_back(end);
      cubic_interp_torso_ = CubicInterpolator(t_values, pose_values);
      return time_cost;
    }

    void controlTorso(const Vector6d& torso_ref)
    {
      std_msgs::Float64MultiArray msg;
      msg.data.resize(6);
      for(int i=0; i<6; i++)
        msg.data[i] = torso_ref(i);
      base_pose_cmd_pub_.publish(msg);
    }

    bool controlArm(const HandPose& left_hand_pose, const HandPose& right_hand_pose, 
      const Eigen::Vector3d& left_hand_vel, const Eigen::Vector3d& right_hand_vel,
      const Vector6d &torso_ref = Vector6d::Zero())
    {
      motion_capture_ik::twoArmHandPoseCmd msg = getIKCmdMsg(left_hand_pose, right_hand_pose, torso_ref);
      ik_cmd_pub_.publish(msg);
      return true;
      // motion_capture_ik::twoArmHandPoseCmdSrv::Response response;
      // bool ret = sendIKCmdSrv(msg, response);
      // if(ret && response.success)
      // {
      //   HandPose l_hand_pose_result, r_hand_pose_result;
      //   const auto& hand_poses = response.hand_poses;
      //   for (int i = 0; i < 3; ++i) {
      //       l_hand_pose_result.first[i] = hand_poses.left_pose.pos_xyz[i];
      //       r_hand_pose_result.first[i] = hand_poses.right_pose.pos_xyz[i];
      //   }

      //   l_hand_pose_result.second.x() = hand_poses.left_pose.quat_xyzw[0];
      //   l_hand_pose_result.second.y() = hand_poses.left_pose.quat_xyzw[1];
      //   l_hand_pose_result.second.z() = hand_poses.left_pose.quat_xyzw[2];
      //   l_hand_pose_result.second.w() = hand_poses.left_pose.quat_xyzw[3];

      //   r_hand_pose_result.second.x() = hand_poses.right_pose.quat_xyzw[0];
      //   r_hand_pose_result.second.y() = hand_poses.right_pose.quat_xyzw[1];
      //   r_hand_pose_result.second.z() = hand_poses.right_pose.quat_xyzw[2];
      //   r_hand_pose_result.second.w() = hand_poses.right_pose.quat_xyzw[3];

      //   double l_error_norm = (left_hand_pose.first - l_hand_pose_result.first).norm();
      //   double r_error_norm = (right_hand_pose.first - r_hand_pose_result.first).norm();
      //   // std::cout << "l_error_norm: " << l_error_norm << std::endl;
      //   // std::cout << "r_error_norm: " << r_error_norm << std::endl;
      //   if(l_error_norm > 1e-2 && r_error_norm > 1e-2)
      //   {
      //     ROS_ERROR_STREAM("IK's result is bad. l_error_norm: " << l_error_norm << ", r_error_norm: " << r_error_norm);
      //     return false;
      //   }
      //   // TO-DO: 添加关节速度 (11/28 by matthew)
      //   std::cout << "left_hand_vel: " << left_hand_vel.transpose() << std::endl;
      //   std::cout << "right_hand_vel: " << right_hand_vel.transpose() << std::endl;
      //   const int arm_num = response.q_arm.size();
      //   Eigen::VectorXd q_arm = Eigen::Map<Eigen::VectorXd>(response.q_arm.data(), response.q_arm.size());
      //   auto J_l = drake_interface_.getHandJacobian(q_arm, HandSide::LEFT);
      //   auto J_r = drake_interface_.getHandJacobian(q_arm, HandSide::RIGHT);
      //   // std::cout << "J_l size: " << J_l.rows() << "x" << J_l.cols() << std::endl;
      //   // std::cout << "J_l: " << J_l << std::endl;
      //   Eigen::VectorXd dq_arm(arm_num);
      //   dq_arm.head(arm_num/2) = pseudoInverse(J_l) * left_hand_vel;
      //   dq_arm.tail(arm_num/2) = pseudoInverse(J_r) * right_hand_vel;
      //   // std::cout << "dq_arm: " << dq_arm.transpose() << std::endl;

      //   std::vector<double> dq_arm_vec(dq_arm.data(), dq_arm.data() + dq_arm.size());
      //   pubArmTraj_.publish(getJointStatesMsg(response.q_arm, dq_arm_vec));
      //   // std::cout << "q_torso: " << Eigen::Map<Eigen::VectorXd>(response.q_torso.data(), response.q_torso.size()).transpose() << std::endl;
      //   if(response.with_torso)
      //     pubBodyPose_.publish(getHeadBodyPoseMsg(response.q_torso));
      //   {
      //     Eigen::VectorXd q_general = Eigen::VectorXd::Zero(6+14); // xyz + ypr + 2*q_arm
      //     q_general.head(6) = Eigen::Map<Eigen::VectorXd>(response.q_torso.data(), response.q_torso.size());// torso
      //     q_general.tail(14) = Eigen::Map<Eigen::VectorXd>(response.q_arm.data(), response.q_arm.size());   // arm
      //     TwoHandPose two_hand_pose;
      //     if(!getTwoHandPose(q_general, two_hand_pose))
      //     {
      //       ROS_ERROR("Failed to call FK service.");
      //     }
      //     solvedTwoHandPoseTrajectory_.push_back(std::make_pair(two_hand_pose, 0.0));
      //     solved_traj_pub_.publish(getVisualizeTrajectoryMsg(solvedTwoHandPoseTrajectory_, {0,1,0,0.6}));
      //   }
      //   return true;
      // }
      // ROS_ERROR("Failed to get IK result.");
      // ROS_ERROR_STREAM("left_hand_pos: " << left_hand_pose.first.transpose());
      // ROS_ERROR_STREAM("right_hand_pos: " << right_hand_pose.first.transpose());
      // return false;
    }

    motion_capture_ik::twoArmHandPoseCmd getIKCmdMsg(const HandPose& left_hand_pose, const HandPose& right_hand_pose, const Vector6d &torso_ref)
    {
      motion_capture_ik::twoArmHandPoseCmd msg;
      msg.ik_param = ikParam_;
      msg.use_custom_ik_param = true;
      msg.joint_angles_as_q0 = false;
      for(int i=0; i<3; i++)
      {
        msg.hand_poses.left_pose.elbow_pos_xyz[i] = 0;
        msg.hand_poses.right_pose.elbow_pos_xyz[i] = 0;
        // 
        msg.hand_poses.left_pose.pos_xyz[i] = left_hand_pose.first[i];
        msg.hand_poses.right_pose.pos_xyz[i] = right_hand_pose.first[i];
      }

      msg.hand_poses.left_pose.quat_xyzw[0] = left_hand_pose.second.x();
      msg.hand_poses.left_pose.quat_xyzw[1] = left_hand_pose.second.y();
      msg.hand_poses.left_pose.quat_xyzw[2] = left_hand_pose.second.z();
      msg.hand_poses.left_pose.quat_xyzw[3] = left_hand_pose.second.w();

      msg.hand_poses.right_pose.quat_xyzw[0] = right_hand_pose.second.x();
      msg.hand_poses.right_pose.quat_xyzw[1] = right_hand_pose.second.y();
      msg.hand_poses.right_pose.quat_xyzw[2] = right_hand_pose.second.z();
      msg.hand_poses.right_pose.quat_xyzw[3] = right_hand_pose.second.w();

      // msg.torso_ref.resize(6);
      // for(int i=0; i<6; i++)
      //   msg.torso_ref[i] = torso_ref[i];

      return std::move(msg);
    }

    bool sendIKCmdSrv(const motion_capture_ik::twoArmHandPoseCmd& msg, motion_capture_ik::twoArmHandPoseCmdSrv::Response& response)
    {
      const std::string service_name = "/ik/two_arm_hand_pose_cmd_srv";
      ros::NodeHandle nh;

      // 等待服务可用
      if (!ros::service::waitForService(service_name, ros::Duration(5))) {
        ROS_ERROR("Service %s not available", service_name.c_str());
        return false;
      }

      // 创建服务代理
      ros::ServiceClient client = nh.serviceClient<motion_capture_ik::twoArmHandPoseCmdSrv>(service_name);
      motion_capture_ik::twoArmHandPoseCmdSrv srv; // 创建服务请求对象
      srv.request.twoArmHandPoseCmdRequest = msg; // 假设 `cmd` 是请求中的字段，将 `msg` 赋值给它

      // 调用服务
      if (client.call(srv)) {
        response = srv.response;
        return true; // 服务调用成功
      } else {
        ROS_ERROR("Failed to call service %s", service_name.c_str());
        return false; // 服务调用失败
      }
    }

    /**
     * @brief 将盒子的位置从世界坐标系转换到机器人的坐标系下
     * @param p_wb 盒子的位置（世界坐标系）
     * @param quat_wb 盒子的姿态（世界坐标系）
     * @refer: https://www.lejuhub.com/highlydynamic/kuavodevlab/-/issues/707
     */
    HandPose transBoxPos2RobotFrame(const Eigen::Vector4d& torso_pose, const Eigen::Vector3d& p_wb, const Eigen::Vector4d& quat_wb)
    {
      // s: local frame, b: box frame, w: world frame, r: robot frame
      Eigen::Vector3d p_ws(torso_pose[0], torso_pose[1], 0);
      Eigen::Matrix3d R_ws = ocs2::getRotationMatrixFromZyxEulerAngles(Eigen::Vector3d(torso_pose(3), 0, 0));
      Eigen::Vector3d p_sb = R_ws.transpose() * (p_wb - p_ws);
      double terrain_height=0.0;
      config().blackboard->get("terrain_height", terrain_height);
      Eigen::Vector3d p_sr(0,0, com_height_ + terrain_height);
      Eigen::Vector3d p_rb = p_sb - p_sr;
      Eigen::Quaterniond q_wb(quat_wb.w(), quat_wb.x(), quat_wb.y(), quat_wb.z());
      auto R_wb = q_wb.toRotationMatrix();
      auto R_rb = R_ws.transpose() * R_wb;
      Eigen::Quaterniond q_rb(R_rb);
      return std::make_pair(p_rb, q_rb);
    }

    /**
     * @brief 将盒子的位置从机器人坐标系转换到世界坐标系的坐标系下
     * @param p_rb_s 盒子的位置（机器人坐标系）
     * @param quat_rb 盒子的姿态（机器人坐标系）
     * @refer: https://www.lejuhub.com/highlydynamic/kuavodevlab/-/issues/707
     */
    HandPose transBoxPos2WorldFrame(const Eigen::Vector4d& torso_pose, const Eigen::Vector3d& p_rb_s, const Eigen::Vector4d& quat_rb=Eigen::Vector4d::UnitW())
    {
      // s: local frame, b: box frame, w: world frame, r: robot frame
      Eigen::Vector3d p_ws(torso_pose[0], torso_pose[1], 0);
      double terrain_height=0.0;
      config().blackboard->get("terrain_height", terrain_height);
      Eigen::Vector3d p_sr(0,0, com_height_ + terrain_height);
      Eigen::Vector3d p_sb = p_sr + p_rb_s;

      Eigen::Matrix3d R_ws = ocs2::getRotationMatrixFromZyxEulerAngles(Eigen::Vector3d(torso_pose(3), 0, 0));
      Eigen::Vector3d p_wb = p_ws + R_ws * p_sb;
      auto q_rb = Eigen::Quaterniond(quat_rb.w(), quat_rb.x(), quat_rb.y(), quat_rb.z());
      auto R_rb = q_rb.toRotationMatrix();
      auto R_wb = R_ws * R_rb; // R_ws=R_wr
      Eigen::Quaterniond q_wb(R_wb);
      return std::make_pair(p_wb, q_wb);
    }

    bool enableWbcArmCtrl(int mode)
    {
      kuavo_msgs::changeArmCtrlMode srv;
      srv.request.control_mode = mode;
      if (enable_wbc_arm_trajectory_control_srv_.call(srv)) 
      {
        return true;
      }
      else
        ROS_ERROR("Failed to call service /enable_wbc_arm_trajectory_control");
      return false;
    }

    bool getTwoHandPose(const Eigen::VectorXd &q, TwoHandPose &hand_poses)
    {
      auto& [l_pose, r_pose] = hand_poses;
      motion_capture_ik::fkSrv srv;
      srv.request.q.resize(q.size());
      for (int i = 0; i < q.size(); ++i)
        srv.request.q[i] = q(i);
      if (fk_srv_.call(srv)) 
      {
        const auto & l_pose_res = srv.response.hand_poses.left_pose;
        const auto & r_pose_res = srv.response.hand_poses.right_pose;
        for (int i = 0; i < 3; ++i) {
          l_pose.first(i) = l_pose_res.pos_xyz[i];
          r_pose.first(i) = r_pose_res.pos_xyz[i];
        }
        //
        // l_pose.first.z() -= com_height_;
        // r_pose.first.z() -= com_height_;

        l_pose.second.x() = l_pose_res.quat_xyzw[0];
        l_pose.second.y() = l_pose_res.quat_xyzw[1];
        l_pose.second.z() = l_pose_res.quat_xyzw[2];
        l_pose.second.w() = l_pose_res.quat_xyzw[3];

        r_pose.second.x() = r_pose_res.quat_xyzw[0];
        r_pose.second.y() = r_pose_res.quat_xyzw[1];
        r_pose.second.z() = r_pose_res.quat_xyzw[2];
        r_pose.second.w() = r_pose_res.quat_xyzw[3];

        return true;
      }
      else
        ROS_ERROR("Failed to call service /ik/fk_srv");
      return false;
    }

    // 可视化轨迹
    visualization_msgs::MarkerArray getVisualizeTrajectoryMsg(const TwoHandPoseWithForceTrajectory& twoHandPoseTrajectory, std::vector<double> rgba={1,0,0,1})
    {
      visualization_msgs::MarkerArray marker_array;
      visualization_msgs::Marker marker_l, marker_r;
      marker_l.header.frame_id = "odom";
      marker_l.header.stamp = ros::Time::now();
      // marker_l.ns = "l_hand";
      marker_l.id = 0;
      marker_l.type = visualization_msgs::Marker::LINE_STRIP;
      marker_l.action = visualization_msgs::Marker::ADD;
      marker_l.scale.x = 0.01;    // 设置线宽
      marker_l.color.r = rgba[0]; // 设置颜色
      marker_l.color.g = rgba[1]; // 设置颜色
      marker_l.color.b = rgba[2]; // 设置颜色
      marker_l.color.a = rgba[3]; // 设置透明度

      marker_r = marker_l;
      // marker_r.ns = "r_hand";
      marker_r.id = 1;
      auto getTraj = [&](visualization_msgs::Marker &marker, bool isLeft)
      {
        for(const auto& pose_with_force : twoHandPoseTrajectory)
        {
          const auto& pose = pose_with_force.first;
          const HandPose& hand_pose = (isLeft ? pose.first : pose.second);
          auto pos = transBoxPos2WorldFrame(initial_torso_pose_, hand_pose.first).first;
          geometry_msgs::Point p;
          p.x = pos.x();
          p.y = pos.y();
          p.z = pos.z();
          marker.points.push_back(p);
        }
      };
      // left hand
      getTraj(marker_l, true);
      marker_array.markers.push_back(marker_l);
      // right hand
      getTraj(marker_r, false);
      marker_array.markers.push_back(marker_r);

      return std::move(marker_array);
    }

    void controlEefForce(double force)
    {
      Vector6d wrench_l = Vector6d::Zero();
      Vector6d wrench_r = Vector6d::Zero();
      wrench_l(1) = +force;
      wrench_r(1) = -force;
      eef_wrench_pub_.publish(getEefWrenchCmdMsg(wrench_l, wrench_r));
    }

    void smoothlyMoveToHomeWithHandKeep()
    {
      Eigen::VectorXd ocs2_state;
      if (!config().blackboard->get("ocs2_state", ocs2_state)) 
      {
        ROS_ERROR("Cannot get ocs2_state from blackboard");
        return;
      }
      Vector6d torso_ref_current = ocs2_state.segment<6>(6);
      double terrain_height=0.0;
      config().blackboard->get("terrain_height", terrain_height);
      // std::cout << "[GraspBox] terrain_height: " << terrain_height << "\n";
      torso_ref_current(2) -= (com_height_ + terrain_height);
      double pitch_ref = M_PI / 180.0 * getParamsFromBlackboard<double>(config(), "normal_torso_pitch");
      Vector6d torso_ref_target = (Vector6d() << torso_ref_current.head<2>(), 0, 0, pitch_ref, 0).finished();
      std::cout << "Torso current: " << torso_ref_current.transpose() << std::endl;
      std::cout << "Torso target: " << torso_ref_target.transpose() << std::endl;
      // auto torsoRefTrajectoryPost = interpolateTorsoRefTrajectory(torso_ref_current, torso_ref_target);
      end_time_ = interpolateTorsoRefTrajectoryByCubicSpline(torso_ref_current, torso_ref_target, torso_z_spd_, torso_pitch_spd_);
      std::cout << "Torso to normal pitch end time: " << end_time_ << " s." << std::endl;
      current_time_ = 0.0;

      const double et = cubic_interp_lh_.getEndTime();
      const Eigen::Vector3d force = (Eigen::Vector3d() << 0, grasp_force_, 0).finished();
      CubicInterpolator ci_lh_tmp, ci_rh_tmp;
      // std::cout << "quat: " << current_two_hand_pose.first.second.coeffs().transpose() << std::endl;
      ci_lh_tmp = CubicInterpolator({0, end_time_}, {cubic_interp_lh_.getPos(et), cubic_interp_lh_.getPos(et)});
      ci_lh_tmp.addQuaternion({cubic_interp_lh_.getQuat(et), cubic_interp_lh_.getQuat(et)});
      ci_lh_tmp.addForce({force, force});
      ci_rh_tmp = CubicInterpolator({0, end_time_}, {cubic_interp_rh_.getPos(et), cubic_interp_rh_.getPos(et)});
      ci_rh_tmp.addQuaternion({cubic_interp_rh_.getQuat(et), cubic_interp_rh_.getQuat(et)});
      ci_rh_tmp.addForce({force, force});

      cubic_interp_lh_ = ci_lh_tmp;
      cubic_interp_rh_ = ci_rh_tmp;
      // ikParam_.torso_ref_type = 1; // 躯干约束
    }

    bool getTwoHandPose(TwoHandPose &two_hand_pose)
    {
      Eigen::VectorXd ocs2_state;
      if (!config().blackboard->get("ocs2_state", ocs2_state)) 
      {
        ROS_ERROR("Cannot get ocs2_state from blackboard");
        return false;
      }
      Eigen::VectorXd q_general = Eigen::VectorXd::Zero(6+14); // xyz + ypr + 2*q_arm
      q_general.head(2) = ocs2_state.segment<2>(6); // xy
      q_general(2) = ocs2_state(8) - com_height_;   // z
      q_general(4) = ocs2_state(10);                // pitch
      q_general.tail(14) = ocs2_state.tail(14);     // q_arm
      if(!getTwoHandPose(q_general, two_hand_pose))
      {
        ROS_ERROR("Failed to call FK service.");
        return false;
      }
      return true;
    }

    bool getTwoHandPoseFromOcs2State(TwoHandPose& two_hand_pose)
    {
      Eigen::VectorXd ocs2_state;
      if (!config().blackboard->get("ocs2_state", ocs2_state)) 
      {
        ROS_ERROR("Cannot get ocs2_state from blackboard");
        return false;
      }
      Eigen::VectorXd q_general = Eigen::VectorXd::Zero(6+14); // xyz + ypr + 2*q_arm
      q_general.head(2) = ocs2_state.segment<2>(6); // xy
      q_general(2) = ocs2_state(8) - com_height_;   // z
      q_general(4) = ocs2_state(10);                // pitch
      q_general.tail(14) = ocs2_state.tail(14);     // q_arm
      if(!getTwoHandPose(q_general, two_hand_pose))
      {
        ROS_ERROR("Failed to call FK service.");
        return false;
      }
      return true;
    }

    void getTorsoRef(Eigen::VectorXd &torso_ref, const std::vector<double>& torso_ref_tol={1e-3, 1e-3, 0.02, 0, 3*M_PI/180, 0})
    {
      // ikParam_.torso_ref_tol = torso_ref_tol;
      // ikParam_.torso_ref_type = 0;

      Eigen::VectorXd ocs2_state;
      config().blackboard->get("ocs2_state", ocs2_state);
      torso_ref = ocs2_state.segment<6>(6);
      torso_ref(3) = 0;//yaw
      torso_ref(5) = 0;//roll
    }
  private:
    motion_capture_ik::ikSolveParam ikParam_;
    ros::Publisher pubArmTraj_;
    ros::Publisher pubBodyPose_;
    ros::Publisher pubBoxMarker_;
    ros::Publisher pubEefMarker_;
    ros::Publisher planed_traj_pub_;
    ros::Publisher solved_traj_pub_;
    ros::Publisher exec_traj_pub_;
    ros::Publisher eef_wrench_pub_;
    ros::Publisher ik_cmd_pub_;
    ros::Publisher base_pose_cmd_pub_;
    ros::ServiceClient enable_wbc_arm_trajectory_control_srv_;
    ros::ServiceClient fk_srv_;
    TwoHandPoseWithForceTrajectory twoHandPoseTrajectory_;
    TwoHandPoseWithForceTrajectory execTwoHandPoseTrajectory_;
    TwoHandPoseWithForceTrajectory solvedTwoHandPoseTrajectory_;
    DrakeInterface drake_interface_;
    // std::vector<Eigen::Vector4d> torsoRefTrajectory_; // z-ypr
    int current_step_ = 0;
    bool is_runing_ = false;
    double com_height_{0};
    double max_hand_dis_{0.02};// m
    double hand_move_spd_{0.4};// m/s
    // grasp
    double pre_x_ = 0.1;
    double pre_y_ = 0.1;
    double bias_y_ = 0.01;
    double grab_up_height_ = 0.1;
    bool is_planed_torso_ref_traj_ = false;
    double grasp_force_ = 10;
    Eigen::Vector3d grasp_zyx_agl_;
    double torso_z_spd_ = 0.1;     // m/s
    double torso_pitch_spd_ = 0.5; // rad/s
    CubicInterpolator cubic_interp_lh_, cubic_interp_rh_, cubic_interp_torso_;
    double end_time_ = 0.0;
    double current_time_ = 0.0;
    double dt_ = 0.01;// s
    bool ik_to_wbc_ = true;
    Eigen::Vector4d initial_torso_pose_;
    std::vector<Vector6d> grab_trajectory_;
    std::vector<Vector6d> put_trajectory_;
  };
} // namespace GrabBox