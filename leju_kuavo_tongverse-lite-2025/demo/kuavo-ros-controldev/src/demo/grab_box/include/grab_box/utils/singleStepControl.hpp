#include <ros/ros.h>
#include <kuavo_msgs/footPose.h>
#include <kuavo_msgs/footPoseTargetTrajectories.h>
#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <grab_box/utils/sat.hpp>

#pragma once

namespace GrabBox
{
  using namespace std;
  using namespace Eigen;

  kuavo_msgs::footPoseTargetTrajectories get_foot_pose_traj_msg(const vector<double>& time_traj,
                                                    const vector<int>& foot_idx_traj,
                                                    const vector<Vector4d>& foot_traj,
                                                    const vector<Vector4d>& torso_traj) {
    kuavo_msgs::footPoseTargetTrajectories msg;
    msg.timeTrajectory = time_traj;
    msg.footIndexTrajectory = foot_idx_traj;

    for (int i = 0; i < foot_traj.size(); ++i) {
      kuavo_msgs::footPose foot_pose_msg;
      for(int j = 0; j < foot_traj[0].size(); ++j) 
      {
        foot_pose_msg.footPose[j] = foot_traj[i][j];
        foot_pose_msg.torsoPose[j] = torso_traj[i][j];
      }
      msg.footPoseTrajectory.push_back(foot_pose_msg);
    }

    return msg;
  }

  pair<Vector3d, Vector3d> generate_steps(const Vector3d& torso_pos, double torso_yaw, double foot_bias) {
    Vector3d l_foot_bias(0, foot_bias, -torso_pos[2]);
    Vector3d r_foot_bias(0, -foot_bias, -torso_pos[2]);

    Matrix3d R_z;
    R_z << cos(torso_yaw), -sin(torso_yaw), 0,
            sin(torso_yaw), cos(torso_yaw), 0,
            0, 0, 1;

    Vector3d l_foot = torso_pos + R_z * l_foot_bias;
    Vector3d r_foot = torso_pos + R_z * r_foot_bias;
    return make_pair(l_foot, r_foot);
  }

  kuavo_msgs::footPoseTargetTrajectories get_multiple_steps_msg(const vector<Vector4d>& body_poses, double dt, double foot_bias, bool collision_check = true) {
    int num_steps = 2 * body_poses.size();
    vector<double> time_traj;
    vector<int> foot_idx_traj;
    vector<Vector4d> foot_traj;
    vector<Vector4d> torso_traj;

    RotatingRectangle l_foot_rect_last(Vector2d(0, 0.1), 0.24, 0.1, 0);
    RotatingRectangle r_foot_rect_last(Vector2d(0, -0.1), 0.24, 0.1, 0);
    bool is_left_first = true;

    Vector4d torso_pose_last = Vector4d::Zero();
    for (int i = 0; i < num_steps; ++i) {
      time_traj.push_back(dt * (i + 1));
      Vector4d body_pose = body_poses[i / 2];
      Vector3d torso_pos(body_pose[0], body_pose[1], body_pose[2]);
      double torso_yaw = body_pose[3] * M_PI / 180.0;

      auto [l_foot, r_foot] = generate_steps(torso_pos, torso_yaw, foot_bias);
      Vector4d l_foot_vec(l_foot[0], l_foot[1], l_foot[2], torso_yaw);
      Vector4d r_foot_vec(r_foot[0], r_foot[1], r_foot[2], torso_yaw);

      if (i % 2 == 0) {
        is_left_first = (torso_yaw - torso_pose_last[3] >= 0.0);

        if (collision_check) {
            RotatingRectangle l_foot_rect_next(Vector2d(l_foot[0], l_foot[1]), 0.24, 0.1, torso_yaw);
            RotatingRectangle r_foot_rect_next(Vector2d(r_foot[0], r_foot[1]), 0.24, 0.1, torso_yaw);

            bool l_collision = l_foot_rect_next.is_collision(r_foot_rect_last);
            bool r_collision = r_foot_rect_next.is_collision(l_foot_rect_last);

            if (l_collision && r_collision) {
              ROS_ERROR("[Error] Detect collision, Please adjust your body_poses input!!!");
              break;
            }
            if (l_collision) {
              ROS_INFO("[Info] Left foot[%d] is in collision, switch to right foot", 1 + i / 2);
              is_left_first = false;
            }
            if (r_collision) {
              ROS_INFO("[Info] Right foot[%d] is in collision, switch to left foot", 1 + i / 2);
              is_left_first = true;
            }

            l_foot_rect_last = l_foot_rect_next;
            r_foot_rect_last = r_foot_rect_next;
          }
        }

        int foot_idx = is_left_first ? 0 : 1;
        auto next_body_pose = Vector4d(body_pose[0], body_pose[1], body_pose[2], torso_yaw);
        if (i % 2 == 0) {
          torso_traj.push_back((torso_pose_last + next_body_pose)/2.0);
          foot_idx_traj.push_back(foot_idx);
          foot_traj.push_back(is_left_first ? l_foot_vec : r_foot_vec);
        } else {
          torso_traj.push_back(next_body_pose);
          foot_idx = 1 - foot_idx;
          foot_idx_traj.push_back(foot_idx);
          foot_traj.push_back(is_left_first ? r_foot_vec : l_foot_vec);
        }

        // torso_traj.push_back(Vector4d(body_pose[0], body_pose[1], body_pose[2], torso_yaw));
        torso_pose_last = torso_traj.back();
    }

    return get_foot_pose_traj_msg(time_traj, foot_idx_traj, foot_traj, torso_traj);
  }
} // namespace GrabBox
// int main(int argc, char** argv) {
//     ros::init(argc, argv, "foot_pose_publisher");
//     ros::NodeHandle nh;
//     ros::Publisher pub = nh.advertise<kuavo_msgs::footPoseTargetTrajectories>(
//         "/humanoid_mpc_foot_pose_target_trajectories", 10);

//     ros::Duration(1).sleep(); // 等待一定时间以确保订阅者已经准备好

//     bool collision_check = true;
//     double dt = 0.8; // 迈一步的时间间隔
//     vector<Vector4d> body_poses = {
//         Vector4d(0.1, 0.0, 0, 0),
//         Vector4d(0.1, -0.1, 0, 0),
//         Vector4d(0.2, -0.1, 0, -30),
//         Vector4d(0.3, 0.0, 0, 0),
//         Vector4d(0.4, 0.0, 0, -30),
//         Vector4d(0.5, 0.0, 0, 0),
//     };

//     auto msg = get_multiple_steps_msg(body_poses, dt, collision_check);
//     pub.publish(msg);

//     ros::spin();
//     return 0;
// }
