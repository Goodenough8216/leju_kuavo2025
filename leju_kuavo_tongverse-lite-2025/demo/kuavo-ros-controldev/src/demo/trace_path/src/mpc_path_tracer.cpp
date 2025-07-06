// mpc_path_tracer.cpp
#include "mpc_path_tracer.h"
#include "common/utils.h"

#include <cmath>
#include <iostream>
#include <thread>
#include <chrono>
#include <tf/tf.h>
#include <Eigen/Eigen>
#include <drake/solvers/solve.h>
#include <drake/common/find_resource.h>
#include <drake/solvers/mathematical_program.h>
#include <drake/solvers/snopt_solver.h>
#include <drake/solvers/ipopt_solver.h>

using namespace drake::solvers;
namespace trace_path {

// 将全局坐标系下的点转换为局部坐标系下的点
static void TransformToBaseLink(const geometry_msgs::Pose &current_pose, 
                        const geometry_msgs::Point &global_point, 
                        geometry_msgs::Point &local_point) {
  // 获取当前姿态的旋转矩阵
  tf::Quaternion q(current_pose.orientation.x, current_pose.orientation.y, current_pose.orientation.z, current_pose.orientation.w);
  tf::Matrix3x3 rot(q);

  // 创建向量来存储转换后的坐标
  tf::Vector3 global_vec(global_point.x - current_pose.position.x,
                         global_point.y - current_pose.position.y,
                         0.0);  // 假设平面运动

  // 应用旋转矩阵
  tf::Vector3 local_vec = rot.inverse() * global_vec;

  // 设置输出点
  local_point.x = local_vec[0];
  local_point.y = local_vec[1];
}

// 定义MPC参数
const int kNum_knots = 31;  // knot点数量
const double dt = 0.1;      // 时间步长
MpcPathTracer::MpcPathTracer(ros::NodeHandle &nh) : PathTracerBase(nh)
{
    mpc_path_pub_ = nh_.advertise<nav_msgs::Path>("trace_path/mpc/path", 10, true);
}

void MpcPathTracer::Follow(const PathGeneratorUniquePtr &path_generator)
{
    ros::Rate rate(1); // 控制频率

    // 等待起始点设置...
    flag_reset_start_point_ = true;
    while (ros::ok() && flag_reset_start_point_) {
        ros::spinOnce();
        rate.sleep();
        std::cout << "Waiting for start point... \n";
    }

    auto global_path = path_generator->GeneratePath(start_point_odom_->pose.pose);
    if (global_path.poses.size() <= 0)  return;

    // 先发布路径提供查看
    PublishPath(global_path);

    // TODO: 添加 按下 Enter 开始
    std::cout << "Following...\n";
    std::cout << "-------------------------------------------------------\n";
    std::cout << "Path Start Pose:\n" << global_path.poses.front().pose << "\n";
    std::cout << "-------------------------------------------------------\n";
    std::cout << "Path End Pose:\n" << global_path.poses.back().pose << std::endl;
    std::cout << "-------------------------------------------------------\n";
    std::cout << "Current yaw: " << utils::GetYawFromOrientation(current_pose_.orientation) << std::endl;
    std::cout << "Path Point Counts: " << global_path.poses.size() << std::endl;

    // 路径终点
    const geometry_msgs::Point &goal_pose = global_path.poses.back().pose.position;                                                      

    // 循环 MPC 求解
    int point_counts = global_path.poses.size();
    int index = 0;
    int unchanged_count = 0;                  
    const int kNextPointUnchangedhTreshold = 10;  // 确保机器人不会卡住
    while (ros::ok() && index < point_counts) {
        // 从当前机器人的位置找到最近的路径点，同时考虑 yaw 的变化最小
        int min_index = index;
        double min_cost = std::numeric_limits<double>::max();
        const double yaw_weight = 0.5; // 可以根据实际情况调整权重

        for (int i = min_index; i < index + kNum_knots && i < point_counts; ++i) {
            double distance = utils::CalculateDistance(current_pose_.position, global_path.poses[i].pose.position);
            double ref_yaw = utils::GetYawFromOrientation(global_path.poses[i].pose.orientation);
            double current_yaw = utils::GetYawFromOrientation(current_pose_.orientation);
            double yaw_diff = std::abs(utils::NormalizeAngle(current_yaw - ref_yaw));

            // 计算复合成本
            double cost = distance * distance + yaw_weight * yaw_diff * yaw_diff;
            if (cost < min_cost) {
                min_cost = cost;
                min_index = i;
            }
        }

        if (min_index == index) {
            unchanged_count ++;
        }
        else {
            unchanged_count = 0;
        }

        // 更新 index
        index = min_index;
        
        // 确保机器人不会卡住
        if (unchanged_count >= kNextPointUnchangedhTreshold) {
            unchanged_count = 0;
            if(index < point_counts) index ++;
            else {
                std::cout << "Unable to find a path to the next point. Aborting...\n";
                break;
            }
        }

        // 到达终点 
        // FIXME: 优化终点到达判断条件
        if(0.1 > utils::CalculateDistance(goal_pose, current_pose_.position) 
            && index >= point_counts - 2) {
            break;
        }

        // 添加终点减速逻辑
        double distance_to_goal = utils::CalculateDistance(goal_pose, current_pose_.position);
        double deceleration_distance = 1.0; // 开始减速的距离阈值，可根据实际情况调整
        double velocity_scale = 1.0;

        if (distance_to_goal < deceleration_distance) {
            // 根据距离终点的远近线性减速
            velocity_scale = std::max(0.3, distance_to_goal / deceleration_distance);
            std::cout << "Approaching goal, distance: " << distance_to_goal 
                      << ", velocity scale: " << velocity_scale << std::endl;
        }

        std::cout << "index:" << index << "\n";
        std::cout << "next point unchanged count:" << unchanged_count << "\n";

        // 转换参考轨迹到base_link坐标系
        int num_remaining_points = point_counts - index;
        double current_yaw = utils::GetYawFromOrientation(current_pose_.orientation);
        Eigen::VectorXd ref_x(kNum_knots), ref_y(kNum_knots), ref_yaw(kNum_knots);
        for (int i = 0; i < kNum_knots; ++i) {
            int global_index = (index + i) % point_counts;

            // 如果剩余的路径点数少于kNum_knots，则使用最后一个路径点进行补充
            if (i >= num_remaining_points) {
                global_index = point_counts - 1;
            }

            geometry_msgs::Point local_point;
            TransformToBaseLink(current_pose_, global_path.poses[global_index].pose.position, local_point);
            ref_x[i] = local_point.x;
            ref_y[i] = local_point.y;
            ref_yaw[i] = utils::GetYawFromOrientation(global_path.poses[global_index].pose.orientation) - current_yaw;
            ref_yaw[i] = utils::NormalizeAngle(ref_yaw[i]); // 确保yaw在[-pi, pi]范围内
            // std::cout  <<"ref_x: " << ref_x[i] << ", ref_y: " << ref_y[i] << ", ref_yaw: " << ref_yaw[i] << std::endl;
        }
        
        // 定义优化问题
        MathematicalProgram prog;

        // 定义优化变量
        auto x = prog.NewContinuousVariables(kNum_knots, "x");
        auto y = prog.NewContinuousVariables(kNum_knots, "y");
        auto yaw = prog.NewContinuousVariables(kNum_knots, "yaw");
        auto v = prog.NewContinuousVariables(kNum_knots, "v");
        auto yaw_dot = prog.NewContinuousVariables(kNum_knots, "yaw_dot");

        // 初始约束
        prog.AddBoundingBoxConstraint(0.0, 0.0, x[0]);
        prog.AddBoundingBoxConstraint(0.0, 0.0, y[0]);
        prog.AddBoundingBoxConstraint(0.0, 0.0, yaw[0]);

        // 终止约束 不再需要，因为我们只关注局部最优

        // 速度限幅约束
        for (int i = 0; i < kNum_knots; ++i) {
            prog.AddBoundingBoxConstraint(-v_max_linear_, v_max_linear_, v[i]);
            prog.AddBoundingBoxConstraint(-v_max_angular_, v_max_angular_, yaw_dot[i]);
        }

        // 动力学约束
        for (int i = 0; i < kNum_knots - 1; ++i) {
            prog.AddConstraint(x[i + 1] == x[i] + v[i] * cos(yaw[i]) * dt);
            prog.AddConstraint(y[i + 1] == y[i] + v[i] * sin(yaw[i]) * dt);
            prog.AddConstraint(yaw[i + 1] == yaw[i] + yaw_dot[i] * dt);
        }

        // 代价函数
        for (int i = 0; i < kNum_knots; ++i) {
            prog.AddQuadraticCost(pow(x[i] - ref_x[i], 2) + pow(y[i] - ref_y[i], 2));
            prog.AddQuadraticCost(0.5 * pow(yaw[i] - ref_yaw[i], 2));
            prog.AddQuadraticCost(0.1 * pow(v[i], 2));       // 速度惩罚项
            prog.AddQuadraticCost(0.15 * pow(yaw_dot[i], 2)); // 角速度惩罚项
        }

        // 求解优化问题
        SnoptSolver solver;
        MathematicalProgramResult result = solver.Solve(prog);

        if (result.is_success()) {
            auto optimal_x = result.GetSolution(x);
            auto optimal_y = result.GetSolution(y);
            auto optimal_yaw = result.GetSolution(yaw);
            auto optimal_v = result.GetSolution(v);
            auto optimal_yaw_dot = result.GetSolution(yaw_dot);

            // 应用速度缩放因子
            double scaled_v = optimal_v(0) * velocity_scale;
            double scaled_yaw_dot = optimal_yaw_dot(0) * velocity_scale;
            
            // 发布控制命令
            PublishCmdVel(scaled_v, scaled_yaw_dot);

            // 发布MPC局部路径
            PublishMpcPath(optimal_x, optimal_y, optimal_yaw);
        } else {
            auto sr = result.get_solution_result();
            std::cerr << "Optimization failed to find a solution: " << sr << std::endl;
            break;
        }

        // 发布固定路径
        PublishPath(global_path);
        ros::spinOnce();
        std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(dt*1000)));
    }

    ros::Duration(1.5).sleep();
    ROS_INFO("Path Follower Finished!");
    std::cout << "current pose:\n" << current_pose_ << std::endl;
}

void MpcPathTracer::PublishMpcPath(const Eigen::VectorXd& x, const Eigen::VectorXd& y, const Eigen::VectorXd& yaw)
{
    // 创建一个路径消息
    nav_msgs::Path mpc_path;
    mpc_path.header.stamp = ros::Time::now();
    mpc_path.header.frame_id = "odom";  // 假设路径是在 odom 坐标系下

    // 获取当前机器人的姿态
    tf::Quaternion q(current_pose_.orientation.x, current_pose_.orientation.y, current_pose_.orientation.z, current_pose_.orientation.w);
    tf::Matrix3x3 rot(q);

    // 获取当前机器人的偏航角
    double current_yaw = utils::GetYawFromOrientation(current_pose_.orientation);

    // 遍历所有的优化解
    for (int i = 0; i < kNum_knots; ++i) {
        // 将局部坐标系下的点转换为全局坐标系
        tf::Vector3 local_vec(x[i], y[i], 0.0);
        tf::Vector3 global_vec = rot * local_vec + tf::Vector3(current_pose_.position.x, current_pose_.position.y, 0.0);

        // 创建一个新的路径点
        geometry_msgs::PoseStamped pose_stamped;
        pose_stamped.header = mpc_path.header;
        pose_stamped.pose.position.x = global_vec[0];
        pose_stamped.pose.position.y = global_vec[1];
        pose_stamped.pose.position.z = 0.3;

        // 将局部偏航角转换为全局偏航角
        double global_yaw = yaw[i] + current_yaw;

        // 将全局偏航角设置到姿态中
        tf::Quaternion global_q;
        global_q.setRPY(0, 0, global_yaw);  // 设置偏航角
        tf::quaternionTFToMsg(global_q, pose_stamped.pose.orientation);

        // 添加到路径消息中
        mpc_path.poses.push_back(pose_stamped);
    }

    // 发布路径
    mpc_path_pub_.publish(mpc_path);
}

} // namespace trace_path