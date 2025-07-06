#include <drake/common/trajectories/piecewise_polynomial.h>
#include <vector>
#include <iostream>
#include <stdexcept>

// 定义专门处理 double 类型的插值器
class DrakeInterpolator {
public:
    // 构造函数，允许传入多个时间点和位置，以及起点和末端的速度
    DrakeInterpolator(const std::vector<double>& times,
                       const std::vector<double>& positions,
                       double start_vel = 0.0,
                       double end_vel = 0.0) {
        // 检查输入参数的合法性
        if (times.size() != positions.size()) {
            throw std::invalid_argument("The number of times must match the number of positions.");
        }

        // 起点和末端的速度
        start_vel_ = start_vel;
        end_vel_ = end_vel;

        // 生成轨迹
        generateTrajectory(times, positions);
    }

    // 获取某个时刻的位置
    double position(double time) const {
        return pos_traj_.value(time)(0, 0);
    }

    // 获取某个时刻的速度
    double velocity(double time) const {
        return vel_traj_.value(time)(0, 0);
    }

    // 获取某个时刻的加速度
    double acceleration(double time) const {
        return acc_traj_.value(time)(0, 0);
    }

private:
    // 起点和末端的速度
    double start_vel_;
    double end_vel_;

    // 轨迹
    drake::trajectories::PiecewisePolynomial<double> pos_traj_;
    drake::trajectories::PiecewisePolynomial<double> vel_traj_;
    drake::trajectories::PiecewisePolynomial<double> acc_traj_;

    // 生成轨迹
    void generateTrajectory(const std::vector<double>& times, const std::vector<double>& positions) {
        // 构建时间断点和位置样本
        std::vector<double> breaks = times;
        std::vector<Eigen::MatrixXd> pos_samples;
        for (const auto& pos : positions) {
            pos_samples.push_back(Eigen::VectorXd::Constant(1, pos)); // 将 double 转换为 1x1 的 Eigen 矩阵
        }

        // 使用三次多项式生成平滑轨迹
        pos_traj_ = drake::trajectories::PiecewisePolynomial<double>::CubicWithContinuousSecondDerivatives(
            breaks, pos_samples, Eigen::VectorXd::Constant(1, start_vel_), Eigen::VectorXd::Constant(1, end_vel_));

        // 计算速度和加速度轨迹
        vel_traj_ = pos_traj_.derivative();
        // acc_traj_ = vel_traj_.derivative();
    }
};
