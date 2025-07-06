#include <iostream>
#include <vector>
#include <drake/common/eigen_types.h>
#include <drake/common/trajectories/piecewise_polynomial.h>

#pragma once

namespace GrabBox
{
  class CubicInterpolator
  {
    public:
      CubicInterpolator(){}
      CubicInterpolator(const std::vector<double>& t_values, const std::vector<Eigen::VectorXd>& pos)
      : t_(t_values)
      {
        // check input size
        if (t_values.size()!= pos.size()) {
          std::cerr << "Error: t_values and pos should have the same size." << std::endl;
        }
        int num = pos[0].size();
        std::vector<Eigen::MatrixXd> y_values;//transform pos to y_values
        for (int i = 0; i < pos.size(); i++) {
          y_values.push_back(pos[i]);
        }

        interpolant_ = drake::trajectories::PiecewisePolynomial<double>::CubicWithContinuousSecondDerivatives(
            t_values, y_values, Eigen::VectorXd::Zero(num), Eigen::VectorXd::Zero(num));
      }

      /**
       * @brief add quaternion to the interpolator
       * @param quat the quaternion to add
       */
      void addQuaternion(const std::vector<Eigen::Quaterniond>& quat)
      {
        //check input size
        if (t_.size()!= quat.size()) {
          std::cerr << "Error: t_values and quat should have the same size." << std::endl;
        }
        quat_ = quat;
      }

      /**
       * @brief add force
       */
      void addForce(const std::vector<Eigen::Vector3d>& force)
      {
        force_ = force;
      }

      /**
       * @brief get the position at time t
       * @param t the time to get the position
       */
      Eigen::VectorXd getPos(double t) const
      {
        t = std::min(t, t_.back()); // make sure t is within the range of t_values
        return interpolant_.value(t);
      }

      /**
       * @brief Get the velocity at time t
       * @param t The time to get the velocity
       */
      Eigen::VectorXd getVel(double t) const
      {
        t = std::min(t, t_.back()); // make sure t is within the range of t_values
        return interpolant_.derivative(1).value(t); // 一阶导数即速度
      }

      /**
       * @brief Get the acceleration at time t
       * @param t The time to get the acceleration
       */
      Eigen::VectorXd getAcc(double t) const
      {
        t = std::min(t, t_.back()); // make sure t is within the range of t_values
        return interpolant_.derivative(2).value(t); // 二阶导数即加速度}
      }

      /**
       * @brief get the quaternion at time t
       * @param t the time to get the quaternion
       */
      Eigen::Quaterniond getQuat(double t) const
      {
        if (quat_.empty()) {
          std::cerr << "Error: quaternion is not added." << std::endl;
          return Eigen::Quaterniond::Identity();
        }
        // find the index of the closest t value
        int index = t_.size() - 1;
        for (int i = 0; i < t_.size(); i++) {
          if (t_[i] > t) {
            index = i - 1;
            break;
          }
        }
        if(index == (t_.size() - 1))// if t >= the last time, use the last quaternion
          return quat_.back();
        // interpolate quaternion
        Eigen::Quaterniond q1 = quat_[index];
        Eigen::Quaterniond q2 = quat_[index + 1];
        double t1 = t_[index];
        double t2 = t_[index + 1];
        double alpha = (t - t1) / (t2 - t1);
        return q1.slerp(alpha, q2);
      }

      /**
       * @brief get the force at time t
       * @param t the time to get the quaternion
       */
      Eigen::Vector3d getForce(double t) const
      {
        if (force_.empty()) {
          std::cerr << "Error: force is not added." << std::endl;
          return Eigen::Vector3d::Zero();
        }
        // find the index of the closest t value
        int index = t_.size() - 1;
        for (int i = 0; i < t_.size(); i++) {
          if (t_[i] > t) {
            index = i - 1;
            break;
          }
        }
        if(index == t_.size() - 1)// if t is the last time, use the last quaternion
          return force_.back();
        // interpolate force
        Eigen::Vector3d f1 = force_[index];
        Eigen::Vector3d f2 = force_[index + 1];
        double t1 = t_[index];
        double t2 = t_[index + 1];
        double alpha = (t - t1) / (t2 - t1);
        auto f = f1 * (1 - alpha) + f2 * alpha;
        return f;
      }

      double getEndTime() const { return t_.back(); }

    private:
      drake::trajectories::PiecewisePolynomial<double> interpolant_;
      std::vector<double> t_;
      std::vector<Eigen::Quaterniond> quat_;
      std::vector<Eigen::Vector3d> force_;
  };
}// namespace GrabBox
