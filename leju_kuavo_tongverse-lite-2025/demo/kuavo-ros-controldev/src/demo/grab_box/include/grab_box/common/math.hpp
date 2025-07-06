#include <Eigen/Dense>
#include <iostream>

#pragma once

namespace GrabBox
{
  Eigen::MatrixXd pseudoInverse(const Eigen::MatrixXd& A)
  {
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);

    // 获取 U, Sigma, V
    Eigen::MatrixXd U = svd.matrixU();
    Eigen::MatrixXd V = svd.matrixV();
    Eigen::VectorXd sigma = svd.singularValues();

    // 计算 Sigma 的伪逆
    Eigen::MatrixXd SigmaPlus = sigma.asDiagonal().inverse();

    // 计算 A 的伪逆
    Eigen::MatrixXd A_plus = V * SigmaPlus * U.transpose();
    return std::move(A_plus);
  }
} // namespace GrabBox
