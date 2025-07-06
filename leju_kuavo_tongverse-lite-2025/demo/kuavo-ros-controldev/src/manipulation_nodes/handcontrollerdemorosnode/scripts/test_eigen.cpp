#include <Eigen/Geometry>
#include <iostream>
#include <Eigen/Geometry>

int main()
{
    // 定义欧拉角
    double roll = 90 * M_PI / 180; // Roll角
    double pitch = 0;              // Pitch角
    double yaw = 0;                // Yaw角

    // 创建AngleAxisd对象
    Eigen::AngleAxisd rollAngle(Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX()));
    Eigen::AngleAxisd pitchAngle(Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()));
    Eigen::AngleAxisd yawAngle(Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()));

    // 计算旋转矩阵
    Eigen::Matrix3d rotation_matrix = yawAngle * pitchAngle * rollAngle;

    // 打印旋转矩阵
    std::cout << "Rotation matrix:\n"
              << rotation_matrix << std::endl;

    return 0;
}
