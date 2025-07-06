#include <ros/package.h>
#include <string>
#include <iostream>

namespace HighlyDynamic
{
  std::string getPackagePath(const std::string& package_name) {
      std::string package_path;
      try {
          package_path = ros::package::getPath(package_name);
          return package_path;
      } catch (const std::runtime_error& e) {
          return ""; // 如果未找到包则返回空字符串
      }
  }
} // namespace HighlyDynamic
