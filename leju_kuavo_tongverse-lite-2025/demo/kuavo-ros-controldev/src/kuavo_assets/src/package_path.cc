#include <iostream>
#include "kuavo_assets/include/package_path.h"
#include <ros/package.h>

namespace ocs2 {
namespace kuavo_assets {

std::string getPath() {
  return PROJECT_SOURCE_DIR;
}
std::string getAbsolutePathFromRosPackage(const std::string &package_name, const std::string &path)
{
    std::string package_path = ros::package::getPath(package_name);

    std::cout << "The path of package " << package_name << " is: " << package_path << std::endl;
    std::string abs_path = path;
    if (path[0] != '/')
    {
        abs_path = (package_path + "/" + path);
    }
    return abs_path;
}

}  // namespace robotic_assets
}  // namespace ocs2
