cmake_minimum_required(VERSION 3.0.2)
project(kuavo_arm_joint_state_publisher_gui)

# Add a custom command to check and install required dependencies
execute_process(
  COMMAND bash -c "dpkg-query -W -f='${Status}' ros-$ROS_DISTRO-joint-state-publisher 2>/dev/null | grep 'install ok installed' || (echo 'Installing required dependency: ros-$ROS_DISTRO-joint-state-publisher' && sudo apt-get update && sudo apt-get install -y ros-$ROS_DISTRO-joint-state-publisher)"
  RESULT_VARIABLE INSTALL_RESULT
  OUTPUT_VARIABLE INSTALL_OUTPUT
)

message(STATUS "Dependency check result: ${INSTALL_OUTPUT}")
if(NOT INSTALL_RESULT EQUAL 0)
  message(WARNING "Dependency installation may have failed, please check and install ros-$ROS_DISTRO-joint-state-publisher manually")
endif()

find_package(catkin REQUIRED)
catkin_package()

catkin_python_setup()

catkin_install_python(PROGRAMS
  scripts/kuavo_arm_joint_state_publisher_gui
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
