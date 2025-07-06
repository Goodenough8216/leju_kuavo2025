cmake_minimum_required(VERSION 3.0.2)
project(gazebo_sim)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  gazebo_ros
  gazebo_ros_control
  controller_manager
  # joint_state_controller
  # robot_state_publisher
)

find_package(gazebo REQUIRED)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES 
    ${PROJECT_NAME}_shm
    ${PROJECT_NAME}_shm_interface

  CATKIN_DEPENDS
    roscpp
    gazebo_ros
    gazebo_ros_control
    controller_manager
    # joint_state_controller
    # robot_state_publisher
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${GAZEBO_INCLUDE_DIRS}
)

link_directories(${GAZEBO_LIBRARY_DIRS})



# 添加Gazebo插件
add_library(${PROJECT_NAME}_shm_interface
  src/gazebo_shm_interface.cpp
)

target_link_libraries(${PROJECT_NAME}_shm_interface
  ${PROJECT_NAME}_shm
  ${catkin_LIBRARIES}
  ${GAZEBO_LIBRARIES}
)

# 安装目标
install(TARGETS ${PROJECT_NAME}_shm ${PROJECT_NAME}_shm_interface
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
)

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h"
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
  FILES_MATCHING PATTERN "*.launch"
)

# 添加测试程序
add_executable(shm_test src/shm_test.cpp)
target_link_libraries(shm_test
  ${PROJECT_NAME}_shm
  ${catkin_LIBRARIES}
  rt
)

