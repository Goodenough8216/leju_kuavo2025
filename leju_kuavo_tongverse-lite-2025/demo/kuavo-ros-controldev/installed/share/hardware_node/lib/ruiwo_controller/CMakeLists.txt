cmake_minimum_required(VERSION 3.12)
project(ruiwoTest)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找 Python 库
find_package(Python3 3.8 EXACT COMPONENTS Interpreter Development NumPy REQUIRED)
# find_package(PythonInterp REQUIRED 3.8)
if(Python3_FOUND)
    message(STATUS "Found Python: ${Python3_EXECUTABLE}")
    message(STATUS "Python version: ${Python3_VERSION}")
    
else()
    message(FATAL_ERROR "Python not found")
endif()
add_library(ruiwo_actuatorLib SHARED ruiwo_actuator.cpp)
target_include_directories(ruiwo_actuatorLib PUBLIC ${Python3_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR})
target_link_libraries(ruiwo_actuatorLib PUBLIC Python3::Python stdc++fs -pthread)

# 查找 YAML-CPP 库
find_package(yaml-cpp REQUIRED)

target_include_directories(ruiwo_actuatorLib PUBLIC ${Python3_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR} yaml-cpp)
target_link_libraries(ruiwo_actuatorLib PUBLIC Python3::Python stdc++fs -pthread yaml-cpp)

# 添加可执行文件
add_executable(ruiwo_test ruiwo_test.cpp)
target_include_directories(ruiwo_test PRIVATE ${PROJECT_SOURCE_DIR})


# 链接 Python 库
target_link_libraries(ruiwo_test PUBLIC ruiwo_actuatorLib)

install(TARGETS ruiwo_actuatorLib
    ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY ./
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/lib/ruiwo_controller
    FILE_PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                   GROUP_EXECUTE GROUP_READ
                   WORLD_EXECUTE WORLD_READ
)


