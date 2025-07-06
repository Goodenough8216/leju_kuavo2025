cmake_minimum_required(VERSION 3.0.0)
project(SBUS_SDK VERSION 0.1.0)

set(CMAKE_BUILD_TYPE "Relese")

add_library(${PROJECT_NAME} STATIC src/drivers_sbus.c)

target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/src)

add_executable(kuavo_remote main.c)

target_link_libraries(kuavo_remote ${PROJECT_NAME})
