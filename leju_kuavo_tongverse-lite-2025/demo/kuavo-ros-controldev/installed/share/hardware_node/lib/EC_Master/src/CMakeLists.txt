cmake_minimum_required(VERSION 3.0.0)
project(EcDemo VERSION 0.1.0)

set(EcDemo_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/Common
    ${CMAKE_CURRENT_SOURCE_DIR}/SDK/INC 
    ${CMAKE_CURRENT_SOURCE_DIR}/SDK/INC/Linux
    ${CMAKE_CURRENT_SOURCE_DIR}/Sharelib/Common/ 
    ${CMAKE_CURRENT_SOURCE_DIR}/Sharelib/EcMasterDemo/
    ${CMAKE_CURRENT_SOURCE_DIR}/Sharelib/Linux/
    CACHE PATH "My include directories")

set(SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/Common/EcThread.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Common/EcTimer.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Sharelib/Common/EcDemoParms.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Sharelib/Common/EcLogging.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Sharelib/Common/EcNotification.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Sharelib/Common/EcSdoServices.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Sharelib/Common/EcSelectLinkLayer.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Sharelib/Common/EcSlaveInfo.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Sharelib/EcMasterDemo/EcDemoApp.cpp
)

set(LINK_LIB
${CMAKE_CURRENT_SOURCE_DIR}/SDK/LIB/Linux/x64/
${CMAKE_CURRENT_SOURCE_DIR}/../lib
)


## EcDemoMain
add_executable(${PROJECT_NAME} ${SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/Examples/Linux/EcDemoMain.cpp)
target_include_directories(${PROJECT_NAME} PUBLIC ${EcDemo_INCLUDE_DIRS})  
target_link_libraries(${PROJECT_NAME} 
${CMAKE_CURRENT_SOURCE_DIR}/SDK/LIB/Linux/x64/libAtemRasSrv.a
${CMAKE_CURRENT_SOURCE_DIR}/SDK/LIB/Linux/x64/libEcMaster.a
${CMAKE_CURRENT_SOURCE_DIR}/../lib/libemllI8254x.so
  -pthread
  -ldl
  # lcm
  # lcm-types-cpp
)

## ec_master_main
add_library(ec_master_main STATIC ${SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/Sharelib/Linux/EcDemoMain.cpp)
target_include_directories(ec_master_main PUBLIC ${EcDemo_INCLUDE_DIRS})  
target_link_libraries(ec_master_main
${CMAKE_CURRENT_SOURCE_DIR}/SDK/LIB/Linux/x64/libAtemRasSrv.a
${CMAKE_CURRENT_SOURCE_DIR}/SDK/LIB/Linux/x64/libEcMaster.a
${CMAKE_CURRENT_SOURCE_DIR}/../lib/libemllI8254x.so
  -pthread
  -ldl
  # lcm
  # lcm-types-cpp
)

## ec_app
add_library(ec_app STATIC ${CMAKE_CURRENT_SOURCE_DIR}/Sharelib/EcMasterDemo/EcDemoApp.cpp )
target_include_directories(ec_app PUBLIC ${EcDemo_INCLUDE_DIRS})  
target_link_libraries(${PROJECT_NAME} 
${CMAKE_CURRENT_SOURCE_DIR}/SDK/LIB/Linux/x64/libAtemRasSrv.a
${CMAKE_CURRENT_SOURCE_DIR}/SDK/LIB/Linux/x64/libEcMaster.a
${CMAKE_CURRENT_SOURCE_DIR}/../lib/libemllI8254x.so
  -pthread
  -ldl
  # lcm
  # lcm-types-cpp
)

install(TARGETS ec_master_main ec_app
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../lib/libemllI8254x.so
  DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
)

# set_target_properties(ec_master_main PROPERTIES IMPORTED_LOCATION ./EC_Master/src/libec_master_main.so)
# set_target_properties(ec_app PROPERTIES IMPORTED_LOCATION ./EC_Master/src/libec_app.so)

## 宏开关
# add_definitions(-DERROR_FIX) ## 快速报错修复开关