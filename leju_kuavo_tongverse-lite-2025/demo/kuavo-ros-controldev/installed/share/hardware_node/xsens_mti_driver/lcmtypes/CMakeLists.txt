
find_package(lcm REQUIRED)
include(${LCM_USE_FILE})

lcm_wrap_types(
  CPP_HEADERS cpp_headers
  lcm_std_msgs.lcm
  lcm_sensor_msgs.lcm
)

lcm_add_library(xsens-lcm-types-lib CPP ${cpp_headers})
target_include_directories(xsens-lcm-types-lib INTERFACE ${CMAKE_CURRENT_BINARY_DIR})
