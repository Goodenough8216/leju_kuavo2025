# Target node
add_executable(handTrajectoryTest
handTrajectoryTest.cpp
)
add_dependencies(handTrajectoryTest
  ${catkin_EXPORTED_TARGETS}
)
target_link_libraries(handTrajectoryTest
  ${catkin_LIBRARIES}
)
target_compile_options(handTrajectoryTest PUBLIC ${FLAGS})
