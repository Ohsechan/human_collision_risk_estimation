cmake_minimum_required(VERSION 3.8)
project(realsense_human_tracking)

# Default to C++14
#if(NOT CMAKE_CXX_STANDARD)
set(CMAKE_CXX_STANDARD 14)
#endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(cv_bridge REQUIRED)


# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

find_package(realsense_human_tracking REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/RiskScore.msg"
  "msg/RiskScoreArray.msg"
  "msg/PoseTracking.msg"
  "msg/PoseTrackingArray.msg"
)

add_executable(human_pcl src/human_pcl.cpp)
ament_target_dependencies( human_pcl
  realsense_human_tracking
  rclcpp
  std_msgs
  sensor_msgs
  OpenCV
  cv_bridge
)

install(TARGETS
 human_pcl
 DESTINATION lib/${PROJECT_NAME}
)

ament_package()
