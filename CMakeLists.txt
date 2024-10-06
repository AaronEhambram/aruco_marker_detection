cmake_minimum_required(VERSION 3.8)
project(aruco_marker_detection)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()


SET(CMAKE_BUILD_TYPE Release)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# find dependencies
set(OpenCV_DIR "/home/aaron/workspace/libraries/opencv-4.x/install/lib/cmake/opencv4")
find_package(OpenCV 4.10 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)
find_package(PCL REQUIRED)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
set(cv_bridge_DIR "/home/aaron/workspace/ros2_ws/install/cv_bridge/share/cv_bridge/cmake")
find_package(cv_bridge REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(visualization_msgs REQUIRED)

include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

add_executable(aruco_marker_detection_node src/aruco_marker_detection_node.cpp src/ArucoMarkerDetector.cpp)
ament_target_dependencies(aruco_marker_detection_node rclcpp std_msgs cv_bridge sensor_msgs message_filters visualization_msgs)
target_link_libraries(aruco_marker_detection_node ${OpenCV_LIBS} OpenMP::OpenMP_CXX)

install(TARGETS
  aruco_marker_detection_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()
