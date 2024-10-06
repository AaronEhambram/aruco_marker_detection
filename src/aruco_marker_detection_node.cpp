#include "rclcpp/rclcpp.hpp"
#include "aruco_marker_detection/ArucoMarkerDetector.hpp"

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ArucoMarkerDetector>());
  rclcpp::shutdown();
  return 0;
}