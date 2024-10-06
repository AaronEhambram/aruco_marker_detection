#include "rclcpp/rclcpp.hpp"
#include <iostream>
#include "sensor_msgs/msg/image.hpp"
#include <opencv2/core.hpp>
#include "Eigen/Dense"
#include "visualization_msgs/msg/marker_array.hpp"

class ArucoMarkerDetector : public rclcpp::Node
{
  public: 
    explicit ArucoMarkerDetector();

  private:
    // camera data from calibration
    cv::Mat rgb_cam, rgb_distor, depth_cam, depth_distor, rgb_R_depth, rgb_t_depth;
    Eigen::Affine3d rgb_T_depth;

    // time synchronization
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr rgb_sub_;

    // output message
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr publisher_;

    // functions
    void cv2eigen(cv::Mat& R, cv::Mat& t, Eigen::Affine3d& T);
    void callback(const sensor_msgs::msg::Image::ConstSharedPtr& rgb_im_msg);

    // aruco marker detection
    cv::Mat marker_frame_points;
    double aruco_marker_length_;
    cv::Vec4d axis_angle2quaternion(const cv::Vec3d& axis_angle);
};
