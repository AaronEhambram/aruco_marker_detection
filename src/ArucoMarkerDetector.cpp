#include "aruco_marker_detection/ArucoMarkerDetector.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "cv_bridge/cv_bridge.hpp"
#include <opencv2/objdetect/aruco_detector.hpp>
#include <opencv2/calib3d.hpp>

ArucoMarkerDetector::ArucoMarkerDetector() : Node("aruco_marker_detection_node")
{
    this->declare_parameter<std::string>("calibration_file", "");
    std::string calibration_file = this->get_parameter("calibration_file").as_string();
    this->declare_parameter<std::string>("rgb_topic", "");
    std::string rgb_topic = this->get_parameter("rgb_topic").as_string();
    this->declare_parameter<double>("aruco_marker_length", 0.055);
    aruco_marker_length_ = this->get_parameter("aruco_marker_length").as_double();

    // Read calibration data
    cv::FileStorage calibration_data(calibration_file, cv::FileStorage::READ);
    calibration_data["rgb_camera_matrix"] >> rgb_cam;
    calibration_data["rgb_dist_coeff"] >> rgb_distor;
    calibration_data["ir_camera_matrix"] >> depth_cam;
    calibration_data["ir_dist_coeff"] >> depth_distor;
    calibration_data["rgb_R_ir"] >> rgb_R_depth;
    calibration_data["rgb_t_ir"] >> rgb_t_depth;
    cv2eigen(rgb_R_depth, rgb_t_depth, rgb_T_depth);
    calibration_data.release();
    std::cout << "rgb_T_depth: " << std::endl
              << rgb_T_depth.matrix() << std::endl;

    // subscribers
    rgb_sub_ = this->create_subscription<sensor_msgs::msg::Image>(rgb_topic, 10, std::bind(&ArucoMarkerDetector::callback, this, std::placeholders::_1));

    // publisher
    publisher_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("aruco_markers", 10);

    // aruco pose estimation
    marker_frame_points = cv::Mat(4, 1, CV_32FC3);
    marker_frame_points.ptr<cv::Vec3f>(0)[0] = cv::Vec3f(-aruco_marker_length_ / 2.f, aruco_marker_length_ / 2.f, 0);
    marker_frame_points.ptr<cv::Vec3f>(0)[1] = cv::Vec3f(aruco_marker_length_ / 2.f, aruco_marker_length_ / 2.f, 0);
    marker_frame_points.ptr<cv::Vec3f>(0)[2] = cv::Vec3f(aruco_marker_length_ / 2.f, -aruco_marker_length_ / 2.f, 0);
    marker_frame_points.ptr<cv::Vec3f>(0)[3] = cv::Vec3f(-aruco_marker_length_ / 2.f, -aruco_marker_length_ / 2.f, 0);
}

void ArucoMarkerDetector::cv2eigen(cv::Mat &R, cv::Mat &t, Eigen::Affine3d &T)
{
    T.matrix()(0, 0) = R.at<double>(0, 0);
    T.matrix()(1, 0) = R.at<double>(1, 0);
    T.matrix()(2, 0) = R.at<double>(2, 0);
    T.matrix()(0, 1) = R.at<double>(0, 1);
    T.matrix()(1, 1) = R.at<double>(1, 1);
    T.matrix()(2, 1) = R.at<double>(2, 1);
    T.matrix()(0, 2) = R.at<double>(0, 2);
    T.matrix()(1, 2) = R.at<double>(1, 2);
    T.matrix()(2, 2) = R.at<double>(2, 2);

    T.matrix()(0, 3) = t.at<double>(0);
    T.matrix()(1, 3) = t.at<double>(1);
    T.matrix()(2, 3) = t.at<double>(2);
}

cv::Vec4d ArucoMarkerDetector::axis_angle2quaternion(const cv::Vec3d &axis_angle)
{
    double angle = norm(axis_angle);
    cv::Vec3d axis(axis_angle(0) / angle, axis_angle(1) / angle, axis_angle(2) / angle);
    double angle_2 = angle / 2;
    cv::Vec4d q(axis(0) * sin(angle_2), axis(1) * sin(angle_2), axis(2) * sin(angle_2), cos(angle_2)); // qx, qy, qz, qw
    return q;
}

void ArucoMarkerDetector::callback(const sensor_msgs::msg::Image::ConstSharedPtr &rgb_im_msg)
{
    // get the image through the cv:bridge
    cv_bridge::CvImageConstPtr rgb_cv_ptr;
    try
    {
        rgb_cv_ptr = cv_bridge::toCvShare(rgb_im_msg, "bgr8");
    }
    catch (cv_bridge::Exception &e)
    {
        return;
    }

    cv::Mat im;
    rgb_cv_ptr->image.copyTo(im);

    // calibration data
    double &fx_rgb = rgb_cam.at<double>(0, 0);
    double &fy_rgb = rgb_cam.at<double>(1, 1);
    double &cx_rgb = rgb_cam.at<double>(0, 2);
    double &cy_rgb = rgb_cam.at<double>(1, 2);

    // detect aruco marker
    std::vector<int> marker_ids;
    std::vector<std::vector<cv::Point2f>> marker_corners, rejected_candidates;
    cv::aruco::DetectorParameters detector_params = cv::aruco::DetectorParameters();
    cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_100);
    cv::aruco::ArucoDetector detector(dictionary, detector_params);
    detector.detectMarkers(im, marker_corners, marker_ids, rejected_candidates);
    cv::aruco::drawDetectedMarkers(im, marker_corners, marker_ids);

    // determine pose
    std::vector<cv::Vec3d> rvecs(marker_corners.size()), tvecs(marker_corners.size());
    visualization_msgs::msg::MarkerArray visu_markers;
    if (!marker_ids.empty())
    {
        // Calculate pose for each marker
        for (size_t i = 0; i < marker_corners.size(); i++)
        {
            cv::solvePnP(marker_frame_points, marker_corners[i], rgb_cam, rgb_distor, rvecs[i], tvecs[i]);
            visu_markers.markers.emplace_back(visualization_msgs::msg::Marker{});
            visualization_msgs::msg::Marker &visu_marker = visu_markers.markers.back();
            visu_marker.header.stamp = rgb_im_msg->header.stamp;
            visu_marker.header.frame_id = "camera_rgb_optical_frame";
            visu_marker.ns = "aruco";
            visu_marker.id = marker_ids[i];
            visu_marker.type = visualization_msgs::msg::Marker::CUBE;
            visu_marker.action = visualization_msgs::msg::Marker::MODIFY;
            visu_marker.pose.position.x = tvecs[i][0];
            visu_marker.pose.position.y = tvecs[i][1];
            visu_marker.pose.position.z = tvecs[i][2];
            cv::Vec4d q = axis_angle2quaternion(rvecs[i]);
            visu_marker.pose.orientation.x = q(0);
            visu_marker.pose.orientation.y = q(1);
            visu_marker.pose.orientation.z = q(2);
            visu_marker.pose.orientation.w = q(3);
            visu_marker.scale.x = aruco_marker_length_;
            visu_marker.scale.y = aruco_marker_length_;
            visu_marker.scale.z = 0.005;
            visu_marker.color.r = 0.9;
            visu_marker.color.g = 1.0;
            visu_marker.color.b = 0.9;
            visu_marker.color.a = 0.7;
            visu_marker.lifetime.sec = 0;
            visu_marker.lifetime.nanosec = 0.5 * 1e9;
        }
    }

    /*for (unsigned int i = 0; i < marker_ids.size(); i++)
    {
        cv::drawFrameAxes(im, rgb_cam, rgb_distor, rvecs[i], tvecs[i], aruco_marker_length_ * 1.5f, 2);
    }*/

    // publish
    publisher_->publish(visu_markers);

    // show the image
    //cv::imshow("image", im);
    //cv::waitKey(1);
}