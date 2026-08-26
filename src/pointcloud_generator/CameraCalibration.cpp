#include "CameraCalibration.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

CameraCalibration::CameraCalibration() {}
CameraCalibration::~CameraCalibration() {}


intrinsics generateIntrinsics(std::vector<double> K_vec){
  if (K_vec.size() >= 4)
    {
      intrinsics intrinsics;
      intrinsics.fx = K_vec[0];
      intrinsics.fy = K_vec[1];
      intrinsics.cx = K_vec[2];
      intrinsics.cy = K_vec[3];
      return intrinsics;
    }
  else
    {
      std::cerr << "Error: K vector must have at least 4 elements." << std::endl;
      return intrinsics();
    }
}

cv::Mat generateDistortionCoeffs(std::vector<double> dist_vec){
  if (!dist_vec.empty())
    {
      cv::Mat dist = cv::Mat(1, static_cast<int>(dist_vec.size()), CV_64F);
      for (size_t i = 0; i < dist_vec.size(); ++i)
      {
        dist.at<double>(0, static_cast<int>(i)) = dist_vec[i];
      }
      return dist;
    }
  else
    {
      std::cerr << "Error: Distortion vector is empty." << std::endl;
      return cv::Mat();
    }
}

void CameraCalibration::setCalibration(std::vector<double> K_vec, std::vector<double> dist_vec){
    instrinsics = generateIntrinsics(K_vec);
    distortion_coeff = generateDistortionCoeffs(dist_vec);
}
