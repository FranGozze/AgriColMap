#pragma once

#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include "../environment_model/environment_representation.h"
#include "../packageDir.h"


struct intrinsics {
  float fx;
  float fy;
  float cx;
  float cy;
};

class CameraCalibration {
public:
  CameraCalibration();
  ~CameraCalibration();

  void setCalibration(std::vector<double> K_vec, std::vector<double> dist_vec);
  intrinsics inline getIntrinsics(){return instrinsics;}
  cv::Mat inline getDistortionCoeff(){return distortion_coeff;}
  cv::Mat inline getK(){ return( cv::Mat_<double>(3,3) <<
                                  instrinsics.fx, 0.0, instrinsics.cx,
                                  0.0, instrinsics.fy, instrinsics.cy,
                                  0.0, 0.0, 1.0
                              );}

private:
  intrinsics instrinsics;
  cv::Mat distortion_coeff; 
};