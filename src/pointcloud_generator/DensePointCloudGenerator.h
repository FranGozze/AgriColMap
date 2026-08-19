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

class DensePointCloudGenerator {
public:
  DensePointCloudGenerator();
  ~DensePointCloudGenerator();


  // Set stereo baseline in meters (used when two IR images are provided)
  void setBaseline(double baseline_m);
  std::string inline getPackagePath(){ return _package_path; }

  void initFromYaml(const std::string &yaml_file);

  // Generate dense pointcloud from rgb + one or two IR images.
  // If ir_right is empty, a single-image heuristic depth is used.
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr generate(const cv::Mat &rgb,
                                                  const cv::Mat &ir_left,
                                                  const cv::Mat &ir_right = cv::Mat());


private:
  intrinsics intrinsics_rgb, intrinsics_ir1, intrinsics_ir2;
  cv::Mat dist_coeff_rgb, dist_coeff_ir1, dist_coeff_ir2; 
  cv::Mat T_ir2_ir1; // Transformation from IR2 to IR1
  cv::Mat T_ir2_rgb; // Transformation from IR2 to RGB

  double baseline_m_ = 0.1;
  double max_depth_ = 10.0;
  double min_depth_ = 0.1;
};
