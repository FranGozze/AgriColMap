#pragma once

#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include "../environment_model/environment_representation.h"
#include "../packageDir.h"
#include "./CameraCalibration.h"


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

  std::string inline getOutputPath()  { return getPackagePath() + output_path; }
  std::string inline getInputRGBPath()  { return getPackagePath() + input_rgb; }
  std::string inline getInputIRLPath()  { return getPackagePath() + input_irL; }
  std::string inline getInputIRRPath()  { return getPackagePath() + input_irR; }
  std::string inline getInputCSVPath()  { return getPackagePath() + input_csv; }
  std::string inline getInputTimestampRGB()  { return input_timestamp_rgb; }
  void inline setInputTimestampRGB(const std::string &timestamp)  { input_timestamp_rgb = timestamp; }
  void inline setMatrixTransformation(cv::Mat &M){transformationMatrix = M;}


private:
// ir_1 is right IR image, ir_2 is left IR image. The transformation from ir_2 to ir_1 is used to compute depth.
  
  CameraCalibration rgb_calib, irL_calib, irR_calib;

  cv::Mat T_ir2_ir1; // Transformation from IR2 to IR1
  cv::Mat T_ir2_rgb; // Transformation from IR2 to RGB
  
  double baseline_m_ = 0.1;
  double max_depth_ = 10.0;
  double min_depth_ = 0.1;

  std::string input_rgb, input_irR, input_irL, input_csv, input_timestamp_rgb;
  std::string output_path= "/params/output/clouds/pointcloud_generated.ply";

  cv::Mat transformationMatrix;
// rectified ir_left and ir_right
  cv::Mat R1, R2;
  cv::Mat P1, P2;
  cv::Mat Q;
// rectified ir_left and rgb
  cv::Mat R3, R4;
  cv::Mat P3, P4;
  cv::Mat Q2;

  cv::Mat l_mapx, l_mapy;
  cv::Mat rgb_mapx, rgb_mapy;

  double roi_width, roi_height;

};
