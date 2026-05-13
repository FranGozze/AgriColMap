#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include "ImageFeature.h"
#include "ImageIO.h"

#include <zmq.hpp>
#include <iostream>
#include <vector>
#include <nlohmann/json.hpp>
#include <fstream>

// Base64 encoding (simple version)
#include <sstream>
#include <iterator>




// struct MatchResult {
//     std::vector<cv::Point2f> pts1;
//     std::vector<cv::Point2f> pts2;
// };

class PythonClient {
public:
    PythonClient(const std::string& addr = "tcp://localhost:5555");

    void extract(
        const FImage& imgExg,
        const FImage& imgElev,
        float cloud_ratio,
        const int method,
        UCImage& outFtImg_Exg,
        UCImage& outFtImg_Elev
    );

    // MatchResult match(const FImage& img1, const FImage& img2, const int mode = 1);

private:
    std::string address;

    std::string encodeFImage(const FImage& img);
    std::string encodeImage(const cv::Mat& img);
};

