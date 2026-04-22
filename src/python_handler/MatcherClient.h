#include <zmq.hpp>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include <fstream>

// Base64 encoding (simple version)
#include <sstream>
#include <iterator>


#pragma once

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

struct MatchResult {
    std::vector<cv::Point2f> pts1;
    std::vector<cv::Point2f> pts2;
};

class MatcherClient {
public:
    MatcherClient(const std::string& address = "tcp://localhost:5555");

    MatchResult match(const cv::Mat& img1, const cv::Mat& img2, const int mode = 1);

private:
    std::string address;

    std::string encodeImage(const cv::Mat& img);
};

