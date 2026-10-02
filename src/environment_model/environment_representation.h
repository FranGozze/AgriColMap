#pragma once
#include "../../include/uav_ugv_collaboration_module/utils.h"

class EnvironmentRepresentation{

    public:

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        EnvironmentRepresentation(const std::string& cloudName);
        ~EnvironmentRepresentation(){}

        void loadFromPCLcloud(const PCLPointCloudXYZRGB::Ptr& pointCloud,
                              const float &square_size,
                              const Vector2& imgCenter = Vector2(0.f, 0.f),
                              const float& radius = 0.01,
                              const cv::Size& gridMapSize = cv::Size(1300,1300) );

        void computeMMGridMap();

        // GetFunctions()
        inline cv::Mat getRgbImg(){ return rgbImg; }
        inline cv::Mat getExgColorImg(){ return exgImgColor; };
        inline cv::Mat getXyzImgUchar(){ return xyzImgUChar; };
        inline cv::Mat getXyzImg(){ return xyzImg; };
        inline cv::Mat getelevImg(){ return xyzImg; };
        inline cv::Mat getExgImg(){ return exgImg; };
        inline cv::Mat getSoilImg(int mode = 0){ switch(mode){
            case 0: return soilImg_WE_3x3; // With Erosion 3x3
            case 1: return soilImg_WE_5x5; // With Erosion 5x5
            default: return soilImg_WE_3x3;
        } };

        inline float getXCoord(){ return x_coord; }
        inline float getYCoord(){ return y_coord; }

    private:

        PCLptXYZRGB computeAveragePoint(std::vector<PCLptXYZRGB> &ptVec,
                                        const unsigned int& col,
                                        const unsigned int& row);

        std::vector< std::vector<PCLptXYZRGB> > _gridMap;
        const std::string _cloudName;
        int _width, _height = 0;
        PCLptXYZRGB minPt, maxPt;
        float _square_size, x_coord, y_coord;
        cv::Mat exgImg, elevImg, xyzImg, xyzImgUChar, exgImgColor, rgbImg;
        cv::Mat soilImg_WE_3x3, soilImg_WE_5x5;
        int altitude_scale;
};
