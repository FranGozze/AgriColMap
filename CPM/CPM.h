/*

Code of the Coarse-to-Fine PatchMatch, published at CVPR 2016 in
"Efficient Coarse-to-Fine PatchMatch for Large Displacement Optical Flow"
by Yinlin.Hu, Rui Song and Yunsong Li.

Email: huyinlin@gmail.com

Version 1.2

Copyright (C) 2016 Yinlin.Hu

Usages:

The program "cpm.exe" has been built and tested on Windows 7.

USAGE: cpm.exe img1Name img2Name outMatchName

Explanations:

The output of the program is a text file, which is in the format of "x1,y1,x2,y2"
corresponding to one match per line.

*/

#ifndef _CPM_H_
#define _CPM_H_

#include "ImagePyramid.h"
#include <unordered_map>
#include <eigen3/Eigen/Geometry>

#ifdef BUILD_WITH_PCL
	#include "features/include/pcl/features/fpfh.h"
	#include "features/include/pcl/features/normal_3d.h"
	#include "features/include/pcl/features/fpfh_omp.h"
#else
	#include <pcl/features/fpfh.h>
	#include <pcl/features/normal_3d.h>
	#include <pcl/features/fpfh_omp.h>
#endif

#include "PythonClient.hpp"



class CPM
{
public:
    CPM();
	~CPM();

    void Match2Flow(FImage& inMat, FImage& ou, FImage& ov, int w, int h);
    int Matching(FImage& img1, FImage &img1Cloud, FImage& img1Soil, FImage& img2, FImage &img2Cloud, FImage& img2Soil, FImage& outMatches, int featureMode);
    void VotingSchemeHough(FImage& inpMatches, FImage& outMatches, const cv::Mat &rgb, const cv::Mat &rgb2);
    void VotingScheme(FImage& inpMatches, FImage& outMatches, const cv::Mat &rgb, const cv::Mat &rgb2);
    void addFlowToAccumulator(const Eigen::Vector2f& pt, cv::Mat& acc);
	void SetStereoFlag(int needStereo);
	void SetStep(int step);
    void SetParams(int step, bool useVisualFeatures, bool useGeometricFeatures, bool useSoilFeatures);
    void SetMatchingWeights(float vis_weight, float geom_weight, float soil_weight);


private:
		struct ImageFeatures{
		// IntImage _kLabels;
		FImagePyramid _pyd;
    	FImagePyramid _pyd_cloud;
		FImagePyramid _pyd_soil;
		UCImage* _im_exg;
		UCImage* _im_elev;
		UCImage* _im_soil;
		// FImage* _pydSeedsFlow;
		// IntImage _seeds;
		// IntImage _neighbors;	
	};

    void imDaisy(int level, ImageFeatures &imgF,  const float &cloud_ratio);
	void CrossCheck(IntImage& seeds, FImage& seedsFlow, FImage& seedsFlow2, IntImage& kLabel2, int* valid, float th);
    float MatchCost(ImageFeatures &imgF1, ImageFeatures &imgF2, int level, int x1, int y1, int x2, int y2);

	// a good initialization is already stored in bestU & bestV
    int Propogate(ImageFeatures &imgF1, ImageFeatures &imgF2, int level, float* radius, int iterCnt, IntImage* pydSeeds, IntImage& neighbors, FImage* pydSeedsFlow, float* bestCosts);
    void PyramidRandomSearch(ImageFeatures &imgF1, ImageFeatures &imgF2, IntImage* pydSeeds, IntImage& neighbors, FImage* pydSeedsFlow);
    void OnePass(ImageFeatures &imgF1, ImageFeatures &imgF2, IntImage& seeds, IntImage& neighbors, FImage* pydSeedsFlow);
	void UpdateSearchRadius(IntImage& neighbors, FImage* pydSeedsFlow, int level, float* outRadius);

    // Creating Pcl from cv::Mat
    void CreateXYZCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, const cv::Mat& orgCloud, cv::Mat& indexes);
    void NormalsAndFPFHEstimation(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::Normal>::Ptr& normals, pcl::PointCloud<pcl::FPFHSignature33>::Ptr &fpfh, const float& ratio);

	// minimum circle
	struct Point{
		double x, y;
	};
	double dist(Point a, Point b);
	Point intersection(Point u1, Point u2, Point v1, Point v2);
	Point circumcenter(Point a, Point b, Point c);
	// return the radius of the minimal circle
	float MinimalCircle(float* x, float*y, int n, float* centerX = NULL, float* centerY = NULL);

	//
    bool _useVisFeats;
    bool _useGeomFeats;
	bool _useSoilFeats;
	int _step;
	int _maxIters;
	float _stopIterRatio;
	float _pydRatio;

	int _isStereo;
	int _maxDisplacement;
	float _checkThreshold;
	int _borderWidth;


	ImageFeatures _img1Features, _img2Features;
	IntImage _kLabels, _kLabels2;
	int _featureMode;

	// FImagePyramid _pyd1;
    // FImagePyramid _pyd1_cloud;
	// FImagePyramid _pyd1_soil;
	// FImagePyramid _pyd2;
    // FImagePyramid _pyd2_cloud;
	// FImagePyramid _pyd2_soil;

    // UCImage* _im1_exg;
    // UCImage* _im1_elev;
	// UCImage* _im1_soil;
    // UCImage* _im2_exg;
    // UCImage* _im2_elev;
	// UCImage* _im2_soil;

	FImage* _pydSeedsFlow;
	FImage* _pydSeedsFlow2;

	IntImage _seeds;
	IntImage _seeds2;
	IntImage _neighbors;
	IntImage _neighbors2;

    float _vis_weight;
    float _geom_weight;
	float _soil_weight;


	PythonClient pythonClient;
	

};

#endif // _CPM_H_
