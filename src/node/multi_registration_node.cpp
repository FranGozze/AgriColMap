#include "../pointcloud_handler/pointcloud_aligner.h"
#include "../visualizer/visualizer.h"

using namespace std;

void createFolder(const std::string &folder_path)
{
    if (!std::filesystem::exists(folder_path))
    {
        std::filesystem::create_directories(folder_path);
    }
}

int main(int argc, char **argv)
{

    if (argc < 6)
    {
        cerr << FRED("Other Params Expected!") << " node_name <params_file.txt> <Scale_Mag> <GPS_Noise_Mag> <Yaw_Noise_Mag> <Exp_ID>" << "\n";
        std::exit(1);
    }

    string yaml_filename = argv[1];
    std::string mov_cloud = "moving_cloud_rgb";
    // std::string mov_cloud2 = "moving_cloud_rgb2";
    // std::string mov_cloud3 = "moving_cloud_rgb3";

    std::string fix_cloud = "fixed_cloud";
    

    PointCloudAligner pclAligner;
    pclAligner.initFromYaml(yaml_filename);
    // pclAligner.loadFromDisk(fix_cloud, mov_cloud);
    pclAligner.loadFixedCloudFromDisk("dron_points_xyz", "RosarioV2Dron", fix_cloud );
    pclAligner.computeExGFilteredPointCloud(fix_cloud, Vector3i(255, 0, 0));
    
        
    string scaleMagStr = argv[2];
    float scaleMag = stof(scaleMagStr) / 100;
    string TranslNoiseMagStr = argv[3];
    float TranslNoiseMag = stof(TranslNoiseMagStr) / 100;
    string YawNoiseMagStr = argv[4];
    float YawNoiseMag = stof(YawNoiseMagStr) / 10;
    string ExpIDStr = argv[5];
    if (argc >= 7)
    {
        string matchingModeStr = argv[6];
        int matchingMode = stoi(matchingModeStr);
        pclAligner.setMatchingMode(matchingMode);
    }
    std::string row = argc >= 8 ? argv[7] : "2";
    int start = 1, end = 6;
    if(argc == 9)    
        end = std::atoi(argv[8]);
    if(argc == 10){
        start = std::atoi(argv[8]);
        end = std::atoi(argv[9]);
    }
    
    // int end = argc >= 9 ? std::atoi(argv[8]) : 6;
    // blue, green, cyan, magenta, yellow, Amethyst Purple: (155, 89, 182)
    Vector3i colors[6] = {Vector3i(0, 0, 255), Vector3i(0, 255, 0), Vector3i(0, 255, 255), Vector3i(255, 0, 255), Vector3i(255, 255, 0), Vector3i(155, 89, 182)};

    for (int i = start; i < end + 1; i++)
    {   
        std::string number = std::to_string(i);
        std::string cloud_name = mov_cloud + number;
        cerr << FRED("Computing " + number + " cloud \n") ;
        pclAligner.loadMovingCloudFromDisk("pointcloud_rgb"+number+"_20_imgs", "frames-RosarioV2-row"+row , cloud_name, fix_cloud, pclAligner.getScale(), "offset_rgb"+number+"_20_imgs");
        
        pclAligner.addNoise(cloud_name, scaleMag, TranslNoiseMag, YawNoiseMag);
        pclAligner.computeAndApplyInitialRelativeGuess(fix_cloud, cloud_name);
        pclAligner.computeExGFilteredPointCloud(cloud_name, colors[(i-1)% 6]);
        pclAligner.computeExGFilteredPointCloud(fix_cloud, Vector3i(255, 0, 0));
        pclAligner.computeEnvironmentalModels(cloud_name, fix_cloud);
        pclAligner.Match(fix_cloud, cloud_name, pclAligner.getInitMovScale(), ExpIDStr, cv::Size(1300, 1300));
        
    }
    
    // // Adding Noise to Initial Guess
    // pclAligner.loadMovingCloudFromDisk("pointcloud_rgb_23_imgs", "frames-RosarioV2-row2" , mov_cloud1, fix_cloud, Vector2(0.75f, 0.75f), "offset_rgb_23_imgs");
    // cerr << FRED("Computing first cloud \n") ;
    
    // // pclAligner.addNoise(mov_cloud1, scaleMag, TranslNoiseMag, YawNoiseMag);
    // pclAligner.computeAndApplyInitialRelativeGuess(fix_cloud, mov_cloud1);
    // pclAligner.computeExGFilteredPointCloud(mov_cloud1, Vector3i(0, 0, 255));
    // pclAligner.computeEnvironmentalModels(mov_cloud1, fix_cloud);
    // pclAligner.Match(fix_cloud, mov_cloud1, pclAligner.getInitMovScale(), ExpIDStr, cv::Size(1300, 1300));
    
    
    // pclAligner.loadMovingCloudFromDisk("pointcloud_rgb2_20_imgs", "frames-RosarioV2-row2", mov_cloud2, fix_cloud, Vector2(0.75f, 0.75f), "offset_rgb2_20_imgs");
    // cerr << FRED("Computing second cloud \n") ;
    // // pclAligner.addNoise(mov_cloud2, scaleMag, TranslNoiseMag, YawNoiseMag);
    // pclAligner.computeAndApplyInitialRelativeGuess(fix_cloud, mov_cloud2);
    // pclAligner.computeExGFilteredPointCloud(mov_cloud2, Vector3i(0, 255, 0));
    // // pclAligner.computeExGFilteredPointCloud(fix_cloud, Vector3i(255, 0, 0));
    // pclAligner.computeEnvironmentalModels(mov_cloud2, fix_cloud);
    // pclAligner.Match(fix_cloud, mov_cloud2, pclAligner.getInitMovScale(), ExpIDStr, cv::Size(1300, 1300));
    

    // pclAligner.loadMovingCloudFromDisk("pointcloud_rgb3_20_imgs", "frames-RosarioV2-row2", mov_cloud3, fix_cloud, Vector2(0.75f, 0.75f), "offset_rgb3_20_imgs");
    // cerr << FRED("Computing third cloud \n") ;
    // // pclAligner.addNoise(mov_cloud3, scaleMag, TranslNoiseMag, YawNoiseMag);
    // pclAligner.computeAndApplyInitialRelativeGuess(fix_cloud, mov_cloud3);
    // pclAligner.computeExGFilteredPointCloud(mov_cloud3, Vector3i(255, 255,0));
    // pclAligner.computeExGFilteredPointCloud(fix_cloud, Vector3i(255, 0, 0));
    // pclAligner.computeEnvironmentalModels(mov_cloud3, fix_cloud);
    // pclAligner.Match(fix_cloud, mov_cloud3, pclAligner.getInitMovScale(), ExpIDStr, cv::Size(1300, 1300));


    // if (pclAligner.saveRegisteredClouds())
    // {
    //     auto registered_cloud = pclAligner.getPcl(mov_cloud);
    //     *registered_cloud += *pclAligner.getPcl(fix_cloud);

        
    //     auto fix_soil_cloud = pclAligner.getSoilPcl(fix_cloud);
    //     auto mov_soil_cloud = pclAligner.getSoilPcl(mov_cloud);
    //     createFolder(pclAligner.getPackagePath() + "/params/output/clouds/" + pclAligner.getMovingCloudPath());
    //     pcl::io::savePLYFileBinary(pclAligner.getPackagePath() + "/params/output/clouds/" + pclAligner.getMovingCloudPath() + "/" + pclAligner.getMovingCloudPath() + "_Registered_" + to_string(scaleMag) + "_" + to_string(TranslNoiseMag) + "_" + to_string(YawNoiseMag) + "_" + pclAligner.getFeatureString() + "_" + ExpIDStr + ".ply", *registered_cloud);
    //     pcl::io::savePLYFileBinary(pclAligner.getPackagePath() + "/params/output/clouds/" + pclAligner.getMovingCloudPath() + "/" + pclAligner.getMovingCloudPath() + "_FixedSoil.ply", *fix_soil_cloud);
    //     pcl::io::savePLYFileBinary(pclAligner.getPackagePath() + "/params/output/clouds/" + pclAligner.getMovingCloudPath() + "/" + pclAligner.getMovingCloudPath() + "_MovingSoil.ply", *mov_soil_cloud);
    //     auto soil_cloud = pclAligner.getSoilPcl(mov_cloud);
    //     *soil_cloud += *pclAligner.getSoilPcl(fix_cloud);
    //     pcl::io::savePLYFileBinary(pclAligner.getPackagePath() + "/params/output/clouds/" + pclAligner.getMovingCloudPath() + "/" + pclAligner.getMovingCloudPath() + "_Soil_" + to_string(scaleMag) + "_" + to_string(TranslNoiseMag) + "_" + to_string(YawNoiseMag) + "_" + pclAligner.getFeatureString() + "_" + ExpIDStr + ".ply", *soil_cloud);
        
    // }

    if (pclAligner.cloudVisualizationEnabled())
    {
        PointCloudViz viz;
        viz.setViewerBackground(255, 255, 255);
        viz.showCloud(pclAligner.getFilteredPcl(fix_cloud), fix_cloud);
        for (int i = start; i < end + 1; i++)
        {   
            std::string number = std::to_string(i);
            std::string cloud_name = mov_cloud + number;
            viz.showCloud(pclAligner.getFilteredPcl(cloud_name), cloud_name);
        }
        // viz.showCloud(pclAligner.getFilteredPcl(mov_cloud2), mov_cloud2);
        // viz.showCloud(pclAligner.getFilteredPcl(mov_cloud3), mov_cloud3);
        // viz.showCloud( pclAligner.getPcl(fix_cloud), fix_cloud);
        // viz.showCloud( pclAligner.getPcl(mov_cloud), mov_cloud);
        viz.spingUntilDeath();
    }

    return 0;
}
