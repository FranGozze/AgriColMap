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
    pclAligner.loadFixedCloudFromDisk("dron_points_xyz_cut", "RosarioV2Dron", fix_cloud );
    pclAligner.computeExGFilteredPointCloud(fix_cloud, Vector3i(255, 0, 0));
    
        
    float scaleMag = 0.f;
    float TranslNoiseMag = 0.f;
    float YawNoiseMag = 0.f;
    string ExpIDStr = "0";

    string suffix = argv[2];
    suffix = suffix == "0" ? "" : "_" + suffix;

    if (argc >= 4)
    {
        string matchingModeStr = argv[3];
        int matchingMode = stoi(matchingModeStr);
        pclAligner.setMatchingMode(matchingMode);
    }
    std::string row = argc >= 5 ? argv[4] : "2";
    int start = 1, end = 6;
    if(argc == 6)    
        end = std::atoi(argv[5]);
    if(argc == 7){
        start = std::atoi(argv[5]);
        end = std::atoi(argv[6]);
    }
    assert(start < end +1);
    // int end = argc >= 9 ? std::atoi(argv[8]) : 6;
    // blue, green, cyan, magenta, yellow, Amethyst Purple: (155, 89, 182)
    Vector3i colors[6] = {Vector3i(0, 0, 255), Vector3i(0, 255, 0), Vector3i(0, 255, 255), Vector3i(255, 0, 255), Vector3i(255, 255, 0), Vector3i(155, 89, 182)};
    std::string colorsString[6] = {"Blue", "Green", "Cyan", "Magenta", "Yellow", "Amethyst Purple"};
    for (int i = start; i < end + 1; i++)
    {   
        std::string number = std::to_string(i);
        std::string cloud_name = mov_cloud + "" + number;
        std::cerr << FRED("Computing " + number + " cloud \n") ;
        pclAligner.loadMovingCloudFromDisk("pointcloud_rgb"+number+suffix, "frames-RosarioV2-row"+row , cloud_name, fix_cloud, pclAligner.getScale(), "offset_rgb"+number+suffix);
        
        pclAligner.addNoise(cloud_name, scaleMag, TranslNoiseMag, YawNoiseMag);
        pclAligner.computeAndApplyInitialRelativeGuess(fix_cloud, cloud_name);
        // pclAligner.transformToInitialGuess(cloud_name);
        pclAligner.computeExGFilteredPointCloud(cloud_name, colors[(i-1)% 6]);
        pclAligner.computeExGFilteredPointCloud(fix_cloud, Vector3i(255, 0, 0));
        pclAligner.computeEnvironmentalModels(cloud_name, fix_cloud);
        pclAligner.Match(fix_cloud, cloud_name, pclAligner.getInitMovScale(), ExpIDStr, cv::Size(1300, 1300));
        
    }
    

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
            std::string cloud_name = mov_cloud + "" + number;
            std::cerr << FRED("Visualizing " + number + " cloud with color " + colorsString[(i-1) % 6] + " \n");
            viz.showCloud(pclAligner.getFilteredPcl(cloud_name), cloud_name);
        }
        viz.spingUntilDeath();
    }

    return 0;
}
