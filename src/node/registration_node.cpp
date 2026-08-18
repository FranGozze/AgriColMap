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
    std::string mov_cloud = "moving_cloud";
    std::string fix_cloud = "fixed_cloud";

    PointCloudAligner pclAligner;
    pclAligner.initFromYaml(yaml_filename);
    pclAligner.loadFromDisk(fix_cloud, mov_cloud);

    string scaleMagStr = argv[2];
    float scaleMag = stof(scaleMagStr) / 100;
    string TranslNoiseMagStr = argv[3];
    float TranslNoiseMag = stof(TranslNoiseMagStr) / 100;
    string YawNoiseMagStr = argv[4];
    float YawNoiseMag = stof(YawNoiseMagStr) / 10;
    string ExpIDStr = argv[5];
    if (argc == 7)
    {
        string matchingModeStr = argv[6];
        int matchingMode = stoi(matchingModeStr);
        pclAligner.setMatchingMode(matchingMode);
    }

    // Adding Noise to Initial Guess
    pclAligner.addNoise(mov_cloud, scaleMag, TranslNoiseMag, YawNoiseMag);
    pclAligner.computeAndApplyInitialRelativeGuess(fix_cloud, mov_cloud);
    pclAligner.computeExGFilteredPointCloud(mov_cloud, Vector3i(0, 0, 255));
    pclAligner.computeExGFilteredPointCloud(fix_cloud, Vector3i(255, 0, 0));
    pclAligner.computeEnvironmentalModels(mov_cloud, fix_cloud);

    pclAligner.Match(fix_cloud, mov_cloud, pclAligner.getInitMovScale(), ExpIDStr, cv::Size(1300, 1300));

    if (pclAligner.saveRegisteredClouds())
    {
        auto registered_cloud = pclAligner.getPcl(mov_cloud);
        *registered_cloud += *pclAligner.getPcl(fix_cloud);

        
        auto fix_soil_cloud = pclAligner.getSoilPcl(fix_cloud);
        auto mov_soil_cloud = pclAligner.getSoilPcl(mov_cloud);
        createFolder(pclAligner.getPackagePath() + "/params/output/clouds/" + pclAligner.getMovingCloudPath());
        pcl::io::savePLYFileBinary(pclAligner.getPackagePath() + "/params/output/clouds/" + pclAligner.getMovingCloudPath() + "/" + pclAligner.getMovingCloudPath() + "_Registered_" + to_string(scaleMag) + "_" + to_string(TranslNoiseMag) + "_" + to_string(YawNoiseMag) + "_" + pclAligner.getFeatureString() + "_" + ExpIDStr + ".ply", *registered_cloud);
        pcl::io::savePLYFileBinary(pclAligner.getPackagePath() + "/params/output/clouds/" + pclAligner.getMovingCloudPath() + "/" + pclAligner.getMovingCloudPath() + "_FixedSoil.ply", *fix_soil_cloud);
        pcl::io::savePLYFileBinary(pclAligner.getPackagePath() + "/params/output/clouds/" + pclAligner.getMovingCloudPath() + "/" + pclAligner.getMovingCloudPath() + "_MovingSoil.ply", *mov_soil_cloud);
        auto soil_cloud = pclAligner.getSoilPcl(mov_cloud);
        *soil_cloud += *pclAligner.getSoilPcl(fix_cloud);
        pcl::io::savePLYFileBinary(pclAligner.getPackagePath() + "/params/output/clouds/" + pclAligner.getMovingCloudPath() + "/" + pclAligner.getMovingCloudPath() + "_Soil_" + to_string(scaleMag) + "_" + to_string(TranslNoiseMag) + "_" + to_string(YawNoiseMag) + "_" + pclAligner.getFeatureString() + "_" + ExpIDStr + ".ply", *soil_cloud);
        
    }

    if (pclAligner.cloudVisualizationEnabled())
    {
        PointCloudViz viz;
        viz.setViewerBackground(255, 255, 255);
        viz.showCloud(pclAligner.getFilteredPcl(fix_cloud), fix_cloud);
        viz.showCloud(pclAligner.getFilteredPcl(mov_cloud), mov_cloud);
        // viz.showCloud( pclAligner.getPcl(fix_cloud), fix_cloud);
        // viz.showCloud( pclAligner.getPcl(mov_cloud), mov_cloud);
        viz.spingUntilDeath();
    }

    return 0;
}
