#include "../visualizer/visualizer.h"
#include "../pointcloud_generator/DensePointCloudGenerator.h"

#include <vector>
#include <pcl/filters/statistical_outlier_removal.h>

using namespace std;

namespace fs = std::filesystem;

// Helper to grab and sort paths in one go
std::vector<fs::path> getSortedPaths(const fs::path& dir) {
    std::vector<fs::path> p;
    if (fs::exists(dir) && fs::is_directory(dir))
        for (const auto& entry : fs::directory_iterator(dir))
            if (entry.is_regular_file()) p.push_back(entry.path());
    std::sort(p.begin(), p.end());
    return p;
}

void createFolder(const std::string &folder_path)
{
    if (!std::filesystem::exists(folder_path))
    {
        std::filesystem::create_directories(folder_path);
    }
}

struct utm{
    
    string easting;
    string northing;
    string altitude;
    string zone;
};

utm getUTMFromCSV(const std::string &csv_file, const std::string &timestamp)
{
    std::ifstream file(csv_file);
    std::string line;
    utm result;

    if (!file.is_open())
    {
        throw std::runtime_error("Could not open file: " + csv_file);
    }

    while (std::getline(file, line))
    {
        std::istringstream ss(line);
        std::string token;
        std::vector<std::string> tokens;

        while (std::getline(ss, token, ','))
        {
            tokens.push_back(token);
        }

        if (tokens.size() >= 5 && tokens[0] == timestamp)
        {
            result.easting = tokens[1];
            result.northing = tokens[2];
            result.altitude = tokens[3];
            result.zone = tokens[4];
            return result;
        }
    }

    throw std::runtime_error("Timestamp not found in CSV: " + timestamp);
}



int main(int argc, char **argv)
{

    // if (argc < 2)
    // {
    //     cerr << FRED("Other Params Expected!") << " node_name <params_file.txt> <Scale_Mag> <GPS_Noise_Mag> <Yaw_Noise_Mag> <Exp_ID>" << "\n";
    //     std::exit(1);
    // }
    DensePointCloudGenerator pointcloud_generator;
    pointcloud_generator.initFromYaml(argv[1]);
    


    // string input_rgb = pointcloud_generator.getInputRGBPath();
    // string input_irR = pointcloud_generator.getInputIRRPath();
    // string input_irL = pointcloud_generator.getInputIRLPath();
    string input_csv = pointcloud_generator.getInputCSVPath();
    auto irR_f = getSortedPaths(pointcloud_generator.getInputIRRPath());
    auto irL_f = getSortedPaths(pointcloud_generator.getInputIRLPath());
    auto rgb_f = getSortedPaths(pointcloud_generator.getInputRGBPath());
    if (irR_f.size() != irL_f.size() || irR_f.size() != rgb_f.size()) {
        std::cerr << "Error: Folder sizes do not match!\n";
        return -1;
    }
    utm first_utm;
    auto globalCloud =pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
    
    size_t start = 0, end,step = 1;
    if(argc == 2){
        end = irR_f.size();
    }
    else if(argc == 3){
        end = static_cast<size_t>(std::stoull(argv[2]));
    }
    else if(argc >= 4) {
        start = static_cast<size_t>(std::stoull(argv[2]));
        end = static_cast<size_t>(std::stoull(argv[3]));
        if(argc == 5)
            step = static_cast<size_t>(std::stoull(argv[4]));
    }
    


// irR_f.size()
    for (size_t i = start; i < end; i+=step) {
        cv::Mat rgb;
        cv::Mat irL;
        cv::Mat irR;
        rgb = cv::imread(pointcloud_generator.getInputRGBPath() + rgb_f[i].filename().string());
        irL = cv::imread(pointcloud_generator.getInputIRLPath() + irL_f[i].filename().string(), cv::IMREAD_GRAYSCALE);
        irR = cv::imread(pointcloud_generator.getInputIRRPath() + irR_f[i].filename().string(), cv::IMREAD_GRAYSCALE);        
        pointcloud_generator.setInputTimestampRGB(rgb_f[i].filename().stem().string());
        std::cout << "Zipped: " << rgb_f[i].filename() << " | " 
                  << irL_f[i].filename() << " | " 
                  << irR_f[i].filename() << "\n";
        std::cout << "Generating PointCloud" << std::endl;

        utm utm_data = getUTMFromCSV(input_csv, pointcloud_generator.getInputTimestampRGB());
        if(i==0)
        first_utm = utm_data;
        std::cout << "UTM Data: Easting: " << utm_data.easting << ", Northing: "  << utm_data.northing << ", Altitude: "<< utm_data.altitude << ", Zone: " << utm_data.zone << std::endl;
        double x = static_cast<double>(std::stod(utm_data.easting)) - static_cast<double>(std::stod(first_utm.easting));
        double y = static_cast<double>(std::stod(utm_data.northing)) - static_cast<double>(std::stod(first_utm.northing));
        double z = static_cast<double>(std::stod(utm_data.altitude)) - static_cast<double>(std::stod(first_utm.altitude));
        cv::Mat m = cv::Mat::eye(4,4,CV_64F);
        m.at<double>(0, 3) = x;
        m.at<double>(1, 3) = y;
        m.at<double>(2, 3) = z;
        pointcloud_generator.setMatrixTransformation(m);
        auto pointcloud = pointcloud_generator.generate(rgb, irL, irR);
        *globalCloud += *pointcloud;

        // Use your files here safely and in order
        
    }

    std::cout << "Puntos antes de voxel: " << globalCloud->size() << std::endl;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr voxelCloud(new pcl::PointCloud<pcl::PointXYZRGB>);

    pcl::VoxelGrid<pcl::PointXYZRGB> voxel;

    voxel.setInputCloud(globalCloud);

    // 2 cm
    voxel.setLeafSize(0.005f,0.005f,0.005f);

    voxel.filter(*voxelCloud);

    std::cout << "Puntos despues de voxel: " << voxelCloud->size() << std::endl;

    // // ========================================================
    // // 11. ELIMINAR OUTLIERS
    // // ========================================================

    // pcl::PointCloud<pcl::PointXYZRGB>::Ptr filteredCloud(
    //     new pcl::PointCloud<pcl::PointXYZRGB>
    // );


    // pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;

    // sor.setInputCloud(voxelCloud);

    // sor.setMeanK(30);

    // sor.setStddevMulThresh(1.0);

    // sor.filter(*filteredCloud);

    // std::cout << "Puntos despues de outlier removal: " <<  filteredCloud->size() << std::endl;


    pcl::io::savePLYFileBinary(pointcloud_generator.getOutputPath(), *voxelCloud);
    
    std::cout << "File saved" << std::endl;
    

    // PointCloudViz viz;
    // viz.setViewerBackground(255, 255, 255);
    // viz.showCloud(pointcloud, "pointcloud", 1);
    // // viz.showCloud( pclAligner.getPcl(fix_cloud), fix_cloud);
    // // viz.showCloud( pclAligner.getPcl(mov_cloud), mov_cloud);
    // viz.spingUntilDeath();
    
    return 0;
}
