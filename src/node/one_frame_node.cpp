#include "../visualizer/visualizer.h"
#include "../pointcloud_generator/DensePointCloudGenerator.h"

using namespace std;

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
    string input_rgb = pointcloud_generator.getPackagePath() + "/maps/frames-RosarioV2-row2/1703261894400792837.png";
    string input_irL = pointcloud_generator.getPackagePath() + "/maps/frames-RosarioV2-row2/infra1_1703261894400734901.png";
    string input_irR = pointcloud_generator.getPackagePath() + "/maps/frames-RosarioV2-row2/infra2_1703261894400734901.png";
    string input_csv = pointcloud_generator.getPackagePath() + "/maps/frames-RosarioV2-row2/utm_jpg_final.csv";
    string input_timestamp_rgb = "1703261894400792837";
    cv::Mat rgb;
    cv::Mat irL;
    cv::Mat irR;
    rgb = cv::imread(input_rgb);
    irL = cv::imread(input_irL, cv::IMREAD_GRAYSCALE);
    irR = cv::imread(input_irR, cv::IMREAD_GRAYSCALE);
    std::cout << "Starting One Frame Node" << std::endl;
    auto pointcloud = pointcloud_generator.generate(rgb, irL, irR);
    utm utm_data = getUTMFromCSV(input_csv, input_timestamp_rgb);
    std::cout << "UTM Data: Easting: " << utm_data.easting << ", Northing: "  << utm_data.northing << ", Altitude: "<< utm_data.altitude << ", Zone: " << utm_data.zone << std::endl;
    pcl::io::savePLYFileBinary(pointcloud_generator.getPackagePath() + "/params/output/clouds/pointcloud_generated.ply", *pointcloud);

    // PointCloudViz viz;
    // viz.setViewerBackground(255, 255, 255);
    // viz.showCloud(pointcloud, "pointcloud", 1);
    // // viz.showCloud( pclAligner.getPcl(fix_cloud), fix_cloud);
    // // viz.showCloud( pclAligner.getPcl(mov_cloud), mov_cloud);
    // viz.spingUntilDeath();
    
    return 0;
}
