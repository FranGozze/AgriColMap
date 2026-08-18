#include "../pointcloud_handler/pointcloud_aligner.h"
#include "../visualizer/visualizer.h"

using namespace std;

int main(int argc, char **argv) {

    if( argc < 2 ){
        std::cerr << FRED("ERROR: too few parameters\n");
        std::exit(1);
    }    
    string yaml_filename = argv[1];
    std::string mov_cloud = "moving_cloud";
    std::string fix_cloud = "fixed_cloud";
    string cloud_viz_type = argc == 3 ? argv[2] : "rgb";
    if( cloud_viz_type.compare("rgb") != 0 && cloud_viz_type.compare("exg") != 0 ){
        std::cerr << FRED("param ERROR: ") << " node_name <VIZ_TYPE = rgb/exg >\n";
        std::exit(1);
    }


    PointCloudAligner pclAligner;
    pclAligner.initFromYaml(yaml_filename);
    pclAligner.loadFromDisk(fix_cloud, mov_cloud);
    // Reading input Clouds
    std::cerr << FBLU("Transforming moving cloud!") << "\n";
    // Transforming Clouds with Grount Truth
    pclAligner.GroundTruthTransformPointCloud(mov_cloud);
    std::cerr << FBLU("Computing Exg Clouds!") << "\n";
    // Computing Filtered ExG Clouds
    pclAligner.computeExGFilteredPointCloud(fix_cloud, Vector3i(255,0,0) );
    pclAligner.computeExGFilteredPointCloud(mov_cloud, Vector3i(0,0,255) );


    /*pclAligner.getPointCloud("cloud")->downsamplePointCloud(0.04);
    pclAligner.getPointCloud("row3_cloud")->downsamplePointCloud(0.01);
    pclAligner.getPointCloud("row4_cloud")->downsamplePointCloud(0.01);
    pclAligner.getPointCloud("row5_cloud")->downsamplePointCloud(0.01);*/


    // Enhance Brightness for the UAV Cloud
    int brightness = 100;
    pclAligner.BrightnessEnhancement(fix_cloud, 75);
    
    // Visualize
    PointCloudViz viz;
    viz.setViewerBackground(255,255,255);

    if( cloud_viz_type.compare("exg") == 0 ){
        viz.showCloud( pclAligner.getFilteredPcl(fix_cloud), fix_cloud );
        viz.showCloud( pclAligner.getFilteredPcl(mov_cloud), mov_cloud );
    } else if ( cloud_viz_type.compare("rgb") == 0 ){
        viz.showCloud( pclAligner.getPcl(fix_cloud), fix_cloud );
        viz.showCloud( pclAligner.getPcl(mov_cloud), mov_cloud );
    }

    // viz.setViewerPosition(0,0,80,-1,0,0);
    viz.spingUntilDeath();

    return 0;
}
