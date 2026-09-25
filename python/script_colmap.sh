#!/bin/bash
# TARGET=5
TRY=1
row=3
TARGET=35
# 2 4 5
# for TARGET in 1 2 3
# do	
    images_path=~/tesina/AgriColMap/maps/frames-RosarioV2-row${row}/rgb${TARGET}/
    colmap_results_path=~/tesina/frame-${row}-rgb${TARGET}-${TRY}/


    rm -rf OpenMVS_Ubuntu_x64/dense${row}${TARGET}${TRY}
    rm -rf ${colmap_results_path}
    python3 colmap.py ${images_path} ${colmap_results_path}
    # rm -rf OpenMVS_Ubuntu_x64/dense-${row}-${TARGET}-${TRY}
    colmap image_undistorter --image_path ${images_path} --input_path ${colmap_results_path}/sparse/0 --output_path OpenMVS_Ubuntu_x64/dense${row}${TARGET}${TRY} --output_type COLMAP
    cd OpenMVS_Ubuntu_x64
    rm -rf depth* 
    rm -rf *.log
    rm -rf scene${row}${TARGET}${TRY}.mvs scene${row}${TARGET}${TRY}_dense.ply  scene${row}${TARGET}${TRY}_dense.mvs 
    ./InterfaceCOLMAP -i dense${row}${TARGET}${TRY} -o scene${row}${TARGET}${TRY}.mvs --image-folder ${images_path}
    ./DensifyPointCloud scene${row}${TARGET}${TRY}.mvs --archive-type 2
    rm -rf depth*
    cd ..
# done
