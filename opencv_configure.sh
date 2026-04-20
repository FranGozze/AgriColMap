#!/bin/bash

git clone https://github.com/opencv/opencv.git 
git clone https://github.com/opencv/opencv_contrib.git 
cd opencv 
git checkout 4.6.0 
cd ../opencv_contrib 
git checkout 4.6.0 
cd ../opencv 
mkdir build 
cd build 
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DCMAKE_CXX_STANDARD=17 \
    -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
    -DBUILD_opencv_xfeatures2d=ON \
    -DWITH_FFMPEG=OFF \
    -DWITH_OPENJPEG=OFF
(echo "#include <cstdint>"; cat 3rdparty/ade/ade-0.1.1f/sources/ade/include/ade/typed_graph.hpp) > tmp.hpp 
 mv tmp.hpp 3rdparty/ade/ade-0.1.1f/sources/ade/include/ade/typed_graph.hpp 
 make -j$(nproc) 
 make install 
 ldconfig