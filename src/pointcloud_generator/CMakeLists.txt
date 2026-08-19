add_library(pointcloud_generator_lib SHARED
  DensePointCloudGenerator.cpp
)

target_include_directories(pointcloud_generator_lib PRIVATE ${OpenCV_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS})
target_link_libraries(pointcloud_generator_lib ${OpenCV_LIBS} environment_representation_lib)
