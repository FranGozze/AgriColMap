add_library(cpm_lib CPM.cpp PythonClient.cpp)
target_link_libraries(cpm_lib ${OpenCV_LIBS})