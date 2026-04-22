add_library(MatcherClient SHARED MatcherClient.cpp)
target_link_libraries(MatcherClient utils_lib GoICP_lib zmq nlohmann_json::nlohmann_json)
