cmake_minimum_required(VERSION 3.10)

project(ImageRetrieval)

# turn on debug mode
set(CMAKE_BUILD_TYPE Debug)

# Set C++ Standard
set(CMAKE_CXX_STANDARD 17)   # Change to 14, 17, or 20 as needed
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) # Use strict standard compliance

# opencv setup
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

#configure depthAnything
set(DEPTH_ANYTHING_DIR /Users/adithyapalle/work/CS5330/depthAnything)
set(ONNX_RUNTIME_DIR ${DEPTH_ANYTHING_DIR}/onnxruntime-osx-x86_64-1.20.1)
include_directories(${ONNX_RUNTIME_DIR}/include)
link_directories(${ONNX_RUNTIME_DIR}/lib)

message(STATUS "DEPTH_ANYTHING_DIR: ${DEPTH_ANYTHING_DIR}")

include_directories(${DEPTH_ANYTHING_DIR}/da2-code)


add_executable(img_retrieval get_matches.cpp find_similar_images.cpp distanceMetric.cpp featureExtractor.cpp ${DEPTH_ANYTHING_DIR}/da2-code/DA2Network.hpp)

add_executable(compare_methods compare_methods.cpp distanceMetric.cpp featureExtractor.cpp find_similar_images.cpp ${DEPTH_ANYTHING_DIR}/da2-code/DA2Network.hpp)

target_link_libraries(img_retrieval ${OpenCV_LIBS} onnxruntime)
target_link_libraries(compare_methods ${OpenCV_LIBS} onnxruntime)