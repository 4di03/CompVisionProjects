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

add_executable(img_retrieval src/find_similar_images.cpp src/distanceMetric.h src/featureExtractor.h)

target_link_libraries(img_retrieval ${OpenCV_LIBS})