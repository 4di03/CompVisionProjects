#[[
* Adithya Palle
* Feb 15 2025
* CS 5330 - Project 3 : Real-time 2D Object Recognition
* 
* Cmake file which describes how to generate the build system for the object recognition project. This needs to be run before make only when changes are made to this file.
]]
cmake_minimum_required(VERSION 3.10)

project(ObjectRecognition)

# turn on debug mode
#set(CMAKE_BUILD_TYPE Debug)

# Set C++ Standard
set(CMAKE_CXX_STANDARD 17)   # Change to 14, 17, or 20 as needed
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) # Use strict standard compliance

# opencv setup
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# create a library for object_recognition
add_library(object_recognition objectRecognition.cpp kmeans.cpp utils.cpp classify.cpp)

# Ensure object_recognition's headers are available to dependent targets
target_include_directories(object_recognition PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(object_recognition PUBLIC ${OpenCV_LIBS})

add_executable(image_or image_or.cpp)
add_executable(image_dir_or image_dir_or.cpp)
add_executable(real_time_or real_time_or.cpp)
add_executable(classify runClassification.cpp)
add_executable(compareDistanceMetrics compareDistanceMetrics.cpp)
add_executable(test_cleanup testGrassfire.cpp)

target_link_libraries(image_dir_or PRIVATE object_recognition)
target_link_libraries(image_or PRIVATE object_recognition)
target_link_libraries(real_time_or PRIVATE object_recognition)
target_link_libraries(classify PRIVATE object_recognition)
target_link_libraries(compareDistanceMetrics PRIVATE object_recognition)
target_link_libraries(test_cleanup PRIVATE object_recognition)


