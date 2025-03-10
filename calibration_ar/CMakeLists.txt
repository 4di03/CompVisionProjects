#[[
* Adithya Palle
* Feb 15 2025
* CS 5330 - Project 4 : Calibration and Augmented Reality
* 
* Cmake file which describes how to generate the build system for the project. This needs to be run before make only when changes are made to this file.
]]
cmake_minimum_required(VERSION 3.10)

project(Calibration_AR)

# turn on debug mode
#set(CMAKE_BUILD_TYPE Debug)

# Set C++ Standard
set(CMAKE_CXX_STANDARD 17)  
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) 

# opencv setup
set(OpenCV_DIR "/Users/adithyapalle/libs/installs/lib/cmake/opencv4") # You can remove this line if you installed opencv with homebrew , or change it if you installed it elsewhere

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# create a library for object_recognition
add_library(calibration calibration.cpp)

# Ensure object_recognition's headers are available to dependent targets
target_include_directories(calibration PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(calibration PUBLIC ${OpenCV_LIBS})
add_executable(draw_corners drawCorners.cpp)
add_executable(calibrate runCalibration.cpp)
add_executable(get_camera_pos getCameraPosition.cpp)
add_executable(display_virtual_object displayVirtualObject.cpp)
add_executable(detect_features detectFeatures.cpp)

target_link_libraries(draw_corners PRIVATE calibration)
target_link_libraries(calibrate PRIVATE calibration)
target_link_libraries(get_camera_pos PRIVATE calibration)
target_link_libraries(display_virtual_object PRIVATE calibration)
target_link_libraries(detect_features PRIVATE calibration)
