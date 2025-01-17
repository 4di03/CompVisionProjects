cmake_minimum_required(VERSION 3.10)

project(Demos)

# opencv setup
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(fourierTransform fourierTransform.cpp)

target_link_libraries(fourierTransform ${OpenCV_LIBS})