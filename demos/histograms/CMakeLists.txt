cmake_minimum_required(VERSION 3.10)

project(Histogram)


add_executable(histogram makeHist.cpp)


# opencv setup
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


target_link_libraries(histogram ${OpenCV_LIBS})