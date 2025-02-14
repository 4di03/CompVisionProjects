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



add_executable(image_or image_or.cpp thresholding.cpp kmeans.cpp utils.cpp)
add_executable(image_dir_or image_dir_or.cpp thresholding.cpp kmeans.cpp utils.cpp classify.cpp)
add_executable(real_time_or real_time_or.cpp thresholding.cpp kmeans.cpp utils.cpp classify.cpp)
add_executable(classify runClassification.cpp classify.cpp utils.cpp thresholding.cpp kmeans.cpp)
add_executable(compareDistanceMetrics compareDistanceMetrics.cpp classify.cpp utils.cpp thresholding.cpp kmeans.cpp)

target_link_libraries(image_dir_or ${OpenCV_LIBS})
target_link_libraries(image_or ${OpenCV_LIBS})
target_link_libraries(real_time_or ${OpenCV_LIBS})
target_link_libraries(classify ${OpenCV_LIBS})
target_link_libraries(compareDistanceMetrics ${OpenCV_LIBS})


