cmake_minimum_required(VERSION 3.10)

project(VideoSpecialEffects)

# turn on debug mode
set(CMAKE_BUILD_TYPE Debug)


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


add_executable(testFilter testFilter.cpp filter.cpp faceDetect/faceDetect.cpp )
add_executable(img_display imgDisplay.cpp)
add_executable(video_display vidDisplay.cpp filter.cpp faceDetect/faceDetect.cpp ${DEPTH_ANYTHING_DIR}/da2-code/DA2Network.hpp)
add_executable(timeBlur timeBlur.cpp filter.cpp faceDetect/faceDetect.cpp ${DEPTH_ANYTHING_DIR}/da2-code/DA2Network.hpp)

target_link_libraries(img_display ${OpenCV_LIBS})
target_link_libraries(testFilter ${OpenCV_LIBS} onnxruntime)
target_link_libraries(video_display ${OpenCV_LIBS} onnxruntime)
target_link_libraries(timeBlur ${OpenCV_LIBS} onnxruntime)