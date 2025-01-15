cmake_minimum_required(VERSION 3.10)

project(VideoSpecialEffects)

# opencv setup
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(img_display imgDisplay.cpp)
add_executable(video_display vidDisplay.cpp filter.cpp)
add_executable(timeBlur timeBlur.cpp filter.cpp)


target_link_libraries(img_display ${OpenCV_LIBS})
target_link_libraries(video_display ${OpenCV_LIBS})
target_link_libraries(timeBlur ${OpenCV_LIBS})