/**
 * Adithya Palle
 * Feb 7 2025
 * CS 5330 - Project 3 : Real-time 2D Object Recognition
 * 
 * Header file for the thresholding function used to segment the object.
 */
#include <opencv2/opencv.hpp>

// function that produces thresholded image for task 1 where the object's pixels are white and the background is black
cv::Mat segmentObjects(const cv::Mat& image);

