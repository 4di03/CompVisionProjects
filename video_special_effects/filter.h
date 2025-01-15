/**
 * Adithya Palle
 * Jan 24 2025
 * CS 5330 - Project 1 : Filter header
 */
#include <opencv2/opencv.hpp>

// applys a custom grayscale filter to the image
int alternativeGrayscale(const cv::Mat& src, cv::Mat& dst);

int sepia(const cv::Mat& src, cv::Mat& dst);

int blur5x5_1( cv::Mat &src, cv::Mat &dst );

int blur5x5_2( cv::Mat &src, cv::Mat &dst );
