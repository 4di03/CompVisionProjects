/**
 * Adithya Palle
 * Jan 24 2025
 * CS 5330 - Project 1 : Filter header
 * 
 * This file contains the declarations of the functions that are used to apply filters to the image.
 */
#include <opencv2/opencv.hpp>

// applys a custom grayscale filter to the image
int alternativeGrayscale(const cv::Mat& src, cv::Mat& dst);

int sepia(const cv::Mat& src, cv::Mat& dst);

int blur5x5_1( cv::Mat &src, cv::Mat &dst );

int blur5x5_2( cv::Mat &src, cv::Mat &dst );

int sobelX3x3( cv::Mat &src, cv::Mat &dst );
int sobelY3x3( cv::Mat &src, cv::Mat &dst );

int magnitude( cv::Mat &sx, cv::Mat &sy, cv::Mat &dst );

int blurQuantize( cv::Mat &src, cv::Mat &dst, int levels );
