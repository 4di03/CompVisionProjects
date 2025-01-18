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

int blur5x5_2( cv::Mat &src, cv::Mat&dst );

int sobelX3x3( cv::Mat &src, cv::Mat &dst );
int sobelY3x3( cv::Mat &src, cv::Mat &dst );

int magnitude( cv::Mat &sx, cv::Mat &sy, cv::Mat &dst );

int blurQuantize( cv::Mat &src, cv::Mat &dst, int levels );

// produces a depth image from the input image
int depth(cv::Mat &src, cv::Mat &dst);


// uses depth information to apply an effect to only the foreground of an immage
int applyToForeground(cv::Mat &src, cv::Mat &dst, int threshold, int (*processingFunction)(const cv::Mat&, cv::Mat&));


// converts an image that may not be in the range [0, 255] to that range, also converts the image to 3 channel uchar
void prepareFrameForDisplay(cv::Mat& src, cv::Mat& dst);

// increases the brightness of the RGB image by adding delta to each channel
int adjustBrightness(cv::Mat& src, cv::Mat& dst, int delta);


// applies a median filter to the image
int medianFilter(cv::Mat &src, cv::Mat &dst);


// applies depth-based fog to the image
int depthFog(cv::Mat &src, cv::Mat &dst);