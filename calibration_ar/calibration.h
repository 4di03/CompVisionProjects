/**
 * Adithya Palle
 * March 3, 2025
 * 
 * Header file for camera calibration and relatred display functions
 */
#include
#include <opencv2/opencv.hpp>
#include <vector>
#define PATTERN_SIZE cv::Size(9, 6)


bool extractCorners(const cv::Mat& chessBoardImage, std::vector<cv::Point2f>& corners , const cv::Size& patternSize = PATTERN_SIZE);


std::vector<cv::Vec3f> calculateWorldPoints(const cv::Size& patternSize = PATTERN_SIZE);

void displayCoordinateAxes(cv::Mat& img,  const cv::Mat& cameraMatrix, const cv::Mat& distCoeffsMat, const cv::Mat&  rvec, const cv::Mat&  tvec);