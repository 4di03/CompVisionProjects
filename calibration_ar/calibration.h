#include <opencv2/opencv.hpp>
#include <vector>
#define PATTERN_SIZE cv::Size(9, 6)


bool extractCorners(const cv::Mat& chessBoardImage, std::vector<cv::Point2f>& corners , const cv::Size& patternSize = PATTERN_SIZE);





