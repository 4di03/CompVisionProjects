#include "calibration.h"


/**
 * Extracts the corners of a chessboard pattern from an image.
 * @param chessBoardImage The image containing the chessboard pattern.
 * @param patternSize The size of the chessboard pattern.
 * @param corners vector of points which will be populated with the corners of the chessboard pattern.
 * @return True if the corners were found, false otherwise.
 */
bool extractCorners(const cv::Mat& chessBoardImage,  std::vector<cv::Point2f>& corners, const cv::Size& patternSize){
    int chessBoardFlags = cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE;
    bool patternFound = findChessboardCorners(chessBoardImage, patternSize, corners, chessBoardFlags);

    return patternFound;
}
