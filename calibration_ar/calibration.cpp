/**
 * Adithya Palle
 * March 3, 2025
 * 
 * Implementation file for camera calibration and relatred display functions
 */
#include "calibration.h"
#define WIN_SIZE 11 // size of search window for corner refinement algorithm


// Global 3D axis points (constant throughout the program)
std::vector<cv::Point3f> axisPoints = {
    cv::Point3f(0, 0, 0), // Origin
    cv::Point3f(1, 0, 0), // X-axis
    cv::Point3f(0, 1, 0), // Y-axis
    cv::Point3f(0, 0, 1)  // Z-axis
};


/**
 * Displays the 3D coordinate axes of the world coordinate system on the chessboard.
 * @param img The image to display the coordinate axes on.
 * @param cameraMatrix The camera matrix.
 * @param distCoeffsMat The distortion coefficients matrix.
 * @param rvec The rotation vector.
 * @param tvec The translation vector.
 */
void displayCoordinateAxes(cv::Mat& img,  const cv::Mat& cameraMatrix, const cv::Mat& distCoeffsMat, const cv::Mat&  rvec, const cv::Mat&  tvec){
    std::vector<cv::Point2f> imagePoints;
    cv::projectPoints(axisPoints, rvec, tvec, cameraMatrix, distCoeffsMat, imagePoints);

    // draw arrowed lines between the origin and the x, y, z axes
    cv::arrowedLine(img, imagePoints[0], imagePoints[1], cv::Scalar(0, 0, 255), 2); // x-axis is red
    cv::arrowedLine(img, imagePoints[0], imagePoints[2], cv::Scalar(0, 255, 0), 2); // y-axis is green
    cv::arrowedLine(img, imagePoints[0], imagePoints[3], cv::Scalar(255, 0, 0), 2); // z-axis is blue

    return;

}
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
    if (patternFound){
        // improve the found corners' coordinate accuracy for chessboard
        cv::Mat viewGray;
        cvtColor(chessBoardImage, viewGray, cv::COLOR_BGR2GRAY);
        cornerSubPix( viewGray, corners, cv::Size(WIN_SIZE,WIN_SIZE),
            cv::Size(-1,-1), cv::TermCriteria( cv::TermCriteria::EPS+cv::TermCriteria::COUNT, 30, 0.0001 ));
    }

    return patternFound;
}



/**
 * Calculates the 3D world points of the chessboard pattern given the pattern size of the chessboard.
 * Assumes each chessboard square is 1 square unit in area.
 * Note that this is the same for every list of corners of the same pattern size.
 * @param patternSize The size of the chessboard pattern.
 * @return The 3D world points of the chessboard pattern.
 */
std::vector<cv::Vec3f> calculateWorldPoints(const cv::Size& patternSize){

    int columns = patternSize.width;
    int rows = patternSize.height;
    std::vector<cv::Vec3f> worldPoints;
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < columns; j++){
            worldPoints.push_back(cv::Vec3f(j, -1*i, 0.0f)); // i becomes negative as negative side of y-axis is down if the z-axis is pointing towards the viewer
        }
    }

    return worldPoints;
}
