/**
 * Adithya Palle
 * February 28, 2025
 * 
 * Main file for program displays camera's translation and rotation relative to chessboard in video stream.
 */
#include "calibration.h"
#define EXPECTED_FRAME_WIDTH 1280
#define EXPECTED_FRAME_HEIGHT 720

/**
 * displays camera's translation and rotation relative to chessboard in video stream.
 * @param argc The number of command line arguments.
 * @param argv The command line arguments. The first is the path to the calibration yaml file.
 */
int main(int argc, char** argv){

    if (argc != 2){
        std::cerr << "Usage: ./get_camera_pos <path_to_calibration_yaml>" << std::endl;
        return -1;
    }

    std::string calibrationYamlPath = argv[1];

    cv::VideoCapture cap(0);

    if (!cap.isOpened())
    {
        std::cerr << "Error opening video stream" << std::endl;
        return -1;
    }



    cv::Mat frame;

    int frameCols = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frameRows = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    if (frameCols != EXPECTED_FRAME_WIDTH || frameRows != EXPECTED_FRAME_HEIGHT){
        std::cerr << "Error: Camera resolution is not " << EXPECTED_FRAME_WIDTH << "x" << EXPECTED_FRAME_HEIGHT << std::endl;
        return -1;
    }


    // load camera matrix and distortion coefficients from yaml
    cv::Mat cameraMatrix;
    std::vector<double> distCoeffs;
    cv::FileStorage fs(calibrationYamlPath, cv::FileStorage::READ);
    fs["camera_matrix"] >> cameraMatrix;
    fs["distortion_coefficients"] >> distCoeffs;
    fs.release();

    while (true){
        cap >> frame;



        if (frame.empty())
        {
            std::cerr << "Error reading frame" << std::endl;
            return 1;
        }

        std::vector<cv::Point2f> corners;
        bool patternFound = extractCorners(frame, corners, PATTERN_SIZE);

        // use solvePNP to get the camera's translation and rotation relative to the chessboard

        if (patternFound)
        {
            std::vector<cv::Vec3f> worldPoints = calculateWorldPoints(PATTERN_SIZE);
            cv::Mat rvec, tvec;
            cv::solvePnP(worldPoints, corners, cameraMatrix, distCoeffs, rvec, tvec);
            cv::Mat rotationMatrix;
            cv::Rodrigues(rvec, rotationMatrix);
            cv::Mat cameraPosition = -rotationMatrix.t() * tvec;
            // draw camera position , rotation and translation on frame
            cv::putText(frame, "Camera Position: " + std::to_string(cameraPosition.at<double>(0, 0)) + ", " + std::to_string(cameraPosition.at<double>(1, 0)) + ", " + std::to_string(cameraPosition.at<double>(2, 0)), cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
            cv::putText(frame, "Camera Rotation: " + std::to_string(rvec.at<double>(0, 0)) + ", " + std::to_string(rvec.at<double>(1, 0)) + ", " + std::to_string(rvec.at<double>(2, 0)), cv::Point(10, 90), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
            cv::putText(frame, "Camera Translation: " + std::to_string(tvec.at<double>(0, 0)) + ", " + std::to_string(tvec.at<double>(1, 0)) + ", " + std::to_string(tvec.at<double>(2, 0)), cv::Point(10, 120), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
            
        }

}