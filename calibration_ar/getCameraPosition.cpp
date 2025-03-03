/**
 * Adithya Palle
 * February 28, 2025
 * 
 * Main file for program displays camera's translation and rotation relative to chessboard in video stream.
 * Also visualizes the 3d axes of the worlc coordinate system on the chessboard.
 */
#include "calibration.h"
#define EXPECTED_FRAME_WIDTH 1280
#define EXPECTED_FRAME_HEIGHT 720




/**
 * Displays camera's translation and rotation relative to chessboard in video stream.
 * Also visualizes the 3d axes of the worlc coordinate system on the chessboard.
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

    if (!fs.isOpened()){
        std::cerr << "Error opening calibration yaml file" << std::endl;
        return -1;
    }

    fs["cameraMatrix"] >> cameraMatrix;

    if (cameraMatrix.empty()){
        std::cerr << "Error reading camera matrix from yaml" << std::endl;
        return -1;
    }

    fs["distCoeffs"] >> distCoeffs;

    if (distCoeffs.empty()){
        std::cerr << "Error reading distortion coefficients from yaml" << std::endl;
        return -1;
    }

    
    fs.release();

    cv::Mat distCoeffsMat(1, distCoeffs.size(), CV_64F, distCoeffs.data());

    std:: cout << "Camera Matrix: " << cameraMatrix << std::endl;
    std:: cout << "Distortion Coefficients: " << distCoeffsMat << std::endl;


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
            cv::solvePnP(worldPoints, corners, cameraMatrix, distCoeffsMat, rvec, tvec);

            // draw camera rotation and translation on frame , denoted with (Rx, Ry, Rz) and (Tx, Ty, Tz) respectively
            cv::putText(frame, "Rx: " + std::to_string(rvec.at<double>(0)) + ", Ry: " + std::to_string(rvec.at<double>(1)) + ", Rz: " + std::to_string(rvec.at<double>(2)) + ")", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
            cv::putText(frame, "Tx: " + std::to_string(tvec.at<double>(0)) + ", Ty: " + std::to_string(tvec.at<double>(1)) + ", Tz: " + std::to_string(tvec.at<double>(2)) + ")", cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);


            displayCoordinateAxes(frame, cameraMatrix, distCoeffsMat, rvec, tvec);
            
        }

        cv::imshow("Frame", frame);
        cv::waitKey(1);

    }
}