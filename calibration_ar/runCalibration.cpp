/**
 * Adithya Palle
 * February 28, 2025
 * 
 * Main file for program that saves calibration images and corner cooridnates in memory when the user presses 's' and later uses them for calibrating the camera.
 */
#include <sys/stat.h>
#include "calibration.h"
#include <cstdio>
#define NUM_DISTORTION_COEFFS 5


/**
 * Draws the (x,y,z) world points on the image frame.
 * @param frame The image frame to draw the world points on.
 * @param worldPoints The world points to draw on the image frame.
 * @param corners The corners of the chessboard pattern in the image frame.
 */
void annotateWorldPoints(cv::Mat& frame, const std::vector<cv::Vec3f>& worldPoints,const std::vector<cv::Point2f>& corners){

    for (int i = 0; i < worldPoints.size(); i++){
        cv::Vec3f worldPoint = worldPoints[i];
        cv::Point2f corner = corners[i];

        char text[50];  // Fixed-size buffer
        snprintf(text, sizeof(text), "(%.0f, %.0f, %.0f)", worldPoint[0], worldPoint[1], worldPoint[2]);
        
        cv::putText(frame, text, corner, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
        }
}

/**
 * Prints the elements of a vector to the console.
 * @param vec The vector to print.
 */
void printVector(const std::vector<double>& vec){
    if (vec.size() == 0){
        std::cout << "Empty vector" << std::endl;
        return;
    }
    for (int i = 0; i < vec.size(); i++){
        std::cout << vec[i] << " ";
    }
    std::cout << std::endl;
}

/**
 * saves calibration images and corner cooridnates in memory when the user presses 's' and later uses them for calibrating the camera.
 */
int main(){


    cv::VideoCapture cap(0);

    if (!cap.isOpened())
    {
        std::cerr << "Error opening video stream" << std::endl;
        return -1;
    }


    cv::Mat frame;
    int numCornersFound = 0;
    std::vector<cv::Point2f> corners; // most recent corners found
    std::vector<std::vector<cv::Point2f>> allCorners; // all corners found

    std::vector<cv::Vec3f> worldPoints; // most recent world points
    std::vector<std::vector<cv::Vec3f>> allWorldPoints; // all world points , TODO: figure out if we cna just use one set of worldPoints since they are always the same for the same pattern

    // Get frame dimensions without capturing
    int frameCols = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frameRows = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

    cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << 
    1, 0, frameCols / 2.0, 
    0, 1, frameRows / 2.0, 
    0, 0, 1);
    std::vector<double> distCoeffs = std::vector<double>(NUM_DISTORTION_COEFFS, 0);



    mkdir("calibration_images", 0777);
    while (true){
        cap >> frame;

        if (frame.empty())
        {
            std::cerr << "Error reading frame" << std::endl;
            return 1;
        }

        bool patternFound = extractCorners(frame, corners, PATTERN_SIZE);
        if (patternFound)
        {

            numCornersFound = corners.size();

            if (numCornersFound != PATTERN_SIZE.area()){
                std::cerr << "Error: Expected " << PATTERN_SIZE.area() << " corners but found " << numCornersFound << std::endl;
                //reinit corners
                corners.clear();
                worldPoints.clear();
                numCornersFound = 0;                
                continue;
            }

            worldPoints = calculateWorldPoints(PATTERN_SIZE);


            drawChessboardCorners(frame,  PATTERN_SIZE, cv::Mat(corners), patternFound);

            annotateWorldPoints(frame, worldPoints, corners);


        }else{
            numCornersFound = 0;
        }   

        // display the number of corners found on top left of frame
        cv::putText(frame, "Corners found: " + std::to_string(numCornersFound), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);


        cv::imshow("Frame", frame);
        char key = cv::waitKey(1);
        if (key == 'q')
        {
            std::cout << "Quitting program" << std::endl;
            break;
        }else if (key == 's' && corners.size() > 0){
            std::cout << "Saving calibration image" << std::endl;
            allCorners.push_back(corners);
            allWorldPoints.push_back(worldPoints);

            // save annotated frame
            cv::imwrite("calibration_images/" + std::to_string(allCorners.size()) + ".jpg", frame);


        }else if (key == 'c' && allCorners.size() >= 5){
            // calibrate the camera if we have more than 5 calibration images

            std::cout << "Calibrating camera" << std::endl;
            std::vector<cv::Mat> rotations, translations;

            std::cout << "Pre Calibration:" << std::endl;
            std::cout << "Camera matrix: " << cameraMatrix << std::endl;
            std::cout << "Distortion coefficients: " << std::endl;
            printVector(distCoeffs);


            double reprojectionError = calibrateCamera(allWorldPoints, allCorners, PATTERN_SIZE, cameraMatrix, distCoeffs, rotations, translations, cv::CALIB_FIX_ASPECT_RATIO);

            std::cout << "Post Calibration:" << std::endl;
            std::cout << "Camera matrix: " << cameraMatrix << std::endl;
            std::cout << "Distortion coefficients: " << std::endl;
            printVector(distCoeffs);
            // save camera matrix and distortion coefficients
            cv::FileStorage fs("calibration_data.yml", cv::FileStorage::WRITE);
            fs << "cameraMatrix" << cameraMatrix;
            fs << "distCoeffs" << distCoeffs;
            fs << "reprojectionError" << reprojectionError;
            fs.release();

            break;
        }


    }

    return 0;
}