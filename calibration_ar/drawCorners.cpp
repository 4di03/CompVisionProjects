/**
 * Adithya Palle
 * February 28, 2025
 * 
 * Main file for program that draws corners on video feed if it finds the chessboard pattern in the video.
 */
#include "calibration.h"



/**
 * draws corners on video feed if it finds the chessboard pattern in the video.
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
    while (true){
        cap >> frame;

        if (frame.empty())
        {
            std::cerr << "Error reading frame" << std::endl;
            return 1;
        }

        std::vector<cv::Point2f> corners = std::vector<cv::Point2f>();
        bool patternFound = extractCorners(frame, corners, PATTERN_SIZE);
        if (patternFound)
        {
            numCornersFound = corners.size();

            drawChessboardCorners(frame,  PATTERN_SIZE, cv::Mat(corners), patternFound);
        }else{
            numCornersFound = 0;
        }   

        // display the number of corners found on top left of frame
        cv::putText(frame, "Corners found: " + std::to_string(numCornersFound), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);


        cv::imshow("Frame", frame);
        cv::waitKey(1);


    }

    return 0;
}