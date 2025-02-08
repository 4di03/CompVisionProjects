/**
 * Adithya Palle
 * Feb 7 2025
 * CS 5330 - Project 3 : Real-time 2D Object Recognition
 * 
 * This file is the entrypoint for the program that runs object recognition in real-time on the webcam feed.
 */
#include <opencv2/opencv.hpp>
#include <iostream>
#include "thresholding.h"

/**
 * Runs object recognition on a webcam feed.
 * @param argc The number of command line arguments.
 * @param argv The command line arguments. There are no arguments.
 */
 int main(int argc, char** argv){
    cv::VideoCapture *capdev;
    // open the video device
    capdev = new cv::VideoCapture(0);
    if( !capdev->isOpened() ) {
            printf("Unable to open video device\n");
            return(-1);
    }

    cv::Mat rawFrame;
    std::cout << "Press q to exit" << std::endl;
    while (true){
        *capdev >> rawFrame; // get a new frame from the camera, treat as a stream

        if( rawFrame.empty() ) {
            printf("frame is empty\n");
            break;
        }

        cv::imshow("Video", rawFrame);
        cv::Mat thresholdedFrame = getObjectMask(rawFrame);
        cv::imshow("Thresholded Frame", thresholdedFrame);

        // see if there is a waiting keystroke
        char key = cv::waitKey(1);

        if (key == 'q'){
            break;
        }


    }

    return 0;


 }