/**
 * Adithya Palle
 * Jan 24 2025
 * CS 5330 - Project 1, Tasks 2 - 4 : Video Display
 * 
 * This file contains the implementation of a program that reads a video stream from a camera and displays it on the screen, allowing the user to apply filters to the video stream.
 */
#define SCREENSHOT_SAVE_LOC "screenshot.jpg" // location to save the screenshot
#include <iostream>
#include <opencv2/opencv.hpp>
#include "filter.h"

/**
 * Processes the last keypress and applies an effect to the frame.
 * If the keypress is unknown, the frame is left unmodified.
 * 
 * The result is placed in dst
 */
void processLastKeypress(cv::Mat& frame, cv::Mat& dst, char lastKeypress){
switch (lastKeypress) {
    case 'g':
        cv::cvtColor(frame, dst, cv::COLOR_BGR2GRAY);
        break;

    case 'h':
        if (alternativeGrayscale(frame, dst) != 0) {
            std::cout << "Error applying alternative grayscale" << std::endl;
            exit(-1);
        }
        break;

    case 'p':
        if (sepia(frame, dst) != 0) {
            std::cout << "Error applying sepia filter" << std::endl;
            exit(-1);
        }
        break;

    case 'b':
        if (blur5x5_2(frame, dst) != 0) {
            std::cout << "Error applying blur5x5_1" << std::endl;
            exit(-1);
        }
        break;

    case 'x':
        if (sobelX3x3(frame, dst) != 0) {
            std::cout << "Error applying sobelX3x3" << std::endl;
            exit(-1);
        }
        break;

    case 'y':
        if (sobelY3x3(frame, dst) != 0) {
            std::cout << "Error applying sobelY3x3" << std::endl;
            exit(-1);
        }
        break;

    case 'm': {
        cv::Mat sx, sy;
        if (sobelX3x3(frame, sx) != 0) {
            std::cout << "Error getting sobelX3x3" << std::endl;
            exit(-1);
        }
        if (sobelY3x3(frame, sy) != 0) {
            std::cout << "Error getting sobelY3x3" << std::endl;
            exit(-1);
        }
        if (magnitude(sx, sy, dst) != 0) {
            std::cout << "Error getting gradient magnitude" << std::endl;
            exit(-1);
        }
        break;
    }

    case 'l':{
        if (blurQuantize(frame, dst, 10) != 0) {
            std::cout << "Error applying blurQuantize" << std::endl;
            exit(-1);
        }
        break;
    }

    default:
        frame.copyTo(dst);
        break;
}
}

/**
 * Converts images that are not in [0,255] range to [0,255] range.
 */
void prepareFrameForDisplay(cv::Mat& src, cv::Mat& dst){
    // printf("Source type: %d\n", src.type());
    // printf("Destination type: %d\n", dst.type());
    // printf("CV_16SC3: %d\n", CV_16SC3);
    // printf("CV_8UC3: %d\n", CV_8UC3);

    double minVal;
    minMaxLoc(src, &minVal, nullptr);

    if (src.type() == CV_16SC3){
        if (minVal < 0){
            // if the image has negative values then we appliy pixel * 0.5 + 127.5 to convert the minimum value to 0, else if only the max is greater, we simply scale it down.

            cv::convertScaleAbs(src, dst, 0.5,  127.5); 
        } else {
            // since all values are positive, we can simply convert to 8 bit
            src.convertTo(dst, CV_8UC3);
        }
    } else {
       src.copyTo(dst);
    }
}

/**
 * Displays a video stream.
 * Press 'q' to exit. Press 's' to save a screenshot. 
 * Press 'g' to apply a grayscale effect, press any other key to reset the effect.
 */
int main(){

    cv::VideoCapture *capdev;
    // open the video device
    capdev = new cv::VideoCapture(0);
    if( !capdev->isOpened() ) {
            printf("Unable to open video device\n");
            return(-1);
    }

    cv::namedWindow("Video", 1); // identifies a window
    cv::Mat rawFrame;
    char lastKeypress;
    cv::Mat resultantFrame;
    while(true){
            *capdev >> rawFrame; // get a new frame from the camera, treat as a stream
            if( rawFrame.empty() ) {
                printf("frame is empty\n");
                break;
            }                

            processLastKeypress(rawFrame,resultantFrame, lastKeypress);


            cv::Mat displayFrame;
            // Convert images potentially in range [-255, 255] to [0, 255]
            prepareFrameForDisplay(resultantFrame, displayFrame);
            cv::imshow("Video", displayFrame);

            // see if there is a waiting keystroke
            char key = cv::waitKey(1);
            if( key == 'q') {
                break;
            }else if(key == 's'){
                // saves the frame as a screenshot
                cv::imwrite(SCREENSHOT_SAVE_LOC, displayFrame);
            } else if (key != -1){ // can compare due to implicit conversion to int
                lastKeypress = key;
            }
    
    }

    delete capdev;
    return 0;


}