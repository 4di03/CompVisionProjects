/**
 * Adithya Palle
 * Jan 24 2025
 * CS 5330 - Project 1, Tasks 2 - 4 : Video Display
 */
#define SCREENSHOT_SAVE_LOC "screenshot.jpg" // location to save the screenshot
#include <iostream>
#include <opencv2/opencv.hpp>
#include "filter.h"

/**
 * Processes the last keypress and applies an effect to the frame.
 * If the last keypress was 'g', the frame is converted to grayscale.
 * If the last keypress was 'h', the frame is converted to an alternative grayscale.
 * If the last keypress was 'p', a sepia filter is applied.
 * Else, the frame is left unmodified.
 */
cv::Mat processLastKeypress(cv::Mat& frame, char lastKeypress){\
    if(lastKeypress == 'g'){
        cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
    } else if (lastKeypress == 'h') {
        if (alternativeGrayscale(frame, frame) != 0){
            std::cout << "Error applying alternative grayscale" << std::endl;
            exit(-1);
        }   
    } else if (lastKeypress == 'p'){
        if(sepia(frame, frame) != 0){
            std::cout << "Error applying sepia filter" << std::endl;
            exit(-1);
        }
    } else if (lastKeypress == 'b'){
        if (blur5x5_2(frame, frame) != 0){
            std::cout << "Error applying blur5x5_1" << std::endl;
            exit(-1);
        }
    }
    return frame;
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
    cv::Mat frame;
    char lastKeypress;
    while(true){
            *capdev >> frame; // get a new frame from the camera, treat as a stream
            if( frame.empty() ) {
                printf("frame is empty\n");
                break;
            }                

            frame = processLastKeypress(frame, lastKeypress);
            cv::imshow("Video", frame);

            // see if there is a waiting keystroke
            char key = cv::waitKey(1);
            if( key == 'q') {
                break;
            }else if(key == 's'){
                // saves the frame as a screenshot
                cv::imwrite(SCREENSHOT_SAVE_LOC, frame);
            } else if (key != -1){ // can compare due to implicit conversion to int
                lastKeypress = key;
            }
    
    }

    delete capdev;
    return 0;


}