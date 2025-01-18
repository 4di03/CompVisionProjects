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
#include "faceDetect/faceDetect.h"

/**
 * Processes the last keypress and applies an effect to the frame.
 * If the keypress is unknown, the frame is left unmodified.
 * 
 * The result is placed in dst, a 3 channel uchar image.
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

        case 'x':{
            cv::Mat sobelOut;

            if (sobelX3x3(frame, sobelOut) != 0) {
                std::cout << "Error applying sobelX3x3" << std::endl;
                exit(-1);
            }
            prepareFrameForDisplay(sobelOut, dst);

            break;
        }

        case 'y':{
            cv::Mat sobelOut;
            
            if (sobelY3x3(frame, sobelOut) != 0) {
                std::cout << "Error applying sobelY3x3" << std::endl;
                exit(-1);
            }
            prepareFrameForDisplay(sobelOut,  dst);

            break;
        }

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
        case 'f':{
            // detects and draws faces on the frame
            std::vector<cv::Rect> faces;
            cv::Mat grey;
            cv::cvtColor(frame, grey, cv::COLOR_BGR2GRAY);

            if (detectFaces(grey, faces) != 0){
                std::cout << "Error detecting faces" << std::endl;
                exit(-1);
            }
            // copy the frame to dst so that we can draw the boxes on dst
            frame.copyTo(dst);
            if (drawBoxes(dst, faces) != 0){
                std::cout << "Error drawing boxes" << std::endl;
                exit(-1);
            }
            break;


        }
        case 'd':{
            if (depth(frame, dst) != 0){
                std::cout << "Error applying depth" << std::endl;
                exit(-1);
            }

            break;
        }
        case 'z':{
            // applies sepia to only the foreground of the image
            if (applyToForeground(frame, dst, 128, sepia) != 0){
                std::cout << "Error applying blurBackground" << std::endl;
                exit(-1);
            }
            break;
        }
        case 'r': {
            if(medianFilter(frame,dst)!= 0){
                std::cout << "Error applying median filter" << std::endl;
                exit(-1);
            }
            break;
        }
        case 'v':{
            if (depthFog(frame, dst) != 0){
                std::cout << "Error applying depth fog" << std::endl;
                exit(-1);
            }
            break;
        }
        case 'n':{
            if (faceSwirl(frame, dst) != 0){
                std::cout << "Error applying face swirl" << std::endl;
                exit(-1);
            }
            break;
        }
        default:{
            frame.copyTo(dst);
            break;
        }
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
    int deltaBrightness = 0;
    cv::Mat resultantFrame;
    while(true){
            *capdev >> rawFrame; // get a new frame from the camera, treat as a stream

            
            if( rawFrame.empty() ) {
                printf("frame is empty\n");
                break;
            }                

            //resize 
            cv::resize(rawFrame, rawFrame, cv::Size(640, 480));

            cv::Mat displayFrame;

            processLastKeypress(rawFrame,displayFrame, lastKeypress);

            // adjust the brightness of the frame
            if (deltaBrightness != 0){
                adjustBrightness(displayFrame, displayFrame, deltaBrightness);
            }

            cv::imshow("Video", displayFrame);

            // see if there is a waiting keystroke
            char key = cv::waitKey(1);
            if( key == 'q') {
                break;
            }else if(key == 's'){
                // saves the frame as a screenshot
                cv::imwrite(SCREENSHOT_SAVE_LOC, displayFrame);
            } else if ( key == '+' ||  key == '='){
                deltaBrightness += 10;
            } else if ( key == '-' || key == '_'){
                deltaBrightness -= 10;
            }
            else if (key != -1){ // can compare due to implicit conversion to int
                lastKeypress = key;
            } 
    
    }

    delete capdev;
    return 0;


}