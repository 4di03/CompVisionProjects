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
        cv::Mat thresholdedFrame = segmentObjects(rawFrame);
        cv::imshow("Thresholded Frame", thresholdedFrame);

        RegionData data = getRegionMap(image);
        cv::Mat regionMap = data.regionMap;
        std::unordered_map<int,int> regionSizes = data.regionSizes;
        // get id of max size region
        int largestRegionId = 0;
        int largestRegionSize = 0;
        for (auto it = regionSizes.begin(); it != regionSizes.end(); it++){
            if (it->second > largestRegionSize){
                largestRegionSize = it->second;
                largestRegionId = it->first;
            }
        }
        // Get features for the largest region
        cv::Mat featuresImage = drawFeatures(image, regionMap, largestRegionId);
        
        cv::imshow("Features Image", featuresImage);

        cv::Mat segmented = segmentObjects(image, regionMap);
        cv::imshow("Segmented Image", segmented);

        // see if there is a waiting keystroke
        char key = cv::waitKey(1);

        if (key == 'q'){
            break;
        }


    }

    return 0;


 }