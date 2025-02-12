/**
 * Adithya Palle
 * Feb 7 2025
 * CS 5330 - Project 3 : Real-time 2D Object Recognition
 * 
 * This file is the entrypoint for the program that runs object recognition in real-time on the webcam feed.
 */
#include <opencv2/opencv.hpp>
#include <iostream>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <thread>
#include <atomic>
#include "thresholding.h"
#define FEATURE_DATA_LOCATION "image_features"


/**
 * Runs object recognition on a webcam feed.
 */
int main(int argc, char** argv) {
    mkdir(FEATURE_DATA_LOCATION, 0777);

    cv::VideoCapture *capdev;
    capdev = new cv::VideoCapture(0);
    if (!capdev->isOpened()) {
        printf("Unable to open video device\n");
        return -1;
    }

    std::cout << "Press q to exit" << std::endl;

    cv::Mat rawFrame;
    while (true) {
        *capdev >> rawFrame; // Get new frame

        if (rawFrame.empty()) {
            printf("Frame is empty\n");
            break;
        }

        cv::imshow("Video", rawFrame);

        RegionData data = getRegionMap(rawFrame);
        cv::Mat regionMap = data.regionMap;
        std::unordered_map<int, int> regionSizes = data.regionSizes;

        cv::Mat thresholdedFrame = segmentObjects(rawFrame, regionMap);
        cv::imshow("Segmented Frame", thresholdedFrame);

        int largestRegionId = 0;
        int largestRegionSize = 0;
        for (auto& [id, size] : regionSizes) {
            if (size > largestRegionSize) {
                largestRegionSize = size;
                largestRegionId = id;
            }
        }

        cv::Mat featuresImage = drawFeatures(rawFrame, regionMap, largestRegionId);
        cv::imshow("Features Image", featuresImage);

        // Check for key press
        char key = cv::waitKey(1); // Read and reset keyPressed
        if (key == 'q') {
            break;
        } else if (key == 'n') {
            std::string label;
            std::cout << "Enter label: ";
            std::cin >> label;

            RegionFeatureVector features = getRegionFeatures(rawFrame, regionMap, largestRegionId);
            std::string featuresFileName = std::string(FEATURE_DATA_LOCATION) + "/" + label + ".features";
            features.save(featuresFileName);
            printf("Saved features to %s\n", featuresFileName.c_str());
        }
    }

    return 0;
}