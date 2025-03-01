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
#include "classify.h"
#include "objectRecognition.h"
#define FEATURE_DATA_LOCATION "image_features"


/**
 * Runs object recognition on a webcam feed, producing a labeled image of the objects in the feed, a oriented bounding box image, and a segmented image.
 * @param argc The number of command line arguments.
 * @param argv The command line arguments. There are none
 */
int main(int argc, char** argv) {
    mkdir(FEATURE_DATA_LOCATION, 0777);

    cv::VideoCapture *capdev;
    capdev = new cv::VideoCapture(0);
    if (!capdev->isOpened()) {
        printf("Unable to open video device\n");
        return -1;
    }

    std::cout << "Ctrl+C to exit" << std::endl;

    cv::Mat rawFrame;

    ObjectFeatures db = loadKnownFeatures(FEATURE_DATA_LOCATION);

    ScaledEuclideanDistance d;
    DistanceMetric& metric = d;
    while (true) {
        *capdev >> rawFrame; // Get new frame



        if (rawFrame.empty()) {
            printf("Frame is empty\n");
            break;
        }
        cv::resize(rawFrame, rawFrame, cv::Size(320, 240));

        RegionData data = getRegionMap(rawFrame);
        cv::Mat regionMap = data.regionMap;
        std::unordered_map<int, int> regionSizes = data.regionSizes;


        int largestRegionId = 0;
        int largestRegionSize = 0;
        for (auto& [id, size] : regionSizes) {
            if (size > largestRegionSize) {
                largestRegionSize = size;
                largestRegionId = id;
            }
        }   

        cv::imshow("Raw Frame", rawFrame);

        cv::Mat thresholdedFrame = segmentObjects(rawFrame, regionMap);
        cv::imshow("Segmented Frame", thresholdedFrame);

        cv::Mat featuresImage = drawFeatures(rawFrame, regionMap, largestRegionId);
        cv::imshow("Features Image", featuresImage);

        std::string predictedLabel = findBestMatch(rawFrame, db,metric);
        cv::Mat labeledImage = labelImage(rawFrame, predictedLabel);
        cv::imshow("Labeled Image", labeledImage);

        // Check for key press
        char key = cv::waitKey(1); 
        if (key == 'n') {
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