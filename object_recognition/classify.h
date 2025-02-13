/** 
* Adithya Palle
* Feb 7 2025
* CS 5330 - Project 3 : Real-time 2D Object Recognition
* 
* This file is the entrypoint for the program that classification with a known image database and a unknown image database.
*/
#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#include "utils.h"
#include "thresholding.h"
#define PREDICTIONS_FOLDER "predictions"
#pragma once


struct ObjectFeatures{
    std::vector<std::string> names;
    std::vector<std::vector<float>> features;


    /**
     * Gets the standard deviations of the features.
     * @return The standard deviations of the features, where stdDevs[i] is the standard deviation of the ith feature.
     */
    std::vector<float> getSTDDevs(){
        std::vector<float> stdDevs;

        int n = features.size(); // number of samples
        int m = features[0].size(); // number of features
        for (int i = 0; i < m; i++){
            float sum = 0;
            for (int j = 0; j <n; j++){
                sum += features[j][i];
            }
            float mean = sum / n;
            float stdDev = 0;
            for (int j = 0; j < n; j++){
                stdDev += (features[j][i] - mean) * (features[j][i] - mean);
            }

            
            stdDev = sqrt(stdDev / n);
            stdDevs.push_back(stdDev);
        }
        return stdDevs;
    };

    int size(){
        return features.size();
    }

};



/**
 * Loads the known features from the given directory.
 * @param knownDBPath The path to the known image database.
 * @return A map of the known features.
 */
ObjectFeatures loadKnownFeatures(const std::string& knownDBPath);

/**
 * Gets the label of the best match for the unknown image in the known image database.
 * Uses the scaled eucliedean deistance metric sum(((x_1 - x_2) / stdev_x ))^2)
 * @param unknownImg The unknown image.
 * @param db The known image database.
 * @return The label of the best match.
 */
std::string findBestMatch(const cv::Mat& unknownImg, ObjectFeatures db);
/**
 * Labels the image with the best match from the known image database.
 * Places the label in the top left corner of the image.
 * @param image The image to label.
 * @param db The known image database to use for classification.
 * @return The labeled image.
 */
cv::Mat labelImage(const cv::Mat& image, ObjectFeatures db);