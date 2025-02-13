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
#include "classify.h"
#define PREDICTIONS_FOLDER "predictions"




/**
 * Loads the known features from the given directory.
 * @param knownDBPath The path to the known image database.
 * @return A map of the known features.
 */
ObjectFeatures loadKnownFeatures(const std::string& knownDBPath){

    ObjectFeatures knownFeatures;

    std::vector<FilePath> filePaths = getFilePathsFromDir(knownDBPath, {".features"});

    for (FilePath fp: filePaths){
            RegionFeatureVector features = RegionFeatureVector(fp.getFullPath());
            knownFeatures.names.push_back(fp.getName());
            knownFeatures.features.push_back(features.toVector());
    }
    return knownFeatures;
}

/**
 * Gets the label of the best match for the unknown image in the known image database.
 * Uses the scaled eucliedean deistance metric sum(((x_1 - x_2) / stdev_x ))^2)
 * @param unknownImg The unknown image.
 * @param db The known image database.
 * @return The label of the best match.
 */
std::string findBestMatch(const cv::Mat& unknownImg, ObjectFeatures db){
    if (db.size() == 0){
        std::cout << "Error: Known image database is empty" << std::endl;
        exit(-1);
    }
    RegionFeatureVector unknownFeatures = getObjectFeatures(unknownImg);
    float minDist = INT_MAX;
    std::string bestMatch = "";

    std::vector<float> stdDevs = db.getSTDDevs();
    std::vector<float> unknownVec = unknownFeatures.toVector();


    for (int i = 0; i < db.size(); i++){

        float dist = 0;
        std::vector<float> knownVec = db.features[i];
        for (int i = 0; i < unknownVec.size(); i++){

            float norm_diff = (unknownVec[i] - knownVec[i]) / stdDevs[i];
            dist += norm_diff * norm_diff;
        }

        if (dist < minDist){
            minDist = dist;
            bestMatch = db.names[i];
        }
    }
    return bestMatch;
}


/**
 * Labels the image with the best match from the known image database.
 * Places the label in the top left corner of the image.
 * @param image The image to label.
 * @param db The known image database to use for classification.
 * @return The labeled image.
 */
cv::Mat labelImage(const cv::Mat& image, ObjectFeatures db){
    cv::Mat labeledImage = image.clone();

    std::string label = findBestMatch(image, db);

    int thickness = image.cols/100;
    float fontScale = thickness * 0.4;
    printf("Label: %s\n", label.c_str());
    cv::putText(labeledImage, label, cv::Point(10, 0.1*image.rows), cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(255,255,255), thickness);


    return labeledImage;
}

