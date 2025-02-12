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
ObjectFeatures loadKnownFeatures(const std::string& knownDBPath){

    ObjectFeatures knownFeatures;

    std::vector<FilePath> filePaths = getFilePathsFromDir(knownDBPath);

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
    RegionFeatureVector unknownFeatures = getObjectFeatures(unknownImg);
    float minDist = INT_MAX;
    std::string bestMatch = "";

    std::vector<float> stdDevs = db.getSTDDevs();
    for (int i = 0; i < db.size(); i++){
        float dist = 0;
        std::vector<float> unknownVec = unknownFeatures.toVector();
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

    cv::putText(labeledImage, label, cv::Point(10,10), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255,255,255), 1);
    return labeledImage;
}

/**
 * Runs classification on objects in a directory , given a known image database, and puts the results in the output directory.
 * @param argc The number of command line arguments.
 * @param argv The command line arguments. The first argument is the path to the directory.
 */
int main(int argc, char** argv){
    
    if (argc < 3){
        std::cout << "Usage: ./classify <path_to_known_db> <path_to_unknown_db>" << std::endl;
        return -1;
    }

    std::string imageDBPath = argv[1];
    std::string unknownDBPath = argv[2];


    ObjectFeatures knownFeatures = loadKnownFeatures(imageDBPath);


    std::vector<FilePath> unknownImgPaths = getFilePathsFromDir(unknownDBPath);

    mkdir(PREDICTIONS_FOLDER, 0777);

    for(FilePath fp : unknownImgPaths){
        cv::Mat unknownImg = cv::imread(fp.getFullPath());
        if (unknownImg.empty()){
            std::cout << "Error: Image not found" << std::endl;
            exit(-1);
        }
        cv::Mat labeledImage = labelImage(unknownImg, knownFeatures);
        std::string outputFileName = std::string(PREDICTIONS_FOLDER) + "/" + fp.getName() + "_prediction.jpg";
        cv::imwrite(outputFileName, labeledImage);
    }

    return 0;

}