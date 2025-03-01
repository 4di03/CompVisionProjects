/** 
* Adithya Palle
* Feb 7 2025
* CS 5330 - Project 3 : Real-time 2D Object Recognition
* 
* This file is the entrypoint for the program that runs classification with a known image feature database and a unknown image database.
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
#include <cmath>
#include <fstream>
#include "utils.h"
#include "objectRecognition.h"
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
 * Computes the scaled euclidean distance between two vectors.
 * @param a The first vector.
 * @param b The second vector.
 * @param std The standard deviations of the features.
 */
float ScaledEuclideanDistance::distance(const std::vector<float>& a, const std::vector<float>& b, const std::vector<float>& std){
    if (std.size() != a.size()){
        std::cout << "Error: Standard deviations not provided" << std::endl;
        exit(-1);
    }
    float dist = 0;
    for (int i = 0; i < a.size(); i++){
        float norm_diff = (a[i] - b[i]) / std[i];
        dist += norm_diff * norm_diff;
    }
    return dist;
}


/**
 * Computes the cosine distance between two vectors.
 * @param a The first vector.
 * @param b The second vector.
 * @param std (NOT NEEDED) The standard deviations of the features.
 */
float CosineDistance::distance(const std::vector<float>& a, const std::vector<float>& b, const std::vector<float>& std){
    float dotProduct = 0;
    float aMag = 0;
    float bMag = 0;
    for (int i = 0; i < a.size(); i++){
        dotProduct += a[i] * b[i];
        aMag += a[i] * a[i];
        bMag += b[i] * b[i];
    }
    aMag = sqrt(aMag);
    bMag = sqrt(bMag);
    return 1 - (dotProduct / (aMag * bMag));
}


/**
 * Computes the scaledd chebyshev distance (L-inf norm) between two vectors.
 * @param a The first vector.
 * @param b The second vector.
 * @param std (NOT NEEDED) The standard deviations of the features.
 */
float ChebyshevDistance::distance(const std::vector<float>& a, const std::vector<float>& b, const std::vector<float>& std){
    if (std.size() != a.size()){
        std::cout << "Error: Standard deviations not provided" << std::endl;
        exit(-1);
    }
    float maxDiff = 0;
    for (int i = 0; i < a.size(); i++){
        float diff = abs((a[i] - b[i])/ std[i]);
        if (diff > maxDiff){
            maxDiff = diff;
        }
    }
    return maxDiff;
}



/**
 * Computes the simple euclidean distance between two vectors.
 * @param a The first vector.
 * @param b The second vector.
 * @param std (NOT NEEDED) The standard deviations of the features.
 */
float SimpleEuclideanDistance::distance(const std::vector<float>& a, const std::vector<float>& b, const std::vector<float>& std){
    float dist = 0;
    for (int i = 0; i < a.size(); i++){
        float diff = a[i] - b[i];
        dist += diff * diff;
    }
    return dist;
}

/**
 * Computes the scaled manhattan distance between two vectors.
 * @param a The first vector.
 * @param b The second vector.
 * @param std The standard deviations of the features.
 */
float ManhattanDistance::distance(const std::vector<float>& a, const std::vector<float>& b, const std::vector<float>& std){
    if (std.size() != a.size()){
        std::cout << "Error: Standard deviations not provided" << std::endl;
        exit(-1);
    }
    float dist = 0;
    for (int i = 0; i < a.size(); i++){
        float diff = (a[i] - b[i])/std[i];
        dist += abs(diff);
    }
    return dist;
}


/**
 * Gets the label of the best match for the unknown image in the known image database.
 * Uses the scaled eucliedean deistance metric sum(((x_1 - x_2) / stdev_x ))^2)
 * @param unknownImg The unknown image.
 * @param db The known image database.
 * @param metric The distance metric to use.
 * @return The label of the best match.
 */
std::string findBestMatch(const cv::Mat& unknownImg, ObjectFeatures db, DistanceMetric& metric){
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

        std::vector<float> knownVec = db.features[i];

        float dist = metric.distance(unknownVec, knownVec, stdDevs);


        if (dist < minDist){
            minDist = dist;
            bestMatch = db.names[i];
        }
    }
    return bestMatch;
}


/**
 * Labels the image with the given string
 * Places the label in the top left corner of the image.
 * @param image The image to label.
 * @param label The label to put on the image.
 * @return The labeled image.
 */
cv::Mat labelImage(const cv::Mat& image, std::string label){
    cv::Mat labeledImage = image.clone();


    int thickness = image.cols/100;
    float fontScale = thickness * 0.4;

    cv::putText(labeledImage, label, cv::Point(10, .2*image.rows), cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(255,255,255), thickness);


    return labeledImage;
}


/**
 * Predicts the labels of the unknown images using the known image database.
 * @param unknownImgPaths The paths to the unknown images.
 * @param db The known image database.
 * @param metric The distance metric to use.
 */
void predict(const std::vector<FilePath>& unknownImgPaths, ObjectFeatures db, DistanceMetric& metric, std::string outputDir){

    mkdir(outputDir.c_str(), 0777);
    std::ofstream csvFile;
    std::string csvFileName = outputDir + "/predictions.csv";

    // delete the file if it exists
    remove(csvFileName.c_str());

    csvFile.open(csvFileName, std::ios_base::app);
    


    csvFile << "Image Name,Label\n";

    for(FilePath fp : unknownImgPaths){
        printf("Processing %s\n", fp.getFullPath().c_str());
        cv::Mat unknownImg = cv::imread(fp.getFullPath());


        if (unknownImg.empty()){
            std::cout << "Error: Image not found" << std::endl;
            exit(-1);
        }

        std::string label = findBestMatch(unknownImg, db, metric);
        cv::Mat labeledImage = labelImage(unknownImg, label);
        std::string outputFileName = outputDir + "/" + fp.getName() + "_prediction.jpg";

        std::string csvLine = fp.getName() + "," + label + "\n";

        csvFile << csvLine;


        cv::imwrite(outputFileName, labeledImage);
    }

    csvFile.close();

}