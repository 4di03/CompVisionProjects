/**
 * Adithya Palle
 * February 4, 2025
 * 
 * This file is an implementation of the function that finds the top N images that are most similar to a target image.
 */

#include <iostream>
#include <stdio.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#include "distanceMetric.h"
#include "featureExtractor.h"



/**
 * Gets the top {numOutputImages} images that are most similar to the target image.
 * @param targetImagePath the path to the target image
 * @param imageDBPath the path to the directory of images
 * @param featureMethod the method of computing features for an image
 * @param distanceMetric the distance metric for comparing the image features from two images
 * @param numOutputImages the number of output images
 * @param outputDir the directory to save the output images
 * @return 0 if the function runs successfully, -1 otherwise
 */
int getMatchingImages(const std::string& targetImagePath, const std::string& imageDBPath, const std::string& featureMethod, const std::string& distanceMetric, int numOutputImages, std::string outputDir) {

    // check for valid feature method
    if (featureExtractorMap.find(featureMethod) == featureExtractorMap.end()){
        std::cerr << "Invalid feature method: " << featureMethod << std::endl;
        return -1;
    }

    // check for valid distance metric
    if (distanceMetricMap.find(distanceMetric) == distanceMetricMap.end()){
        std::cerr << "Invalid distance metric: " << distanceMetric << std::endl;
        return -1;
    }

    std::cout << "Finding matches for target image: " << targetImagePath << std::endl;
    std::cout << "Using feature method: " << featureMethod << std::endl;
    std::cout << "Using distance metric: " << distanceMetric << std::endl;
    std::cout << "Will output the top " << numOutputImages << " images to: " << outputDir << std::endl;

    FeatureExtractor* featureExtractorMethod = featureExtractorMap[featureMethod];
    DistanceMetric* distanceMetricMethod = distanceMetricMap[distanceMetric];

    // Extract features from the target image
    std::vector<cv::Mat> targetFeatures = featureExtractorMethod->extractFeaturesFromFile(targetImagePath);
    std::vector <std::pair<std::string, double>> imageDistances;

    // compare with images in the database
    DIR* dirp = opendir(imageDBPath.c_str());
    if (dirp == NULL) {
        printf("Cannot open directory %s\n", imageDBPath.c_str());
        return -1;
    }
    char buffer[256]; // buffer to store the full path name of the image
 // loop over all the files in the image file listing
   struct dirent *dp;

  while( (dp = readdir(dirp)) != NULL ) {


    // check if the file is an image
    if( strstr(dp->d_name, ".jpg") ||
    strstr(dp->d_name, ".png") ||
    strstr(dp->d_name, ".ppm") ||
    strstr(dp->d_name, ".tif") ) {



    // Implicit conversion (creates a copy)
    std::string imgName = imageDBPath + "/" + dp->d_name;

    double dist = distanceMetricMethod->distance(targetFeatures, featureExtractorMethod->extractFeaturesFromFile(imgName));
    
    imageDistances.push_back(std::make_pair(imgName, dist));



    }
  }


    // sort the images by distance in ascending order
    std::sort(imageDistances.begin(), imageDistances.end(), [](const std::pair<std::string, double>& a, const std::pair<std::string, double>& b){
        return a.second < b.second;
    });

    // print the top N images
    printf("Top %d images (ascending):\n", numOutputImages);
    for (int i = 0; i < numOutputImages; i++){
        printf("Image: %s, Distance: %.10f\n", imageDistances[i].first.c_str(), imageDistances[i].second);
    }

    printf("Bottom %d images (ascending):\n", numOutputImages);
    for (int i = imageDistances.size() - numOutputImages; i < imageDistances.size(); i++){
        printf("Image: %s, Distance: %.10f\n", imageDistances[i].first.c_str(), imageDistances[i].second);
    }


    // create a tmp_output directory and save the target image and the top N images

    // delete the directory if it already exists
    std::string deleteCommand = "rm -rf " + outputDir;
    std::system(deleteCommand.c_str()); 

    mkdir(outputDir.c_str(), 0777);

    mkdir((outputDir + "/top").c_str(), 0777);
    mkdir((outputDir + "/bottom").c_str(), 0777);

    
    // save the target image
    cv::imwrite(outputDir + "/target.jpg", cv::imread(targetImagePath));

    // save the top N images
    for (int i = 0; i < numOutputImages; i++){
        std::string saveDir = outputDir + "/top/output_" + std::to_string(i) + ".jpg";
        cv::imwrite(saveDir, cv::imread(imageDistances[i].first));
    }

    // save the bottom N images

    for (int i = imageDistances.size() - numOutputImages; i < imageDistances.size(); i++){

        std::string saveDir = outputDir + "/bottom/output_" + std::to_string(i) + ".jpg";

        cv::imwrite(saveDir, cv::imread(imageDistances[i].first));
    }

    


    return 0;
}


