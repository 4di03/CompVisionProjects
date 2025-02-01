
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
 * Given a target image, a database of images, a method of computing features for an image, 
 * a distance metric for comparing the image features from two images, and the desired number of output images,
 * this program finds the N most similar images in the database to the target image.
 * It will print the file path of the N most similar images and the distance between the target image and the N most similar images.
 * It will also create a tmp_output directory in the current directory and save the N most similar images(by index) 
 * as well as the target image in the tmp directory.
 * 
 * 
 * @param argc the number of arguments
 * @param argv the arguments. The first is the target image path
 *                            the second is the directory of images path
 *                            the third is the method of computing features as a string
 *                            the fourth is the distance metric as a string
 *                            and the fifth is the number of output images(N).
 */
int main(int argc, char *argv[]) {
    if (argc != 6){
        printf("usage: %s <target image path> <directory of images path> <method of computing features> <distance metric> <N>\n", argv[0]);
        exit(-1);
    }

    std::string targetImagePath = argv[1];
    std::string imageDBPath = argv[2];
    std::string featureMethod = argv[3];
    std::string distanceMetric = argv[4];
    int numOutputImages = std::stoi(argv[5]);


    FeatureExtractor* featureExtractorMethod = featureExtractorMap[featureMethod];
    DistanceMetric* distanceMetricMethod = distanceMetricMap[distanceMetric];

    // Extract features from the target image
    std::vector<cv::Mat> targetFeatures = featureExtractorMethod->extractFeaturesFromFile(targetImagePath);
    std::cout << "Features extracted" << std::endl;
    std::vector <std::pair<std::string, double>> imageDistances;

    // compare with images in the database
    DIR* dirp = opendir(imageDBPath.c_str());
    if (dirp == NULL) {
        printf("Cannot open directory %s\n", imageDBPath.c_str());
        exit(-1);
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
    for (int i = 0; i < numOutputImages; i++){
        printf("Image: %s, Distance: %f\n", imageDistances[i].first.c_str(), imageDistances[i].second);
    }


    // create a tmp_output directory and save the target image and the top N images

    std::string outputDir = "tmp_output";

    // delete the directory if it already exists
    std::string deleteCommand = "rm -rf " + outputDir;
    std::system(deleteCommand.c_str()); 

    mkdir(outputDir.c_str(), 0777);

    // save the target image
    cv::imwrite(outputDir + "/target.jpg", cv::imread(targetImagePath));

    // save the top N images
    for (int i = 0; i < numOutputImages; i++){
        cv::imwrite(outputDir + "/output_" + std::to_string(i) + ".jpg", cv::imread(imageDistances[i].first));
    }

    


    return 0;

}