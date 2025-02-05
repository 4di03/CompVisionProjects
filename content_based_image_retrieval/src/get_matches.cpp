/**
 * Adithya Palle
 * February 4, 2025
 * 
 * This contains the entry point for the program that finds the top N images that are most similar to a target image.
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
#include "find_similar_images.h"

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

    return getMatchingImages(targetImagePath, imageDBPath, featureMethod, distanceMetric, numOutputImages, "tmp_output");

}