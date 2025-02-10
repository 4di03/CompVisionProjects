/** 
* Adithya Palle
* Feb 7 2025
* CS 5330 - Project 3 : Real-time 2D Object Recognition
* 
* This file is the entrypoint for the program that runs object recognition on all images in a directory and saves the results in the output directory.
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
#include "thresholding.h"

/**
 * Runs object recognition on objects ina directory and saves to output
 * @param argc The number of command line arguments.
 * @param argv The command line arguments. The first argument is the path to the directory.
 */
int main(int argc, char** argv){
    
    if (argc < 2){
        std::cout << "Usage: ./real_time_or <path_to_image>" << std::endl;
        return -1;
    }

    std::string imageDBPath = argv[1];


    // compare with images in the database
    DIR* dirp = opendir(imageDBPath.c_str());
    if (dirp == NULL) {
        printf("Cannot open directory %s\n", imageDBPath.c_str());
        return -1;
    }
    char buffer[256]; // buffer to store the full path name of the image
    // loop over all the files in the image file listing
    struct dirent *dp;

    mkdir("output", 0777);


    while( (dp = readdir(dirp)) != NULL ) {


        // check if the file is an image
        if( strstr(dp->d_name, ".jpg") ||
        strstr(dp->d_name, ".png") ||
        strstr(dp->d_name, ".ppm") ||
        strstr(dp->d_name, ".tif") ||
        strstr(dp->d_name, "jpeg") ) {



        // Implicit conversion (creates a copy)
        std::string imgName = imageDBPath + "/" + dp->d_name;

        runObjectRecognition(imgName);

        }
    }

    return 0;

}