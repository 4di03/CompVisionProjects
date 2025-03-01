/** 
* Adithya Palle
* Feb 7 2025
* CS 5330 - Project 3 : Real-time 2D Object Recognition
* 
* This file is the entrypoint for the program that runs object recognition on a given image.
*/
#include <opencv2/opencv.hpp>
#include <iostream>
#include "objectRecognition.h"

/**
 * Runs object recognition on the given image.
 * @param argc The number of command line arguments.
 * @param argv The command line arguments. The first argument is the path to the image.
 */
int main(int argc, char** argv){
    
    if (argc < 2){
        std::cout << "Usage: ./real_time_or <path_to_image>" << std::endl;
        return -1;
    }

    std::string imagePath = argv[1];

    runObjectRecognition(imagePath);

    std::cout << "Press any key to exit" << std::endl;
    cv::waitKey(0);
    return 0;

}