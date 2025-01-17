
/**
 * Adithya Palle
 * Jan 24 2025
 * CS 5330 - Project 1, Task 1 : Image Display
 * 
 * This file contains the implementation of a program that reads an image from a file and displays it on the screen.
 */

#include <iostream>
#include <opencv2/opencv.hpp>
/**
 * This program reads an image from a file and displays it on the screen.
 * If the user types 'q', the program will exit.
 * The 2nd argument is the name of the image file.
 */
int main(int argc, char** argv) {


    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <image file>" << std::endl;
        return -1;
    }

    std::string imageFile = argv[1];


    cv::Mat image = cv::imread(imageFile, cv::IMREAD_COLOR);

    if (image.empty()) {
        std::cout << "Could not read the image: " << imageFile << std::endl;
        return -1;
    }
    while (true){

        cv::imshow("Image", image);
        // wait for a key press
        if (cv::waitKey(0) == 'q') {
            std::cout << "Exiting..." << std::endl;
            break;
        }
    }

    return 0;
}