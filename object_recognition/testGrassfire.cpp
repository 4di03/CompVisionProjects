/**
 * Adithya Palle
 * Feb 15 2025
 * CS 5330 - Project 3 : Real-time 2D Object Recognition
 * 
 * Entrypoint for code that runs the slow version of cleanup (naive dilation and erosion) on the image, 
 * and then runs the optimized version of cleanup (grassfire algorithm) on the image.
 * Compares the two results based on both speed and accuracy.
 */


#include <iostream>
#include <opencv2/opencv.hpp>
#include "thresholding.h"
#define MASK_SIZE 1000
#define PROP_BG 0.7 // proportion of background pixels in the mask
#define NUM_TESTS 5
int main(){


    // generate an arbitrary mask (255 is background, 0 is foreground)

    cv::Mat mask = cv::Mat::zeros(MASK_SIZE,MASK_SIZE, CV_8UC1);


    // Random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(0, 1); // generates float between 0 and 1

    // Fill the mask randomly
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            // we want to fill the mask with PROP_BG% background and (1-PROP_BG)% foreground so that the foreground is sparse, and we can see the effect of the dilation and erosion
            mask.at<uchar>(i, j) = dist(gen < PROP_BG) ? 255 : 0;
        }
    }

    // run the slow version of cleanup 5 times (time it)
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_TESTS; i++){
        cv::Mat cleaned = cleanupNaive(mask);
    }
    auto end = std::chrono::high_resolution_clock::now();

    cv::Mat naiveCleaned = cleaned.clone();
    std::chrono::duration<double> elapsed_seconds = end-start;
    std::cout << "Time taken for naive cleanup: " << elapsed_seconds.count() << "s" << std::endl;

    // run the optimized version of cleanup 5 times (time it)
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_TESTS; i++){
        cv::Mat cleaned = cleanupFast(mask);
    }
    end = std::chrono::high_resolution_clock::now();

    elapsed_seconds = end-start;
    std::cout << "Time taken for optimized cleanup: " << elapsed_seconds.count() << "s" << std::endl;

    cv::Mat optimizedCleaned = cleaned.clone();


    // compare the two results
    int numDifferentPixels = 0;
    for (int i = 0; i < MASK_SIZE; i++){
        unsigned char* naiveRow = naiveCleaned.ptr<unsigned char>(i);
        unsigned char* optimizedRow = optimizedCleaned.ptr<unsigned char>(i);
        for (int j = 0; j < MASK_SIZE; j++){
            if (naiveRow[j] != optimizedRow[j]){
                numDifferentPixels++;
            }
        }
    }

    if (numDifferentPixels == 0){
        std::cout << "The two results are the same" << std::endl;
    }else{
        std::cout << "The two results are different. Number of different pixels: " << numDifferentPixels << std::endl;
    }

    cv::imshow("Original Mask", mask);
    cv::imshow("Slow Cleanup", naiveCleaned);
    cv::imshow("Fast Cleanup", optimizedCleaned);


    return 0;


}