/**
 * Adithya Palle
 * February 4, 2025
 * 
 * This is a script that compares the 4 primary different methods of finding similar images for a given image and dataset.
 */
#include "find_similar_images.h"
#include <iostream>


int main(int argc, char** argv){

    std::string targetImagePath = argv[1];
    std::string imageDBPath = argv[2];
    int numOutputImages = std::stoi(argv[3]);


    // try all possible methods

    if (getMatchingImages(targetImagePath, imageDBPath, "EdgeUniformity", "SSD_float", numOutputImages, "sobel_fft_ssd") != 0){
        std::cerr << "Error running EdgeUniformity and SSD" << std::endl;
    };

    if (getMatchingImages(targetImagePath, imageDBPath, "Histogram3D", "HistogramIntersection", numOutputImages, "rgb_hist") != 0){
        std::cerr << "Error running Histogram3D and HistogramIntersection" << std::endl;
    };
    if (getMatchingImages(targetImagePath, imageDBPath, "MultiHistogram", "MultiHistogramIntersection", numOutputImages, "rgb_chroma_hist") != 0){
        std::cerr << "Error running MultiHistogram and MultiHistogramIntersection" << std::endl;
    };

    if (getMatchingImages(targetImagePath, imageDBPath, "TextureAndColor", "MultiHistogramIntersection", numOutputImages, "texture_color")!= 0){
        std::cerr << "Error running TextureAndColor and MultiHistogramIntersection" << std::endl;
    };
    if (getMatchingImages(targetImagePath, imageDBPath, "Resnet", "CosineDistance", numOutputImages, "resnet") != 0){
        std::cerr << "Error running Resnet and CosineDistance" << std::endl;
    };





    return 0;





}