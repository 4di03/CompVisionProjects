/**
 * Adithya Palle
 * Jan 24 2025
 * CS 5330 - Project 1 : Filter implementation
 */

#include "filter.h"
#include <opencv2/opencv.hpp>
#include <iostream>
/**
 * Applies a custom grayscale filter to the image.
 * Produces the gray value by summing the RGB values and then modding by 256.
 * 
 * @param src The source image.
 * @param dst The destination image.
 * @returns 0 if the operation was successful, -1 otherwise.
 */
int alternativeGrayscale(const cv::Mat& src, cv::Mat& dst){
    if(src.empty() || src.size() != dst.size()){
        return -1;
    }
    // copy src so that we dont have to read and write to the same image
    cv::Mat _src = src.clone();
    // reset the dst image (for case where src and dst are the same)
    dst.create(src.size(), CV_8UC1);
    

    for(int i = 0; i < _src.rows; i++){
        for(int j = 0; j < _src.cols; j++){
            cv::Vec3b pixel = _src.at<cv::Vec3b>(i, j);
            dst.at<uchar>(i, j) = (pixel[0] + pixel[1] + pixel[2]) % 256;
        }
    }
    return 0;
}