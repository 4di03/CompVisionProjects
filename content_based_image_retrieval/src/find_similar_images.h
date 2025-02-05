/**
 * Adithya Palle
 * February 4, 2025
 * 
 * 
 * This is a simple header file that exposes the function getMatchingImages that finds the top N images that are most similar to a target image.
 * It is used so multiple executables can use this function.
 */
#include <iostream>
#pragma once

int getMatchingImages(const std::string& targetImagePath, const std::string& imageDBPath, const std::string& featureMethod, const std::string& distanceMetric, int numOutputImages, std::string outputDir);
