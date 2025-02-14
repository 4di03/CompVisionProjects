/** 
* Adithya Palle
* Feb 7 2025
* CS 5330 - Project 3 : Real-time 2D Object Recognition
* 
* This file is the entrypoint for the program that classification with a known image database and a unknown image database.
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
#include "utils.h"
#include "thresholding.h"
#define PREDICTIONS_FOLDER "predictions"
#pragma once


struct ObjectFeatures{
    std::vector<std::string> names;
    std::vector<std::vector<float>> features;


    /**
     * Gets the standard deviations of the features.
     * @return The standard deviations of the features, where stdDevs[i] is the standard deviation of the ith feature.
     */
    std::vector<float> getSTDDevs(){
        std::vector<float> stdDevs;

        int n = features.size(); // number of samples
        int m = features[0].size(); // number of features
        for (int i = 0; i < m; i++){
            float sum = 0;
            for (int j = 0; j <n; j++){
                sum += features[j][i];
            }
            float mean = sum / n;
            float stdDev = 0;
            for (int j = 0; j < n; j++){
                stdDev += (features[j][i] - mean) * (features[j][i] - mean);
            }

            
            stdDev = sqrt(stdDev / n);
            stdDevs.push_back(stdDev);
        }
        return stdDevs;
    };

    int size(){
        return features.size();
    }

};



class DistanceMetric{
    public:
        /**
         * Computes the  distance between two vectors.
         * @param a The first vector.
         * @param b The second vector.
         * @param std (optional) The standard deviations of the features.
         */
        virtual float distance(const std::vector<float>& a, const std::vector<float>& b, const std::vector<float>& std = {}) = 0;


        /**
         * Gets the name of the distance metric.
         * @return The name of the distance metric.
         */
        std::string getName(){
            
            return typeid(*this).name();
        }

        virtual ~DistanceMetric() = default;

};

class ScaledEuclideanDistance : public DistanceMetric{
    public:
        /**
         * Computes the scaled euclidean distance between two vectors.
         * @param a The first vector.
         * @param b The second vector.
         * @param std The standard deviations of the features.
         */
        float distance(const std::vector<float>& a, const std::vector<float>& b, const std::vector<float>& std);
};


class CosineDistance : public DistanceMetric{
    public:
        /**
         * Computes the cosine distance between two vectors.
         * @param a The first vector.
         * @param b The second vector.
         * @param std (NOT NEEDED) The standard deviations of the features.
         */
        float distance(const std::vector<float>& a, const std::vector<float>& b, const std::vector<float>& std);
};

class ChebyshevDistance: public DistanceMetric{
    public:
        /**
         * Computes the scaledd chebyshev distance (L-inf norm) between two vectors.
         * @param a The first vector.
         * @param b The second vector.
         * @param std (NOT NEEDED) The standard deviations of the features.
         */
        float distance(const std::vector<float>& a, const std::vector<float>& b, const std::vector<float>& std);
};


class SimpleEuclideanDistance: public DistanceMetric{
    public:
        /**
         * Computes the simple euclidean distance between two vectors.
         * @param a The first vector.
         * @param b The second vector.
         * @param std (NOT NEEDED) The standard deviations of the features.
         */
        float distance(const std::vector<float>& a, const std::vector<float>& b, const std::vector<float>& std);
};

class ManhattanDistance: public DistanceMetric{
    public:
        /**
         * Computes the scaled manhattan distance between two vectors.
         * @param a The first vector.
         * @param b The second vector.
         * @param std The standard deviations of the features.
         */
        float distance(const std::vector<float>& a, const std::vector<float>& b, const std::vector<float>& std);
};




ObjectFeatures loadKnownFeatures(const std::string& knownDBPath);


std::string findBestMatch(const cv::Mat& unknownImg, ObjectFeatures db, DistanceMetric& metric);
cv::Mat labelImage(const cv::Mat& image, std::string label);



void predict(const std::vector<FilePath>& unknownImgPaths, ObjectFeatures db, DistanceMetric& metric, std::string outputDir = PREDICTIONS_FOLDER);