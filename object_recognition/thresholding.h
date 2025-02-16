/**
 * Adithya Palle
 * Feb 7 2025
 * CS 5330 - Project 3 : Real-time 2D Object Recognition
 * 
 * Header file for the thresholding function used to segment the object.
 */
#include <opencv2/opencv.hpp>
#define NUM_EROSION_ITERATIONS 1
#define NUM_DILATION_ITERATIONS 5
#pragma once

// dataclass representing translation , scale , and rotationally invaraint feature vector for regions in image
class RegionFeatureVector{
    public:
        float bboxPctFilled; // percentage of the bounding box filled by the region
        float bboxAspectRatio; // longest side of the bounding box divided by the shortest side
        float circularity; // how circular the object is
        cv::Vec3f meanColor; // mean color of the region, using a float vector cause that is how its saved to file


        // default constructor
        RegionFeatureVector(float bboxPctFilled, float bboxAspectRatio, float circularity, cv::Vec3b meanColor){
            this->bboxPctFilled = bboxPctFilled;
            this->bboxAspectRatio = bboxAspectRatio;
            this->circularity = circularity;
            this->meanColor = meanColor;
        }
        /**
         * Loads the feature vector from a file.
         * @param filename The name of the file to load the feature vector from.
         * @return The feature vector loaded from the file.
         */
        RegionFeatureVector(std::string filename){
            FILE* featureFile = fopen(filename.c_str(), "r");
            fscanf(featureFile, "%f\n", &bboxPctFilled);
            fscanf(featureFile, "%f\n", &bboxAspectRatio);
            fscanf(featureFile, "%f\n", &circularity);
            fscanf(featureFile, "%f\n", &meanColor[0]);
            fscanf(featureFile, "%f\n", &meanColor[1]);
            fscanf(featureFile, "%f\n", &meanColor[2]);
            fclose(featureFile);
        }

        /**
         * Converts the feature vector to a vector of floats.
         * @return A vector of floats representing the feature vector.
         */
        std::vector<float> toVector(){
                return {
                bboxPctFilled,
                bboxAspectRatio,
                circularity,
                static_cast<float>(meanColor[0]),  // Convert uchar -> float
                static_cast<float>(meanColor[1]),
                static_cast<float>(meanColor[2])
            };
        }
        
        /**
         * Saves the feature vector to a file.
         * @param filename The name of the file to save the feature vector to.
         */
        void save(std::string filename){
            FILE* featureFile = fopen(filename.c_str(), "w");
            for (float f : this->toVector()){
                fprintf(featureFile, "%f\n", f);
            }
            fclose(featureFile);
        }


};


// struct representing the region data
struct RegionData{
    cv::Mat regionMap;
    std::unordered_map<int,int> regionSizes;
};
RegionData getRegionMap(const cv::Mat& image);

// function that produces thresholded image for task 1 where the object's pixels are white and the background is black
cv::Mat segmentObjects(const cv::Mat& image, const cv::Mat& regionMap);

RegionFeatureVector getRegionFeatures(const cv::Mat& image, const cv::Mat& regionMap, int regionId);

// gets the features of the largest region in the image
RegionFeatureVector getObjectFeatures(const cv::Mat& image);
cv::Mat drawFeatures(const cv::Mat& image, const cv::Mat& regionMap, int regionId);

void runObjectRecognition(std::string imgPath, bool saveFeatures = false);

// function that cleans up the mask by removing small regions. both here for comparison
cv::Mat cleanupSimple(const cv::Mat& mask, int numErosionIterations = NUM_EROSION_ITERATIONS, int numDilationIterations = NUM_DILATION_ITERATIONS);
cv::Mat cleanupGrassfire(const cv::Mat& mask, int numErosionIterations = NUM_EROSION_ITERATIONS, int numDilationIterations = NUM_DILATION_ITERATIONS);
