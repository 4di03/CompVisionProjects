/**
 * Adithya Palle
 * Feb 7 2025
 * CS 5330 - Project 3 : Real-time 2D Object Recognition
 * 
 * Header file for the thresholding function used to segment the object.
 */
#include <opencv2/opencv.hpp>

// struct representing translation , scale , and rotationally invaraint feature vector for regions in image
struct RegionFeatureVector{
    float bboxPctFilled; // percentage of the bounding box filled by the region
    float bboxAspectRatio; // longest side of the bounding box divided by the shortest side
    float circularity; // how circular the object is
    cv::Vec3b meanColor; // mean color of the region



    /**
     * Converts the feature vector to a vector of floats.
     * @return A vector of floats representing the feature vector.
     */
    std::vector<float> toVector(){
        std::vector<float> vec;
        vec.push_back(bboxPctFilled);
        vec.push_back(bboxAspectRatio);
        vec.push_back(circularity);
        vec.push_back(meanColor[0]);
        vec.push_back(meanColor[1]);
        vec.push_back(meanColor[2]);
        return vec;
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

RegionFeatureVector getRegionFeatures(const cv::Mat& image, const cv::Mat& mask);

cv::Mat drawFeatures(const cv::Mat& image, const cv::Mat& regionMap, int regionId);

void runObjectRecognition(std::string imgPath);
