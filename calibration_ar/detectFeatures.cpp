/**
 * Adithya Palle
 * February 28, 2025
 * 
 * Main file for program that detects a custom pattern from a reference image and places a marker at its center
 * This code takes inspiration from https://docs.opencv.org/3.4/d7/d66/tutorial_feature_detection.html
 */
#include "calibration.h"
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <iostream>
#define EXPECTED_FRAME_WIDTH 1280
#define EXPECTED_FRAME_HEIGHT 720
#define SURF_MIN_HESSIAN 10000


/**
 * Get matches between keypoints in the reference image and the frame
 * Uses KNN with k =2 to get 2 best keypoints for  based on each descriptor in the reference image. Then it applies Lowe's ratio test to filter out the best matches.
 * Then it tries to use the best match if it is not already used, if it is used, it tries to use the second best match.
 * @param descriptorsRef The descriptors of the reference image
 * @param descriptorsFrame The descriptors of the frame
 * @param matcher The matcher object
 * @return A vector of DMatch objects representing the matches between the keypoints in the reference image and the frame
 */
std::vector<cv::DMatch> getMatches(const cv::Mat& descriptorsRef, const cv::Mat& descriptorsFrame, cv::Ptr<cv::FlannBasedMatcher> matcher) {
    // Match descriptors using k-NN (k=2)
    std::vector<std::vector<cv::DMatch>> knnMatches;
    if (!descriptorsFrame.empty()) {
        matcher->knnMatch(descriptorsRef, descriptorsFrame, knnMatches, 2);
    }
    // Apply Lowe’s Ratio Test and Track Unique Keypoints
    std::vector<cv::DMatch> matches;
    std::set<int> uniqueMatchedPoints;  // Keep track of keypoints already matched
    const float ratioThreshold = 0.7;

    for (const auto& matchPair : knnMatches) {
        if (matchPair.size() == 2 && matchPair[0].distance < ratioThreshold * matchPair[1].distance) {
            int keypointIdx = matchPair[0].trainIdx;  // Index of matched keypoint
            int secondaryKeypointIdx = matchPair[1].trainIdx;  // Index of matched keypoint (2nd)
            // Ensure keypoint is not duplicated
            if (uniqueMatchedPoints.find(keypointIdx) == uniqueMatchedPoints.end()) {
                matches.push_back(matchPair[0]);
                uniqueMatchedPoints.insert(keypointIdx);  // Mark as used
            } else if (uniqueMatchedPoints.find(secondaryKeypointIdx) == uniqueMatchedPoints.end()) { // try the second match if the first one is already used
                matches.push_back(matchPair[1]);
                uniqueMatchedPoints.insert(secondaryKeypointIdx);  // Mark as used
            }
        }
    }
    return matches;
}
/**
 * detects a custom pattern from a reference image and places a marker at its center
 * @param argc The number of command line arguments.
 * @param argv The command line arguments. The first is the path to the reference pattern image.
 */
int main(int argc, char** argv){

    if (argc != 2){
        std::cerr << "Usage: ./detect_features <path_to_reference_pattern_image>" << std::endl;
        return -1;
    }


    std::string patternImagePath = argv[1];
    cv::Mat referenceImage = cv::imread(patternImagePath, cv::IMREAD_GRAYSCALE);
    if (referenceImage.empty()) {
        std::cout << "Error: Could not load the reference image!" << std::endl;
        return -1;
    }
    cv::VideoCapture cap(0);

    if (!cap.isOpened())
    {
        std::cerr << "Error opening video stream" << std::endl;
        return -1;
    }



    cv::Mat frame;

    // Create SURF detector
    int minHessian = SURF_MIN_HESSIAN;
    cv::Ptr<cv::xfeatures2d::SURF> surf = cv::xfeatures2d::SURF::create(minHessian); // Threshold: Adjust for feature detection sensitivity
    // Detect keypoints and descriptors in the reference image
    std::vector<cv::KeyPoint> keypointsRef;
    cv::Mat descriptorsRef;
    surf->detectAndCompute(referenceImage, cv::noArray(), keypointsRef, descriptorsRef);


    // draw keypoints on the reference image and save to disk
    cv::Mat referenceImageWithKeypoints;
    cv::drawKeypoints(referenceImage, keypointsRef, referenceImageWithKeypoints, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    cv::imwrite("reference_image_with_keypoints.jpg", referenceImageWithKeypoints);



    // FLANN-based matcher (used for feature matching)
    cv::Ptr<cv::FlannBasedMatcher> matcher = cv::FlannBasedMatcher::create();

    while (true){
        cap >> frame;

        if (frame.empty())
        {
            std::cerr << "Error reading frame" << std::endl;
            return 1;
        }

        cv::Mat frameGray;
        cv::cvtColor(frame, frameGray, cv::COLOR_BGR2GRAY);

        // Detect keypoints and descriptors in the frame
        std::vector<cv::KeyPoint> keypointsFrame;
        cv::Mat descriptorsFrame;
        surf->detectAndCompute(frameGray, cv::noArray(), keypointsFrame, descriptorsFrame);


        std::vector<cv::DMatch> matches = getMatches(descriptorsRef, descriptorsFrame, matcher);


        if (!matches.empty()) {
            // Compute the center of matched keypoints
            cv::Point2f center(0, 0);
            for (const auto& match : matches) {
                center += keypointsFrame[match.trainIdx].pt;
                // draw the matched keypoints
                cv::circle(frame, keypointsFrame[match.trainIdx].pt, 5, cv::Scalar(0, 255, 0), -1); // Green marker
            }
            center.x /= matches.size();
            center.y /= matches.size();

            // Draw the center marker
            cv::circle(frame, center, 10, cv::Scalar(0, 0, 255), -1); // Red marker
            cv::putText(frame, "Center", center + cv::Point2f(10, -10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 2);
            // put number of matches on screen
            cv::putText(frame, "Matches: " + std::to_string(matches.size()), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 0), 2);
            

        }

        // Show results
        cv::imshow("Frame", frame);

        cv::waitKey(1);

    }
}