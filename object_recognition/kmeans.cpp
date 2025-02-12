/**
 * Adithya Palle
 * Feb 8 2025
 * CS 5330 - Project 3 : Real-time 2D Object Recognition
 * 
 * Implementation file for my kmeans implementation.
 */
#include <iostream>
#include <opencv2/opencv.hpp>
#include "kmeans.h"
/**
 * Get the clusters of pixels based on the means.
 * @param image The image to get the clusters from.
 * @param means The means of the clusters.
 * @param k The number of clusters.
 * @return The clusters of pixels.
 */
std::vector<std::vector<cv::Vec3b>> getClusters(const cv::Mat& image, const std::vector<cv::Vec3b>& means, int k){
    // points in each cluster (reinitialize at each iteration)
    std::vector<std::vector<cv::Vec3b>> clusters(k);
    // calculate which pixel belongs to which cluster
    for(int i = 0; i < image.rows; i++){
        const cv::Vec3b* row = image.ptr<cv::Vec3b>(i);
        for (int j = 0; j < image.cols; j++){
            cv::Vec3b pixel = row[j];
            // find the closest mean to the pixel
            // set max distance to infinity
            float minDist = INT_MAX;
            int minIndex = 0;
            for (int m = 0; m < k; m++){
                cv::Vec3b mean = means[m];

                // use SSD as the distance metric
                float dist = (mean[0] - pixel[0]) * (mean[0] - pixel[0]) + (mean[1] - pixel[1]) * (mean[1] - pixel[1]) + (mean[2] - pixel[2]) * (mean[2] - pixel[2]);
                if (dist < minDist){
                    minDist = dist;
                    minIndex = m;
                }
            }


            clusters[minIndex].push_back(pixel);
        }
    }

    return clusters;

}


/**
 * Get the new means for each cluster.
 * @param clusters The clusters of pixels.
 * @return The new means for each cluster.
 */
std::vector<cv::Vec3b> getNewMeans(const std::vector<std::vector<cv::Vec3b>>& clusters){
    std::vector<cv::Vec3b> newMeans(clusters.size());
    // calculate the new means of the clusters
    for (int i =0; i < clusters.size() ; i++){

        std::vector<cv::Vec3b> cluster = clusters[i];
        if (cluster.empty()){
            // if the cluster is empty, set the mean to a random pixel
            newMeans[i] = cv::Vec3b(rand() % 256, rand() % 256, rand() % 256);
            continue;
        }
        // get the mean of the cluster
        cv::Vec3f newMean(0, 0, 0);
        for (cv::Vec3b pixel : cluster){
            newMean[0] += pixel[0];
            newMean[1] += pixel[1];
            newMean[2] += pixel[2];
        }
        newMean[0] /= cluster.size();
        newMean[1] /= cluster.size();
        newMean[2] /= cluster.size();

        newMeans[i] = static_cast<cv::Vec3b>(newMean);
    }

    return newMeans;

}

/**
 * Get the summed SSD betwen the old and new means.
 * @param means The old means.
 * @param newMeans The new means.
 * @return The summed SSD between the old and new means.
 */
float getSumSSD(const std::vector<cv::Vec3b>& means, const std::vector<cv::Vec3b>& newMeans){
    float sum = 0;
    for (int i = 0; i < means.size(); i++){
        sum += (means[i][0] - newMeans[i][0]) * (means[i][0] - newMeans[i][0]) + (means[i][1] - newMeans[i][1]) * (means[i][1] - newMeans[i][1]) + (means[i][2] - newMeans[i][2]) * (means[i][2] - newMeans[i][2]);
    }
    return sum;
}


/**
 * Run the kmeans algorithm on the given image to find the k dominant colors.
 * 
 * @param image The image to run the kmeans algorithm on (3 channel uchar image).
 * @param k The number of dominant colors to find.
 * @return A vector of k dominant colors (3 channel uchar pixels).
 */
std::vector<cv::Vec3b> kmeans(const cv::Mat& image, int k){
    if (image.empty()){
        std::cout << "Error: Image is empty" << std::endl;
        exit(-1);
    }


    int epsilon = k * 3; //  we tolerate differnece of 6 SSD per cluster

    printf("Doing kmeans with k = %d and epsilon = %d\n", k, epsilon);

    float totalChange = 0;
    // mean of each cluster
    std::vector<cv::Vec3b> means(k);
    for (int i = 0; i < k; i++){
        // pick random pixels as the initial means
        means[i] = image.at<cv::Vec3b>(rand() % image.rows, rand() % image.cols);
    }
    do{

        // seperate pixels into clusters based on SSD to means
        std::vector<std::vector<cv::Vec3b>> clusters = getClusters(image,means, k);

        // calculate new means from clusters
        std::vector<cv::Vec3b> newMeans = getNewMeans(clusters);

        // get change in means
        totalChange = getSumSSD(means, newMeans);

        // update means
        means = newMeans;

        //printf("Current total change: %f\n", totalChange);
    } while (totalChange > epsilon);


    return means;
}
