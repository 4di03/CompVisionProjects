/**
 * Adithya Palle
 * Feb 7 2025
 * CS 5330 - Project 3 : Real-time 2D Object Recognition
 * 
 * Implementation file for the thresholding function used to segment the object.
 */
#include "thresholding.h"
#include "kmeans.h"
#define NUM_MEANS 5 // number of means to use for kmeans in ISODATA
#define SATURATION_THRESHOLD // threshold for saturation increase

/**
 * Run the ISODATA algorithm on the given image to find the pixel in between the two dominant color means.
 * The algorithm runs kmeans with k = NUM_MEANS to get the NUM_MEANS means,  and then pick the value in between the two colors with lowest overall brightness
 * This function assumes the object in the image is on a white background, so there are only 2 dominant colors in the image.
 * 
 * @param image The image to run the ISODATA algorithm on (3 channel uchar image).
 * @return The pixel in between the two dominant color means (3 channel uchar pixel).
 */
cv::Vec3b isodata(const cv::Mat& image){
    if (image.empty()){
        std::cout << "Error: Image is empty" << std::endl;
        exit(-1);
    }

    std::vector<cv::Vec3b> means = kmeans(image, NUM_MEANS);


    // sort means based on brightness (V value in HSV space)
    std::sort(means.begin(), means.end(), [](cv::Vec3b a, cv::Vec3b b){
        return std::max({a[0],  a[1] , a[2]}) < std::max({b[0] + b[1] + b[2]});
    });

    // get the pixel in between the two darkest colors

    cv::Vec3b mean1 = means[0];
    cv::Vec3b mean2 = means[1];
    return cv::Vec3b((mean1[0] + mean2[0]) / 2, (mean1[1] + mean2[1]) / 2, (mean1[2] + mean2[2]) / 2);


}
/**
 * Increases the saturation for already saturated areas of the image, making more colorful areas vibrant
 * 
 * @param image The image to increase the saturation of (3 channel uchar image).
 * @param brightnessFactor The factor to change the brightness by.
 * @return a new image with the saturation increased.
 */
cv::Mat increaseSaturation(const cv::Mat& image, float brightnessFactor){
    cv::Mat saturated;
    cv::Mat hsvImage;
    cv::cvtColor(image, hsvImage, cv::COLOR_BGR2HSV);  // Convert to HSV

    std::vector<cv::Mat> channels;
    cv::split(hsvImage, channels);  // Split HSV channels (H, S, V)

    cv::Mat& S = channels[1];  // Saturation channel
    cv::Mat& V = channels[2];  // Value channel

    for (int i = 0; i < image.rows; i++) {
        unsigned char* rowS = S.ptr<unsigned char>(i);
        unsigned char* rowV = V.ptr<unsigned char>(i);
        for (int j = 0; j < image.cols; j++) {
            if (rowS[j]> 100) {  // Only modify dark pixels
                int newValue = rowV[j] * brightnessFactor;
                rowV[j] = 255;//rowV[j] * brightnessFactor; // Clamp to 255
            }else{
                rowV[j] = 0;
            }
        }
    }



    cv::merge(channels, hsvImage);  // Merge modified channels
    cv::cvtColor(hsvImage, saturated, cv::COLOR_HSV2BGR); // convert back to bgr

    return saturated;
}

/**
 * Get the mask of the object in the image using ISODATA to threshold the image based on the two dominant colors.
 * 
 * @param image The image to get the object mask of (3 channel uchar image).
 * @return The mask of the object in the image (1 channel uchar image).
 */
cv::Mat getObjectMask(const cv::Mat& image){
    if (image.empty()){
        std::cout << "Error: Image is empty" << std::endl;
        exit(-1);
    }
    // skipping saturation because it seems to make isodata algorithm obsolete
    cv::Mat processedImage = image; //increaseSaturation(image, 0.01);

    // get the pixel in between the two dominant color values
    cv::Vec3b meanPixel = isodata(processedImage);

    cv::Mat mask = cv::Mat::zeros(processedImage.size(), CV_8UC1);

    for (int i = 0 ; i < processedImage.rows; i++){
        const cv::Vec3b* row = processedImage.ptr<cv::Vec3b>(i);
        unsigned char* maskRow = mask.ptr<unsigned char>(i);
        for (int j = 0; j < processedImage.cols; j++){
            cv::Vec3b pixel = row[j];
            if (pixel[0] < meanPixel[0] && pixel[1] < meanPixel[1] && pixel[2] < meanPixel[2]){
                maskRow[j] = 255; // since the pixel is above the middle of the means, it means it belongs to the lighter color region which is probably the foreground since the background is white
            }
        }
    }

    return mask;
}