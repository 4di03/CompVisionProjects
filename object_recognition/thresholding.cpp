/**
 * Adithya Palle
 * Feb 7 2025
 * CS 5330 - Project 3 : Real-time 2D Object Recognition
 * 
 * Implementation file for the thresholding function used to segment the object.
 */
#include "thresholding.h"
#include "kmeans.h"
#include <iostream>
#define NUM_MEANS 5 // number of means to use for kmeans in ISODATA
#define SATURATION_THRESHOLD 100 // saturation threshold for darkening saturated areas
#define NUM_EROSION_ITERATIONS 1
#define NUM_DILATION_ITERATIONS 5
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
 * Darkens saturated areas in the image, seperating the foreground from the background.
 * 
 * @param image The image to increase the saturation of (3 channel uchar image).
 * @param brightnessFactor The factor to change the brightness by.
 * @return a new image with the saturation increased.
 */
cv::Mat darkenSaturatedAreas(const cv::Mat& image, float brightnessFactor){
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
            if (rowS[j]> SATURATION_THRESHOLD) {  // Only modify dark pixels
                int newValue = rowV[j] * brightnessFactor;
                rowV[j] = newValue;
            }
        }
    }



    cv::merge(channels, hsvImage);  // Merge modified channels
    cv::cvtColor(hsvImage, saturated, cv::COLOR_HSV2BGR); // convert back to bgr

    return saturated;
}


/**
 * Applies opening (erosion followed by dilation) to the mask to remove noise.
 * @param mask The mask to dilate (1 channel uchar image).
 * @return The processed (eroded and then dilated) mask (1 channel uchar image).
 */
cv::Mat cleanup(const cv::Mat& mask){
    cv::Mat dilated = mask.clone();

    // Define a 4-connected structuring element (cross-shaped kernel)
    cv::Mat erosionKernel = (cv::Mat_<uchar>(3,3) <<
    0, 1, 0,
    1, 1, 1,
    0, 1, 0);  

    for (int i = 0; i < NUM_EROSION_ITERATIONS; i++){
        cv::erode(dilated, dilated, erosionKernel);
    }

    // Define a 8-connected structuring element (cross-shaped kernel)
    cv::Mat dilationKernel = (cv::Mat_<uchar>(3,3) <<
    1, 1, 1,
    1, 1, 1,
    1, 1, 1);  
    for (int i = 0; i < NUM_DILATION_ITERATIONS; i++){
        cv::dilate(dilated, dilated, dilationKernel);
    }
    return dilated;

}

/**
 * Checks if two pixels are equal.
 * @param a The first pixel.
 * @param b The second pixel.
 * @return True if the pixels are equal, false otherwise.
 */
bool pixelsEqual(const cv::Vec3b& a, const cv::Vec3b& b){
    return a[0] == b[0] && a[1] == b[1] && a[2] == b[2];
}

/**
 * Depth first search to c assign a region id to the connected component starting at the given pixel.
 * @param regionMap The region map to assign the region id to (1 channel uchar image). This should contain 0 for the background initally (meaning it is a copy of the mask).
 * @param loc The location of the pixel to start the DFS at. (i,j)
 * @param regionId The region id to assign to the connected component.
 */
void dfs(cv::Mat& regionMap, std::pair<int,int> loc, int regionId){

    // make a stack to keep track of the neighbors to color
    std::stack<std::pair<int,int>> stack;
    stack.push(loc);

    while (!stack.empty()){
        std::pair<int,int> loc = stack.top();
        stack.pop(); // returns void
        int i = loc.first;
        int j = loc.second;
        // check that the pixel is white (meaning it is in the mask and has not already been colored/tinted)
        if ( regionMap.at<unsigned char>(i,j) != 255){
            continue;
        }
        // if we are on the last region, we don't need to check for connected components as the regionMap is already filled with the region id
        if (regionId == 255){
            continue;
        }
    
        // check that the pixel is in bounds
        if (i < 0 || i >= regionMap.rows || j < 0 || j >= regionMap.cols){
            continue;
        }
    

        // color the pixel
        regionMap.at<unsigned char>(i,j) = regionId;


        // try and color neighbors if they are connected
        for (int k = -1; k <= 1; k++){
            for (int l = -1; l <= 1; l++){
                if (k == 0 && l == 0){ // no point in coloring the same pixel
                    continue;
                }
                stack.push(std::pair<int,int>(i+k, j+l));
            }
        }
        }

    return;
}

/**
 * Creates a region map, assigning a unique integer region id for each foreground region in the mask.
 * 
 * 
 * @param image The image to segment (3 channel uchar image).
 * @param mask The mask of the object in the image (1 channel uchar image).
 * @return A region map where each unique masked foreground region is assigned a unique integer id (1 channel uchar image).
 */
cv::Mat getRegionMap(const cv::Mat& image, const cv::Mat& mask){
    if (image.empty() || mask.empty()){
        std::cout << "Error: Image or mask is empty" << std::endl;
        exit(-1);
    }
    if (image.rows != mask.rows || image.cols != mask.cols){
        std::cout << "Error: Image and mask are not the same size" << std::endl;
        exit(-1);
    }

    if (image.channels() != 3){
        std::cout << "Error: Image is not 3 channel" << std::endl;
        exit(-1);
    }
    if (mask.channels() != 1){
        std::cout << "Error: Mask is not 1 channel" << std::endl;
        exit(-1);
    }


    cv::Mat regionMap = mask.clone();




    // initalize a set of seen colors to keep track of which colors have been used
    std::set<std::tuple<int,int,int>> seenColors;
    int curRegionId = 1;
    for (int i = 0; i< image.rows; i++){
        for (int j = 0; j < image.cols; j++){



            if (regionMap.at<unsigned char>(i,j) != 255){ // if its not 255, it either means it is not in the mask or has already been colored
                continue;
            }

            // color the pixels (will return right away if the pixel is not in the mask or has already been colored)
            dfs(regionMap, std::pair<int,int>(i,j), curRegionId);
            curRegionId++;


        }
    }

    return regionMap;

}



/**
 * Creates a segmentaiton where each unique masked foreground region is colored differently.
 * Regions are considered connected if there is an adjacent foreground pixel in 8 directions.
 * 
 * 
 * @param image The image to segment (3 channel uchar image).
 * @param mask The mask of the object in the image (1 channel uchar image).
 * @return The image with the connected components colored differently (3 channel uchar image).
 */
cv::Mat colorConnectedComponents(const cv::Mat& image, const cv::Mat& mask){
    if (image.empty() || mask.empty()){
        std::cout << "Error: Image or mask is empty" << std::endl;
        exit(-1);
    }
    if (image.rows != mask.rows || image.cols != mask.cols){
        std::cout << "Error: Image and mask are not the same size" << std::endl;
        exit(-1);
    }

    if (image.channels() != 3){
        std::cout << "Error: Image is not 3 channel" << std::endl;
        exit(-1);
    }
    if (mask.channels() != 1){
        std::cout << "Error: Mask is not 1 channel" << std::endl;
        exit(-1);
    }


    cv::Mat colored = image.clone();

    cv::Mat regionMap = getRegionMap(image, mask);
    std::unordered_map<int,cv::Vec3b> regionColors;
    // initalize a set of seen colors to keep track of which colors have been used
    std::set<std::tuple<int,int,int>> seenColors;
    for (int i = 0; i< image.rows; i++){
        cv::Vec3b* coloredRow = colored.ptr<cv::Vec3b>(i);
        const cv::Vec3b* imageRow = image.ptr<cv::Vec3b>(i);
        unsigned char* regionRow = regionMap.ptr<unsigned char>(i);
        for (int j = 0; j < image.cols; j++){

            if (regionRow[j] == 0){ // if it's 0, it either means it is not in the foreground
                continue;
            }

            int regionId = regionRow[j];

            cv::Vec3b color;
            // check if the region has already been colored
            if (regionColors.find(regionId) == regionColors.end()){
                int c1 = rand() % 256;
                int c2 = rand() % 256;
                int c3 = rand() % 256;
                std::tuple<int,int,int> colorVals = std::make_tuple(c1,c2,c3);
                while (seenColors.find(colorVals) != seenColors.end()){ // to make sure we don't use the same color twice
                    int c1 = rand() % 256;
                    int c2 = rand() % 256;
                    int c3 = rand() % 256;
                    colorVals = std::make_tuple(c1,c2,c3);
                }
                
                seenColors.insert(colorVals);
                regionColors[regionId] = cv::Vec3b(c1,c2,c3);
                color = cv::Vec3b(c1,c2,c3);
            }else{
                color = regionColors[regionId];
            }

            // tint the pixel if it is in the mask
            coloredRow[j] = imageRow[j] * 0.5 + color * 0.5; 


        }
    }

    return colored;

}



/**
 * Get the mask of the object in the image using ISODATA to threshold the image based on the two dominant colors.
 * 
 * @param image The image to get the object mask of (3 channel uchar image).
 * @return The mask of the object in the image (1 channel uchar image).
 */
cv::Mat segmentObjects(const cv::Mat& image){
    if (image.empty()){
        std::cout << "Error: Image is empty" << std::endl;
        exit(-1);
    }
    cv::Mat processedImage =darkenSaturatedAreas(image, 0.01);

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

    mask = cleanup(mask);

    cv::Mat segmented = colorConnectedComponents(image, mask);

    return segmented;
}