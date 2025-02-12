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
 * @return The size of the region.
 */
int dfs(cv::Mat& regionMap, std::pair<int,int> loc, int regionId){

    // make a stack to keep track of the neighbors to color
    std::stack<std::pair<int,int>> stack;
    stack.push(loc);
    int regionSize = 0;
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
        regionSize++;


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

    return regionSize;
}


/**
 * draws the axis of least central moment for each region in the region map.
 * The axis of least central moment is the eigenvector corresponding to the smallest eigenvalue of the covariance matrix (PCA) of the region.
 * It represents the axis along the "shortest" direction of the region.
 * @param src The source image.
 * @param dst The destination image. In the end this will be the source image with the axis of least central moment drawn on it. Thus function will overwrite any existing data in the destination image.
 * @param regionMask Binary mask of the region of interest.
 */
 void drawAxisOfLeastCentralMoment(const cv::Mat& src, cv::Mat& dst, const cv::Mat& regionMask) {
    if (src.empty()){
        std::cout << "Error: Source image is empty" << std::endl;
        exit(-1);
    }
    if (dst.empty()){
        std::cout << "Error: Destination image is empty" << std::endl;
        exit(-1);
    }
    if (regionMask.empty()){
        std::cout << "Error: Region mask is empty" << std::endl;
        exit(-1);
    }
    if (regionMask.channels() != 1){
        std::cout << "Error: Region mask is not 1 channel" << std::endl;
        exit(-1);
    }

    cv::Mat mask = regionMask;

    std::vector<std::vector<cv::Point>> contours; 
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE); // get boundary of the region


    if (contours.size() != 1){
        printf("Error: Expected 1 contour, got %d\n", static_cast<int>(contours.size()));
        exit(-1);
    }

    cv::Moments mu = cv::moments(contours[0]);

    // Compute covariance matrix from second-order moments
    double mu20 = mu.mu20 / mu.m00;  // Normalize by m00 (area)
    double mu02 = mu.mu02 / mu.m00;
    double mu11 = mu.mu11 / mu.m00;

    cv::Mat covariance = (cv::Mat_<double>(2, 2) << mu20, mu11, mu11, mu02);
    cv::Mat eigenvalues, eigenvectors;
    cv::eigen(covariance, eigenvalues, eigenvectors);

    // The eigenvector corresponding to the smallest eigenvalue is the axis of least moment (the 2nd column of the eigenvectors matrix as that is the vector corresponding to the minor eigenvalue)
    cv::Point2d leastMomentAxis(eigenvectors.at<double>(1, 0), eigenvectors.at<double>(1, 1));


    // get centroid of the region using the moments since we already have them
    cv::Point2d centroid(mu.m10 / mu.m00, mu.m01 / mu.m00);

    // draw the axis of least moment as an arrow
    cv::arrowedLine(dst, centroid, centroid + 100 * leastMomentAxis, cv::Scalar(0, 0, 255), 4);

}
/**
 * Gets the oriented bounding box of the region in the image.
 * @param src The source image.
 * @param regionMask The binary mask of the region of interest.
 * @return The oriented bounding box of the region.
 */
cv::RotatedRect getBoundingBox(const cv::Mat& src, const cv::Mat& regionMask){
    if (src.empty() || regionMask.empty() ){
        std::cout << "Error: Image or region mask is empty" << std::endl;
        exit(-1);
    }
    std::vector<std::vector<cv::Point>> contours; 
    cv::findContours(regionMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE); // get boundary of the region


    if (contours.size() != 1){
        printf("Error: Expected 1 contour, got %d\n", static_cast<int>(contours.size()));
        exit(-1);
    }

    return cv::minAreaRect(contours[0]);
}

/**
 * Draws the oriented bounding box of the region in the image.
 * @param dst The destination image on which the object is drawn;
 * @param boundingBox The oriented bounding box of the region.
 */
void drawBoundingBox(cv::Mat& dst, const cv::RotatedRect& boundingBox){


    cv::Point2f vertices[4];
    boundingBox.points(vertices);

    for (int i = 0; i < 4; i++){
        cv::line(dst, vertices[i], vertices[(i+1)%4], cv::Scalar(255,20,0), 2);
    }

}



/**
 * Creates a region map, assigning a unique integer region id for each foreground region in the mask.
 * 
 * 
 * @param image The image to segment (3 channel uchar image).
 * @param mask The mask of the object in the image (1 channel uchar image).
 * @return A region map where each unique masked foreground region is assigned a unique integer id (1 channel uchar image). 
 *      Also return a map of region id to region size.
 */
RegionData getRegionMapFromForegroundMask(const cv::Mat& image, const cv::Mat& mask){
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

    std::unordered_map<int,int> regionSizes;



    // initalize a set of seen colors to keep track of which colors have been used
    std::set<std::tuple<int,int,int>> seenColors;
    int curRegionId = 1;
    for (int i = 0; i< image.rows; i++){
        for (int j = 0; j < image.cols; j++){



            if (regionMap.at<unsigned char>(i,j) != 255){ // if its not 255, it either means it is not in the mask or has already been colored
                continue;
            }

            // color the pixels (will return right away if the pixel is not in the mask or has already been colored)
            int regionSize = dfs(regionMap, std::pair<int,int>(i,j), curRegionId);
            regionSizes[curRegionId] = regionSize;
            curRegionId++;


        }
    }

    return RegionData{regionMap, regionSizes};

}



/**
 * Colors the connected components in the image with different colors.
 * 
 * @param image The image to segment (3 channel uchar image).
 * @param regionMap The region map where each unique masked foreground region is assigned a unique integer id (1 channel uchar image).
 * @return The image with the connected components colored differently (3 channel uchar image).
 */
cv::Mat colorConnectedComponents(const cv::Mat& image, const cv::Mat& regionMap){
    if (image.empty() || regionMap.empty()){
        std::cout << "Error: Image or regionMap is empty" << std::endl;
        exit(-1);
    }

    if (image.channels() != 3){
        std::cout << "Error: Image is not 3 channel" << std::endl;
        exit(-1);
    }


    cv::Mat colored = image.clone();

    std::unordered_map<int,cv::Vec3b> regionColors;
    // initalize a set of seen colors to keep track of which colors have been used
    std::set<std::tuple<int,int,int>> seenColors;
    for (int i = 0; i< image.rows; i++){
        cv::Vec3b* coloredRow = colored.ptr<cv::Vec3b>(i);
        const cv::Vec3b* imageRow = image.ptr<cv::Vec3b>(i);
        const unsigned char* regionRow = regionMap.ptr<unsigned char>(i);
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
 * Creates a region map, assigning a unique integer region id for each foreground region in the image
 * Get the mask of the foreground to compute regions in the image using ISODATA to threshold the image based on the two dominant colors.
 * 
 * @param image The image to segment (3 channel uchar image).
 * @return A region map where each unique masked foreground region is assigned a unique integer id (1 channel uchar image).
 */
RegionData getRegionMap(const cv::Mat& image){
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
    return getRegionMapFromForegroundMask(image, mask);

}



cv::Mat drawFeatures(const cv::Mat& image, const cv::Mat& regionMap, int regionId){

    cv::Mat featuresImage = image.clone();


    cv::Mat largestRegionMask = (regionMap == regionId);

    cv::RotatedRect boundingBox = getBoundingBox(image, largestRegionMask);
    drawBoundingBox(featuresImage, boundingBox);
    drawAxisOfLeastCentralMoment(image, featuresImage, largestRegionMask);

    return featuresImage;

}
/**
 * Calculates the region features for the given region in the image.
 * @param image The image to get the object mask of (3 channel uchar image).
 * @param regionMap The region map where each unique masked foreground region is assigned a unique integer id (1 channel uchar image).
 * @param regionId The id of the region to get the features of.
 * @return The region features as a RegionFeatureVector.
 */
RegionFeatureVector getRegionFeatures(const cv::Mat& image, const cv::Mat& regionMap, int regionId){

    cv::Mat mask = (regionMap == regionId);
    cv::RotatedRect boundingBox =getBoundingBox(image, mask);
    // get area and perimeter of the region using contours
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (contours.size() != 1){
        printf("Error: Expected 1 contour, got %d\n", static_cast<int>(contours.size()));
        exit(-1);
    }
    float perimeter = cv::arcLength(contours[0], true);
    float area = cv::contourArea(contours[0]);
    
    // get circularity of the region
    float circularity = 4 * M_PI * area/ (perimeter * perimeter);

    // get the percentage of the bounding box filled by the region
    float bboxPctFilled = (area / (boundingBox.size.width * boundingBox.size.height)) * 100;

    // get the aspect ratio of the bounding box (max side length / min side length)
    float bboxAspectRatio = std::max(boundingBox.size.width, boundingBox.size.height) / std::min(boundingBox.size.width, boundingBox.size.height);

    // get the mean color of the region
    cv::Scalar meanColor = cv::mean(image, mask);
    cv::Vec3b meanColorVec = cv::Vec3b(meanColor[0], meanColor[1], meanColor[2]);
    return RegionFeatureVector{bboxPctFilled, bboxAspectRatio, circularity, meanColorVec};
}


/**
 * Gets the object features for the largest region in the image.
 * @param image The image to get the object mask of (3 channel uchar image).
 * @return The object features as a RegionFeatureVector.
 */
RegionFeatureVector getObjectFeatures(const cv::Mat& image){
    RegionData data = getRegionMap(image);
    cv::Mat regionMap = data.regionMap;
    std::unordered_map<int,int> regionSizes = data.regionSizes;
    
    // get id of max size region
    int largestRegionId = 0;
    int largestRegionSize = 0;
    for (auto it = regionSizes.begin(); it != regionSizes.end(); it++){
        if (it->second > largestRegionSize){
            largestRegionSize = it->second;
            largestRegionId = it->first;
        }
    }
    RegionFeatureVector features =  getRegionFeatures(image, regionMap, largestRegionId);
    return features;
}

/**
 * outputs a tinted region image.
 * 
 * @param image The image to get the object mask of (3 channel uchar image).
 * @param regionMap The region map where each unique masked foreground region is assigned a unique integer id (1 channel uchar image).
 * @return The mask of the object in the image (1 channel uchar image).
 */
cv::Mat segmentObjects(const cv::Mat& image, const cv::Mat& regionMap){
    cv::Mat segmented = colorConnectedComponents(image,regionMap);

    return segmented;
}


/**
 * Run all the object recognition tasks on the given image.
 * @param imgPath The path to the image.
 */
void runObjectRecognition(std::string imgPath){
    // get basename of the path
    std::string imageFileName = imgPath.substr(imgPath.find_last_of("/\\") + 1); 

    cv::Mat image = cv::imread(imgPath);
    if (image.empty()){
        std::cout << "Error: Image not found" << std::endl;
        exit(-1);
    }

    RegionData data = getRegionMap(image);
    cv::Mat regionMap = data.regionMap;
    std::unordered_map<int,int> regionSizes = data.regionSizes;
    
    // get id of max size region
    int largestRegionId = 0;
    int largestRegionSize = 0;
    for (auto it = regionSizes.begin(); it != regionSizes.end(); it++){
        if (it->second > largestRegionSize){
            largestRegionSize = it->second;
            largestRegionId = it->first;
        }
    }

    cv::Mat featuresImage = drawFeatures(image, regionMap, largestRegionId);

    // save features image to file
    std::string featuresFileName = "output/features_" + imageFileName;
    printf("Writing to %s\n", featuresFileName.c_str());
    cv::imwrite(featuresFileName, featuresImage);
    RegionFeatureVector features =  getRegionFeatures(image, regionMap, largestRegionId);
    std::vector<float> featureVec = features.toVector();
    // save feature vector to file

    // get basename exclduing extension of imageFileName
    std::string fvecFileName = "image_features/" + imageFileName.substr(0, imageFileName.find_last_of(".")) + ".features";
    printf("Writing features to %s\n", fvecFileName.c_str());
    features.save(fvecFileName);
    // save segmented image to a file
    cv::Mat segmented = segmentObjects(image, regionMap);
    std::string outputName = "output/segmented_" + imageFileName ;
    printf("Writing to %s\n", outputName.c_str());
    cv::imwrite(outputName, segmented);
}