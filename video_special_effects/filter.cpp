/**
 * Adithya Palle
 * Jan 24 2025
 * CS 5330 - Project 1 : Filter implementation
 * 
 * This file contains the implementations of the functions that are used to apply filters to the image.
 */

#include "filter.h"
#include "faceDetect/faceDetect.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include "DA2Network.hpp"
#include <cmath>
#define SWIRL_FALLOFF 5 // increasing this value will weaken the swirl effect the further from the cneter of the face
#define MAX_SWIRL (M_PI)
#define DEPTH_ANYTHING_MODEL_PATH "/Users/adithyapalle/work/CS5330/depthAnything/da2-code/model_fp16.onnx"

template <typename PixelType>
class Filter{
    public:
        /**
         * Strategy function for modifying a pixel in the image.
         */
        virtual PixelType modifyPixel(int i, int j, const cv::Mat& src) = 0;

        /**
         * Returns the datatype of the destination image.
         */
        virtual int getDatatype() = 0;
};

class AlternativeGrayscale : public Filter<uchar>{
    public:
        /**
         * Applies a custom grayscale filter by summing the RGB values and then modding by 256.
         */
        uchar modifyPixel(int i, int j, const cv::Mat& src){
            cv::Vec3b pixel = src.at<cv::Vec3b>(i, j);
            //dst.at<uchar>(i, j) = (pixel[0] + pixel[1] + pixel[2]) % 256;
            return (pixel[0] + pixel[1] + pixel[2]) % 256;
        }

        // because the output is a single channel grayscale image
        int getDatatype(){
            return CV_8UC1;
        }
};

class Sepia: public Filter<cv::Vec3b>{
    public:
        /**
         * Applies a Sepia filter by modifying each color using a weighted sum of all three colors
         */
        cv::Vec3b modifyPixel(int i, int j, const cv::Mat& src){
            cv::Vec3b pixel = src.at<cv::Vec3b>(i, j);
            int originalRed = pixel[2];
            int originalGreen = pixel[1];
            int originalBlue = pixel[0];
            int newBlue = originalRed * 0.272 + originalGreen * 0.534 + originalBlue * 0.131;
            int newGreen = originalRed * 0.349 + originalGreen * 0.686 + originalBlue * 0.168;
            int newRed = originalRed * 0.393 + originalGreen * 0.769 + originalBlue * 0.189;
            return cv::Vec3b(newBlue, newGreen, newRed);

        }

        // because the output is a 3 channel image
        int getDatatype(){
            return CV_8UC3;
        }
};

class NaiveBlur : public Filter<cv::Vec3b>{
    private:
        cv::Mat kernel;
    public:

        NaiveBlur(){
            kernel = (cv::Mat_<float>(5, 5) << 1, 2, 4, 2, 1,
                                                2, 4, 8, 4, 2,
                                                4, 8, 16, 8, 4,
                                                2, 4, 8, 4, 2,
                                                1, 2, 4, 2, 1);
            float kernelSum = cv::sum(kernel)[0];

            kernel /= kernelSum;

        }
        /**
         * Applies a 5x5 blur using the valid convolution algorithm (no padding) with a gaussian kernel.
         */
        cv::Vec3b modifyPixel(int i, int j, const cv::Mat& src){
            // check if the pixel is on the border
            if (i < 2 || i >= src.rows - 2 || j < 2 || j >= src.cols - 2){
                return src.at<cv::Vec3b>(i, j); // just copy it 
            }

            cv::Vec3f sum = cv::Vec3f(0, 0, 0); // use a 32 bit float to prevent overflow
            for (int k = -2; k <= 2; k++){
                for (int l = -2; l <= 2; l++){
                    // get the weighted sum of the pixel and the kernel by centering the kernel at the pixel
                    cv::Vec3b pixel = src.at<cv::Vec3b>(i + k, j + l);
                    //std::cout << pixel * (kernel.at<float>(k + 2, l + 2)/kernelSum) << std::endl;
                    sum += pixel * (kernel.at<float>(k + 2, l + 2)); // divide by kernelSum to normalize the kernel
                }
            }

            return static_cast<cv::Vec3b>(sum);
        }
        // because this is for a 3 channel image
        int getDatatype(){
            return CV_8UC3;
        }

};

template <typename PixelType>
class AdjustBrightness : public Filter<PixelType> {
private:
    int delta;
    int datatype;

public:
    AdjustBrightness(int delta, int datatype) : delta(delta), datatype(datatype) {}

    /**
     * Adjusts the brightness of the image by adding delta to each channel.
     */
    PixelType modifyPixel(int i, int j, const cv::Mat& src) override {
        PixelType pixel = src.at<PixelType>(i, j);

        // Adjust brightness for each channel
        for (int c = 0; c < src.channels(); ++c) {
            pixel[c] = static_cast<uchar>(std::min(255, std::max(0,pixel[c] + delta)));
        }

        return pixel;
    }

    /**
     * Returns the data type of the image this filter operates on.
     */
    int getDatatype() override {
        return datatype;
    }
};

class Median : public Filter< cv::Vec3b > {
    private:
        int kernelSize = 3;
    public:

        /**
         * Applies a 5x5 median filter to the image, replacing the channel values in each pixel with the median of the 5x5 neighborhood.
         * Works by accumulating all the red, green, and blue values in the kernel and then sorting them to get the median at the middle of the sorted array.
         */
        cv::Vec3b modifyPixel(int i, int j, const cv::Mat& src) override {
            std::vector<uchar> reds;
            std::vector<uchar> greens;
            std::vector<uchar> blues;

            for (int k = -kernelSize/2; k <= kernelSize/2; k++){
                for (int l = -kernelSize/2; l <= kernelSize/2; l++){
                    cv::Vec3b pixel = src.at<cv::Vec3b>(i + k, j + l);
                    reds.push_back(pixel[2]);
                    greens.push_back(pixel[1]);
                    blues.push_back(pixel[0]);
                }
            }

            std::sort(reds.begin(), reds.end());
            std::sort(greens.begin(), greens.end());
            std::sort(blues.begin(), blues.end());

            return cv::Vec3b(blues[(kernelSize*kernelSize)/2], greens[(kernelSize*kernelSize)/2],reds[(kernelSize*kernelSize)/2]);

        }


        int getDatatype() override {
            return CV_8UC3;
        }

};


class DepthFog: public Filter<cv::Vec3b>{

    private:
        cv::Mat depthValues;
        float k;
        cv::Vec3b fogColor;
    public:
        DepthFog(float k = 4, cv::Vec3b fogColor = cv::Vec3b(128, 128, 128)){
            if(k < 0){
                throw std::invalid_argument("fog density must be greater than or equal to 0");
            }
            this->k = k;
            this->fogColor = fogColor;
        }

        void setDepthValues(cv::Mat depthValues){
            this->depthValues = depthValues;
        }
        /**
         * Applies a depth-based fog to the image using the following formula:
         * 
         * 
         * F=1−exp(−k⋅(1-d))
         * 
         * Output Color=(1−F)⋅Image Color+F⋅Fog Color
         * 
         * where k is the fog density (higher values will lead to more fog), d is the depth value [0,1] with 0 representing further obejcts and 1 represnting closer ones, and Fog Color is the color of the fog (gray).
         * 
         */
        cv::Vec3b modifyPixel(int i, int j, const cv::Mat& src){
            float d = depthValues.at<float>(i, j);
            float F = 1 - std::exp(-k*(1-d));
            cv::Vec3b pixel = src.at<cv::Vec3b>(i, j);
            return (1-F)*pixel + F*fogColor;
        }

        // because the output is a 3 channel image
        int getDatatype(){
            return CV_8UC3;
        }
};

template <typename PixelType>
int applyFilter(const cv::Mat& src, cv::Mat& dst, Filter<PixelType>* filter){
    if(src.empty()){
        std::cout << "Error: Source image is empty" << std::endl;
        return -1;
    }
    // copy src so that we dont have to read and write to the same image
    cv::Mat srcCopy = src.clone();
    // reset the dst image (for case where src and dst are the same)
    dst.create(src.size(), filter->getDatatype());


    for(int i = 0; i < srcCopy.rows; i++){
        PixelType* row = dst.ptr<PixelType>(i);
        for(int j = 0; j < srcCopy.cols; j++){
            row[j] = filter->modifyPixel(i, j, srcCopy);
        }
    }
    return 0;
}

template <typename PixelType>
int applyFilterSlow(const cv::Mat& src, cv::Mat& dst, Filter<PixelType>* filter){
    if(src.empty()){
        std::cout << "Error: Source image is empty" << std::endl;
        return -1;
    }
    // copy src so that we dont have to read and write to the same image
    cv::Mat srcCopy = src.clone();
    // reset the dst image (for case where src and dst are the same)
    dst.create(src.size(), filter->getDatatype());


    for(int i = 0; i < srcCopy.rows; i++){
        for(int j = 0; j < srcCopy.cols; j++){
            dst.at<PixelType>(i, j) = filter->modifyPixel(i, j, srcCopy);
        }
    }
    return 0;
}


/**
 * Applies a custom grayscale filter to the image.
 * Produces the gray value by summing the RGB values and then modding by 256.
 * 
 * @param src The source image.
 * @param dst The destination image.
 * @returns 0 if the operation was successful, -1 otherwise.
 */
int alternativeGrayscale(const cv::Mat& src, cv::Mat& dst){
    return applyFilter<uchar>(src, dst, new AlternativeGrayscale());
}

/**
 * Applies a sepia filter to the image.
 * 
 * @param src The source image.
 * @param dst The destination image.
 * @returns 0 if the operation was successful, -1 otherwise.
 */
int sepia(const cv::Mat& src, cv::Mat& dst){
    return applyFilter<cv::Vec3b>(src, dst, new Sepia());
}

/**
 * Applies a 5x5 blur using the valid convolution algorithm (no padding).
 * 
 * The kernel used is:
 * 
 *  1 2  4 2 1
 *  2 4  8 4 2 
 *  4 8 16 8 4
 *  2 4  8 4 2
 *  1 2  4 2 1
 * 
 * @param src The source image.
 * @param dst The destination image.
 * @returns 0 if the operation was successful, -1 otherwise.
 */
int blur5x5_1( cv::Mat &src, cv::Mat &dst ){
    return applyFilterSlow<cv::Vec3b>(src, dst, new NaiveBlur());

}

/**
 * Applies a separable filter to the image.
 * 
 * if K is a seperable filter, then K = V * H where H and V are 1D filters.
 * This function then does V*(H*src)  which is equivalent to K*src by the associative and commutative properties of convolution.
 * 
 */
int applySeparableFilter(cv::Mat& src, cv::Mat& dst, std::vector<float>& verticalFilter, std::vector<float>& horizontalFilter){
    if (horizontalFilter.size() %2 != 1 || verticalFilter.size() %2 != 1){
        // make sure the filters are odd in dimensions so that we can center them
        return -1;
    }
    if(src.empty()){
        return -1;
    }

    // copy src so that we dont have to read and write to the same image
    cv::Mat srcCopy;

    src.convertTo(srcCopy, CV_32FC3); // convert to float so that we can apply math with proper precision
    // make dst a copy of the data in source so that border pixels are not modified
    src.copyTo(dst);
    dst.convertTo(dst, CV_16SC3); // convert to 16 bit signed int so that we can store the results of the kernel operation

    
    cv::Mat tmp = cv::Mat::zeros(src.size(), CV_32FC3);

    int horizontalCutoff = horizontalFilter.size()/2;

    //std::cout << "Kernel vector: " << kernelVector[0] << " " << kernelVector[1] << " " << kernelVector[2] << " " << kernelVector[3] << " " << kernelVector[4] << std::endl;
    // horizontal filter pass
    for(int i = 0; i < srcCopy.rows; i++){
        cv::Vec3f* origRow = srcCopy.ptr<cv::Vec3f>(i);
        cv::Vec3f* row = tmp.ptr<cv::Vec3f>(i);
        for(int j = horizontalCutoff; j < srcCopy.cols-horizontalCutoff; j++){
            // modify the row in srcCopy as we need to store these results for the vertical pass
            for(int k = -horizontalCutoff; k <= horizontalCutoff; k++){
                row[j] += origRow[j+k] * horizontalFilter[horizontalCutoff + k];
            }
        }
    }

    int verticalCutoff = verticalFilter.size()/2;


    // vertical filter pass
    for (int j = horizontalCutoff; j < tmp.cols - horizontalCutoff; j++){
        for (int i = verticalCutoff; i < tmp.rows - verticalCutoff; i++){

            cv::Vec3f sum = cv::Vec3f(0, 0, 0); 

            for(int k = -verticalCutoff; k <= verticalCutoff; k++){
                sum += tmp.at<cv::Vec3f>(i+k, j) * verticalFilter[verticalCutoff + k];
            }

            // modify dst directly
            dst.at<cv::Vec3s>(i, j) = sum;

        }
    }
    return 0;

}
/**
 * Converts images that are not in [0,255] range to [0,255] range.
 */
void prepareFrameForDisplay(cv::Mat& src, cv::Mat& dst){
    // printf("Source type: %d\n", src.type());
    // printf("Destination type: %d\n", dst.type());
    // printf("CV_16SC3: %d\n", CV_16SC3);
    // printf("CV_8UC3: %d\n", CV_8UC3);

    double minVal;
    minMaxLoc(src, &minVal, nullptr);

    if (src.type() == CV_16SC3){
        if (minVal < 0){
            // if the image has negative values then we appliy pixel * 0.5 + 127.5 to convert the minimum value to 0, else if only the max is greater, we simply scale it down.

            cv::convertScaleAbs(src, dst, 0.5,  127.5); 
        } else {
            // since all values are positive, we can simply convert to 8 bit
            src.convertTo(dst, CV_8UC3);
        }
    } else {
       src.copyTo(dst);
    }

    if (dst.channels() == 1){
        // convert to 3 channel image
        cv::cvtColor(dst, dst, cv::COLOR_GRAY2BGR);
    }
}
/**
 * Applies a 5x5 blur using the valid convolution algorithm (no padding).
 * Applies optimizations to improve performance
 * 
 * The kernel used is:
 * 
 *  1 2  4 2 1
 *  2 4  8 4 2 
 *  4 8 16 8 4
 *  2 4  8 4 2
 *  1 2  4 2 1
 * 
 * @param src The source image.
 * @param dst The destination image (3-channel uchar).
 * @returns 0 if the operation was successful, -1 otherwise.
 */
int blur5x5_2( cv::Mat &src, cv::Mat&dst ){
    cv::Mat blurOut;
    std::vector<float> kernelVector = {0.1, 0.2, 0.4, 0.2, 0.1};
    applySeparableFilter(src, blurOut, kernelVector, kernelVector);

    prepareFrameForDisplay(blurOut, dst);
    return 0;
}

/**
 * Appies a 3x3 sobel X filter to the image to highlight vertical edges.
 * 
 */
int sobelX3x3( cv::Mat &src, cv::Mat &dst ){
    std::vector<float> vert = {0.25,0.5,0.25};
    std::vector<float> horiz = {0.5,0,-0.5};
    return applySeparableFilter(src, dst, vert,horiz);
}

/**
 * Appies a 3x3 sobel Y filter to the image to highlight horizontal edges.
 * 
 */
int sobelY3x3( cv::Mat &src, cv::Mat &dst ){
    std::vector<float> vert = {0.5,0,-0.5};
    std::vector<float> horiz = {0.25,0.5,0.25};

    return applySeparableFilter(src, dst, vert,horiz);
}


/**
 * Computes the gradient magnitude image from two 3 channel gradient images.
 * The output is remains a three channel image.
 * 
 * @param sx The gradient image in the x direction (3 - channel, signed short).
 * @param sy The gradient image in the y direction (3 - channel, signed short).
 * @param dst The destination image (3 - channel, unsigned int).
 * @returns 0 if the operation was successful, -1 otherwise.
 */
int magnitude( cv::Mat &sx, cv::Mat &sy, cv::Mat &dst ){
    if (sx.empty() || sy.empty() || sx.size() != sy.size()){
        return -1;
    }

    // initalize dst as a uchar 3 channel image
    dst.create(sx.size(), CV_8UC3);

    for (int i = 0; i < sx.rows; i++){
        cv::Vec3s* rowX = sx.ptr<cv::Vec3s>(i);
        cv::Vec3s* rowY = sy.ptr<cv::Vec3s>(i);
        cv::Vec3b* rowDst = dst.ptr<cv::Vec3b>(i);

        for (int j = 0;  j < sx.cols; j++){
            cv::Vec3s pixelX = rowX[j];
            cv::Vec3s pixelY = rowY[j];

            // compute the magnitude of the gradient for all 3 channels
            cv::Vec3b magnitude = cv::Vec3s(sqrt(pixelX[0]*pixelX[0] + pixelY[0]*pixelY[0]),
                                            sqrt(pixelX[1]*pixelX[1] + pixelY[1]*pixelY[1]),
                                            sqrt(pixelX[2]*pixelX[2] + pixelY[2]*pixelY[2]));

            rowDst[j] = magnitude;
        }


    }

    return 0;

}

/**
 * Blurs an image and then quantizes the colors to a specified number of levels.
 * 
 * @param src The source image (3-channel, uchar).
 * @param dst The destination image (3-channel, uchar).
 * @param levels The number of quantization levels.
 */
int blurQuantize( cv::Mat &src, cv::Mat &dst, int levels ){
    
    if (blur5x5_2(src, dst) != 0){
        std::cout << "Error applying blur5x5_1" << std::endl;
        return -1;
    }

    // get the number of values in each bucket
    float bucketSize = 256/levels;

    for (int i = 0; i < dst.rows; i++){
        cv::Vec3b* row = dst.ptr<cv::Vec3b>(i);
        for (int j = 0; j < dst.cols; j++){
            cv::Vec3b pixel = row[j];

            // determine which bucket each value for this pixel belongs to
            cv::Vec3b bucketIndices = pixel/bucketSize;
        
            // get the quantized values by multiplying the index with teh bucket size
            cv::Vec3b quantizedPixel = bucketIndices * bucketSize;

            row[j] = quantizedPixel;


        }
    }

    return 0;

}

/**
 * Gets depth values in range [0,255] from the input image. 255 represents the closest object and 0 represents the furthest object.
 * 
 * @param src The source image.
 * @param dst The destination single channel mat.
 * 
 * @returns 0 if the operation was successful, -1 otherwise.
 */
int getDepthValues(cv::Mat&src, cv::Mat &dst){
    // initalize the network once
    static   DA2Network da_net( DEPTH_ANYTHING_MODEL_PATH );
    // scale according to the smaller dimension
    float scale_factor = 256.0 / (src.rows > src.cols ? src.cols : src.rows);
    scale_factor = scale_factor > 1.0 ? 1.0 : scale_factor;
    da_net.set_input( src, scale_factor );
    da_net.run_network( dst, src.size() );

    return 0;
}

/**
 * Produces a depth image from the input image, with inferno colormap applied.
 * 
 * @param src The source image.
 * @param dst The destination image, mroe orange colors represent closer objects.
 * 
 * @returns 0 if the operation was successful, -1 otherwise.
 */
int depth(cv::Mat &src, cv::Mat &dst){
    if(getDepthValues(src, dst) != 0){
        std::cout << "Error getting depth values" << std::endl;
        return -1;
    }

    // dst is now a 0-255 grayscale image, apply the inferno colormap

    cv::applyColorMap(dst, dst, cv::COLORMAP_INFERNO );

    return 0;


}

/**
 * Uses depth information to blur the background of an image.
 * The blur is only applied to pixels that have a depth value below a specific threshold.
 * @param src The source image.
 * @param dst The destination image.
 * @param threshold The depth value threshold.
 * @param processingFunction The function to apply to the image. It must produce a 3 channel uchar image for this to work.
 * 
 * @returns 0 if the operation was successful, -1 otherwise.
 */
int applyToForeground(cv::Mat &src, cv::Mat &dst, int threshold, int (*processingFunction)(const cv::Mat&, cv::Mat&)){


    // apply the thing to the image
    if (processingFunction(src, dst) != 0){
        std::cout << "Error applying processing function" << std::endl;
        return -1;
    }


    cv::Mat depthImage;
    if (getDepthValues(src, depthImage) != 0){
        std::cout << "Error applying depth" << std::endl;
        return -1;
    }

    // iterate thorugh the image and restore the original pixel if the depth value is < threshold (meaning it is further)
    for (int i=0;i < dst.rows;i++){
        cv::Vec3b* srcRow = src.ptr<cv::Vec3b>(i);
        cv::Vec3b* dstRow = dst.ptr<cv::Vec3b>(i);
        uchar* depthRow = depthImage.ptr<uchar>(i);
        for (int j = 0; j < dst.cols; j++){
            // if the thing is closer, restore the original pixel
            if (depthRow[j] < threshold){
                dstRow[j] = srcRow[j];

            }
        }
    }

}

/**
 * Increases the brightness of the RGB image by adding delta to each channel.
 * If the value exceeds 255, it is clamped to 255.
 * @param src The source image (3-channel uchar).
 * @param dst The destination image.
 * @param delta The amount to add to each channel.
 * @template PixelType The datatype of the image.
 * @returns 0 if the operation was successful, -1 otherwise.
 */
int adjustBrightness(cv::Mat& src, cv::Mat& dst, int delta){
    return applyFilter<cv::Vec3b>(src, dst, new AdjustBrightness<cv::Vec3b>(delta, src.type()));
}


/**
 * Applies a 5x5 median filter to the image, replacing the channel values in each pixel with the median of the 5x5 neighborhood.
 * 
 * @param src The source image.
 * @param dst The destination image.
 * @returns 0 if the operation was successful, -1 otherwise.
 */
int medianFilter(cv::Mat &src, cv::Mat &dst){
    return applyFilter<cv::Vec3b>(src, dst, new Median());
}


/**
 * Applies depth-based fog to the image.
 */
int depthFog(cv::Mat &src, cv::Mat &dst){

    DepthFog* filter = new DepthFog();

    cv::Mat depthValues;
    if (getDepthValues(src, depthValues) != 0){
        std::cout << "Error getting depth values" << std::endl;
        return -1;
    }

    // normalize the depth values to [0,1]
    depthValues.convertTo(depthValues, CV_32F);
    depthValues /= 255.0;

    filter->setDepthValues(depthValues);

    return applyFilter<cv::Vec3b>(src, dst, filter);
}



/**
 * Applies a swirl effect in the given face region of the image.
 * We apply an inverse mapping to the pixels in the face region to create the swirl effect without having holes in the distortion.
 * The swirl is strongest in the center, and decreases as we move away from the center.
 * @param src The source image.
 * @param dst The destination image, which should already be a copy of the source image foor which the facial data exists.
 * @param face The face region, represented by a cv::Rect.
 * @returns 0 if the operation was successful, -1 otherwise.
 */
int applySwirl(cv::Mat& src ,cv::Mat &dst, cv::Rect& face){
    
    cv::Point center = cv::Point(face.x + face.width/2, face.y + face.height/2);

    int topRightX = face.x + face.width;
    int topRightY = face.y;
    float maxRadius = norm(center - cv::Point(topRightX, topRightY));
    for (int i =face.y; i<= face.y+face.height; i++){
        cv::Vec3b* dstRow = dst.ptr<cv::Vec3b>(i);
        for (int j = face.x ; j < face.x+face.width; j++){
            int y = i - center.y;
            int x = j - center.x;
            double r = sqrt(x*x + y*y);
            double angle = std::atan2(y, x);
            // rotate the pixel by increasing its angle, with a maximum rotation of MAX_SWIRL radians at the furthest points
            double prevAngle = angle -  (MAX_SWIRL *(std::exp(-(r/maxRadius)*SWIRL_FALLOFF)));

            // we refer to these points as previous points as they are the result of doing the inverse of the rotation, so we can get the pixel value from the previous point
            int prevX = r * std::cos(prevAngle) + center.x;
            int prevY = r * std::sin(prevAngle) + center.y;

            dstRow[j] = src.at<cv::Vec3b>(prevY, prevX);
        }

    }

    return 0;

}

/**
 * Applies a Swirl effect in the detected facial region of the image.
 * This essentially applies a rotation to the pixels in the facial region using the spherical coordinate system, 
 * and the rotation is dependent on the distance from the center of the facial region.
 * @param src The source image.
 * @param dst The destination image.
 * @returns 0 if the operation was successful, -1 otherwise.
 */
int faceSwirl(cv::Mat &src, cv::Mat &dst){

    cv::Mat srcCopy = src.clone();// copy src so that we dont have to read and write to the same image

    src.copyTo(dst);
    std::vector<cv::Rect> faces;
    cv::Mat grey;
    cv::cvtColor(srcCopy, grey, cv::COLOR_BGR2GRAY);
    detectFaces(grey, faces);


    for (cv:: Rect face : faces){

        if (applySwirl(src, dst, face) != 0){
            std::cout << "Error applying swirl" << std::endl;
            return -1;
        }

    }

    return 0;

}