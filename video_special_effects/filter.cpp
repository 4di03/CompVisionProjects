/**
 * Adithya Palle
 * Jan 24 2025
 * CS 5330 - Project 1 : Filter implementation
 * 
 * This file contains the implementations of the functions that are used to apply filters to the image.
 */

#include "filter.h"
#include <opencv2/opencv.hpp>
#include <iostream>

class Filter{
    public:
        /**
         * Strategy function for modifying a pixel in the image.
         */
        virtual void modifyPixel(int i, int j, const cv::Mat& src, cv::Mat& dst) = 0;

        /**
         * Returns the datatype of the destination image.
         */
        virtual int getDatatype() = 0;
};

class AlternativeGrayscale : public Filter{
    public:
        /**
         * Applies a custom grayscale filter by summing the RGB values and then modding by 256.
         */
        void modifyPixel(int i, int j, const cv::Mat& src, cv::Mat& dst){
            cv::Vec3b pixel = src.at<cv::Vec3b>(i, j);
            dst.at<uchar>(i, j) = (pixel[0] + pixel[1] + pixel[2]) % 256;
        }

        // because the output is a single channel grayscale image
        int getDatatype(){
            return CV_8UC1;
        }
};

class Sepia: public Filter{
    public:
        /**
         * Applies a Sepia filter by modifying each color using a weighted sum of all three colors
         */
        void modifyPixel(int i, int j, const cv::Mat& src, cv::Mat& dst){
            cv::Vec3b pixel = src.at<cv::Vec3b>(i, j);
            int originalRed = pixel[2];
            int originalGreen = pixel[1];
            int originalBlue = pixel[0];
            int newBlue = originalRed * 0.272 + originalGreen * 0.534 + originalBlue * 0.131;
            int newGreen = originalRed * 0.349 + originalGreen * 0.686 + originalBlue * 0.168;
            int newRed = originalRed * 0.393 + originalGreen * 0.769 + originalBlue * 0.189;
            dst.at<cv::Vec3b>(i, j) = cv::Vec3b(newBlue, newGreen, newRed);

        }

        // because the output is a 3 channel image
        int getDatatype(){
            return CV_8UC3;
        }
};

class NaiveBlur : public Filter{
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
        void modifyPixel(int i, int j, const cv::Mat& src, cv::Mat& dst){
            // check if the pixel is on the border
            if (i < 2 || i >= src.rows - 2 || j < 2 || j >= src.cols - 2){
                dst.at<cv::Vec3b>(i, j) = src.at<cv::Vec3b>(i, j); // just copy it 
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

            dst.at<cv::Vec3b>(i, j) = sum;
        }
        // because this is for a 3 channel image
        int getDatatype(){
            return CV_8UC3;
        }

};


int applyFilter(const cv::Mat& src, cv::Mat& dst, Filter* filter){
    if(src.empty()){
        return -1;
    }
    // copy src so that we dont have to read and write to the same image
    cv::Mat srcCopy = src.clone();
    // reset the dst image (for case where src and dst are the same)
    dst.create(src.size(), filter->getDatatype());
    


    for(int i = 0; i < srcCopy.rows; i++){
        for(int j = 0; j < srcCopy.cols; j++){
            filter->modifyPixel(i, j, srcCopy, dst);
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
    return applyFilter(src, dst, new AlternativeGrayscale());
}

/**
 * Applies a sepia filter to the image.
 * 
 * @param src The source image.
 * @param dst The destination image.
 * @returns 0 if the operation was successful, -1 otherwise.
 */
int sepia(const cv::Mat& src, cv::Mat& dst){
    return applyFilter(src, dst, new Sepia());
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
    return applyFilter(src, dst, new NaiveBlur());

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
 * @param dst The destination image.
 * @returns 0 if the operation was successful, -1 otherwise.
 */
int blur5x5_2( cv::Mat &src, cv::Mat &dst ){
    std::vector<float> kernelVector = {0.1, 0.2, 0.4, 0.2, 0.1};
    return applySeparableFilter(src, dst, kernelVector, kernelVector);
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