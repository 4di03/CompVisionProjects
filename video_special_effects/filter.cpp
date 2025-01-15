/**
 * Adithya Palle
 * Jan 24 2025
 * CS 5330 - Project 1 : Filter implementation
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
    if(src.empty()){
        return -1;
    }

    // copy src so that we dont have to read and write to the same image
    cv::Mat srcCopy;

    src.convertTo(srcCopy, CV_32FC3); // convert to float so that we can apply math with proper precision
    // make dst a copy of the data in source so that border pixels are not modified
    src.copyTo(dst);
    

    std::vector<float> kernelVector= {0.1,0.2,0.4,0.2,0.1};

    //std::cout << "Kernel vector: " << kernelVector[0] << " " << kernelVector[1] << " " << kernelVector[2] << " " << kernelVector[3] << " " << kernelVector[4] << std::endl;
    // horizontal filter pass
    for(int i = 0; i < srcCopy.rows; i++){
        cv::Vec3f* row = srcCopy.ptr<cv::Vec3f>(i);
        for(int j = 2; j < srcCopy.cols-2; j++){
            // modify the row in srcCopy as we need to store these results for the vertical pass
            row[j] = row[j-2] * kernelVector[0] + 
                row[j-1] * kernelVector[1] + 
                row[j] * kernelVector[2] + 
                row[j+1] * kernelVector[3] + 
                row[j+2] * kernelVector[4];
        }
    }

    // vertical filter pass
    for (int j = 2; j < srcCopy.cols - 2; j++){
        for (int i = 2; i < srcCopy.rows - 2; i++){
            cv::Vec3f sum = srcCopy.at<cv::Vec3f>(i-2, j) * kernelVector[0] + 
                                            srcCopy.at<cv::Vec3f>(i-1, j) * kernelVector[1] + 
                                            srcCopy.at<cv::Vec3f>(i, j) * kernelVector[2] + 
                                            srcCopy.at<cv::Vec3f>(i+1, j) * kernelVector[3] + 
                                            srcCopy.at<cv::Vec3f>(i+2, j) * kernelVector[4];


            // modify dst directly
            dst.at<cv::Vec3b>(i, j) = sum;

        }
    }


    return 0;

}