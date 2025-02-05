/**
 * Adithya Palle
 * February 4, 2025
 * 
 * Header exposing various feature extraction methods for images, as well as a map of strings to feature extractors.
 */
#include <opencv2/opencv.hpp>
#include "DA2Network.hpp"
#include <fstream>  
#pragma once

#define DEFAULT_RESNET_EMBEDDINGS_FILE_PATH "/Users/adithyapalle/work/CS5330/content_based_image_retrieval/ResNet18_olym.csv"





int applySeparableFilter(const cv::Mat& src, cv::Mat& dst, std::vector<float>& verticalFilter, std::vector<float>& horizontalFilter);

int sobelX3x3( const cv::Mat &src,  cv::Mat &dst );

int sobelY3x3( const cv::Mat &src,  cv::Mat &dst );


int magnitude( cv::Mat &sx, cv::Mat &sy, cv::Mat &dst );

cv::Mat readEmbeddingsFromFile(std::string resnetEmbeddingsFilePath, std::string imagePath);



class FeatureExtractor {

public:
    // Extract features from an image (cv::Mat) into a feature vector (cv::Mat)
    virtual std::vector<cv::Mat> extractFeatures(const cv::Mat& image) = 0;

    // Extract features from the path to an image file (std::string) into a feature vector (cv::Mat)
    virtual std::vector<cv::Mat> extractFeaturesFromFile(const std::string& imagePath){
        cv::Mat image = cv::imread(imagePath);
        return extractFeatures(image);
    }

};


/**
 * Feature Extractor that produces a single feature vector from an image
 */
class SingleFeatureExtractor : public FeatureExtractor {
public:
    // Extract features from an image (cv::Mat) into a feature vector (cv::Mat)
    virtual cv::Mat _extractFeatures(const cv::Mat& image) = 0;
    // Extract features from an image (cv::Mat) into a feature vector (cv::Mat)
    std::vector<cv::Mat> extractFeatures(const cv::Mat& image){
        std::vector<cv::Mat> features;
        features.push_back(_extractFeatures(image));
        return features;
    }
};



class CenterSquareFeatureExtractor : public SingleFeatureExtractor{
    private:
        // The size of the center square
        int size;

    public:
        CenterSquareFeatureExtractor(int size) : size(size) {}


        /**
         * Extract the center square of the image with dimensions size x size
         */
        cv::Mat _extractFeatures(const cv::Mat& image) override
        {
            // Check if the image is empty
            if (image.empty())
            {
                throw std::invalid_argument("Image is empty");
            }

            // Get the center square of the image
            cv::Rect roi(image.cols / 2 - size / 2, image.rows / 2 - size / 2, size, size);
            cv::Mat centerSquare = image(roi);

            return centerSquare;
        }

};


class Histogram3D : public SingleFeatureExtractor{

    private:
        int numBins;
        bool countMissingPixels = true; // whether or not to count (255,0,255) pixels
    public:
        Histogram3D(int numBins) : numBins(numBins) {}
        Histogram3D(int numBins, bool countMissingPixels) : numBins(numBins), countMissingPixels(countMissingPixels) {}
        /**
         * Extract the 3D histogram of the image, using numBins bins for each channel (RGB).
         * The histogram is normalized so that the sum of all bins is 1.
         * @param image the input image (CV_8UC3)
         * @return the 3D histogram (CV_8U) with dimension (numBins, numBins, numBins)
         */
        cv::Mat _extractFeatures(const cv::Mat& image) override
        {

            // Check if the image is empty
            if (image.empty())
            {
                throw std::invalid_argument("Image is empty");
            }
            int histSize[] = {numBins, numBins, numBins};

            
            cv::Mat hist(3, histSize, CV_32F, cv::Scalar(0)); // Proper 3D histogram

            unsigned char binSize = 256 / numBins;
            int nPixels = 0;
            // Loop over all pixels
            for (int i = 0; i < image.rows; i++)
            {
                const cv::Vec3b* ptr = image.ptr<cv::Vec3b>(i); // Pointer to row i
                for (int j = 0; j < image.cols; j++)
                {

                    // Get the RGB values
                    unsigned char b = ptr[j][0];
                    unsigned char g = ptr[j][1];
                    unsigned char r = ptr[j][2];

                    if (!countMissingPixels && b == 255 && g == 0 && r == 255){ // because (255,0,255) is the color we use to mask out the background
                        continue;
                    }

                    unsigned char bIndex = b/binSize;
                    unsigned char gIndex = g/binSize;
                    unsigned char rIndex = r/binSize;


                    // Increment the histogram
                    int idx[] = {bIndex, gIndex, rIndex};
                    hist.at<float>(idx)++;

                    nPixels++;

                }
            }
           
            hist = hist / (nPixels); // Normalize the histogram by the number of pixels that were considered

            return hist;
        }
};


class Histogram2D : public SingleFeatureExtractor{

    private:
        int numBins = 8;
    public:
        Histogram2D(int numBins) : numBins(numBins) {}

        /**
         * Generate a 2D chromaticity histogram of the image (R and G channels).
         * @param image the input image (CV_8UC3)
         * @ return the 2D histogram (CV_8U) with dimension (numBins, numBins). Only the top left triangle is filled as the histogram values are normalized such that r+g+b=1
         */
        cv::Mat _extractFeatures(const cv::Mat& image) override{


            // The below code is a modified snippet from Bruce Maxwell's makeHist.cpp demo


            cv::Mat hist = cv::Mat::zeros( cv::Size( numBins, numBins ), CV_32FC1 );

            // loop over all pixels
            for( int i=0;i<image.rows;i++) {
                const cv::Vec3b *ptr = image.ptr<cv::Vec3b>(i); // pointer to row i
                for(int j=0;j<image.cols;j++) {

                // get the RGB values
                float B = ptr[j][0];
                float G = ptr[j][1];
                float R = ptr[j][2];

                // compute the r,g chromaticity
                float divisor = R + G + B;
                divisor = divisor > 0.0 ? divisor : 1.0; // check for all zeros
                float r = R / divisor;
                float g = G / divisor;

                // compute indexes, r, g are in [0, 1]
                int rindex = (int)( r * (numBins - 1) + 0.5 );
                int gindex = (int)( g * (numBins - 1) + 0.5 );

                // increment the histogram
                hist.at<float>(rindex, gindex)++;

                // keep track of the size of the largest bucket (just so we know what it is)
                float newvalue = hist.at<float>(rindex, gindex);
                }
            }

              hist /= (image.rows * image.cols); // normalize the histogram by the number of pixels

            return hist;
        }
};



class TextureExtractor : public SingleFeatureExtractor{
    public:

        /** 
         * Generates a sobel gradient magnitude iamage from the input image.
         * @param src - the input image (CV_8UC3)
         * @param numBins - the number of bins to use for the histogram
         * @return the gradient magnitude image (CV_8UC3)
         */
        cv::Mat _extractFeatures(const cv::Mat& src) override{
            cv::Mat hist = cv::Mat::zeros( cv::Size( 256, 1 ), CV_8U );

            cv::Mat sx, sy;
            if (sobelX3x3(src, sx) != 0) {
                std::cerr << "Error applying sobelX3x3" << std::endl;
                exit(-1);
            }
            if (sobelY3x3(src, sy) != 0) {
                std::cerr << "Error applying sobelY3x3" << std::endl;
                exit(-1);
            }
            cv::Mat mag;
            if (magnitude(sx, sy, mag) != 0) {
                std::cerr << "Error getting gradient magnitude" << std::endl;
                exit(-1);
            }
            
            return mag;
        }   
};


/**
 * Feature Extractor that produces multiple feature vectors from an image
 */
class MultiFeatureExtractor : public FeatureExtractor {
private: 
    std::vector <SingleFeatureExtractor*> featureExtractors;
    /**
     * Computes multiple feature vectors from an image
     * @param image the input image (CV_8UC3)
     * @return  a vector of 3D histograms (CV_8U) with dimension (numBins, numBins, numBins)
     */
    std::vector<cv::Mat> _extractFeatures(const cv::Mat& image) 
    {

        std::vector<cv::Mat> histograms;
        for (SingleFeatureExtractor* featureExtractor : featureExtractors){
            histograms.push_back(featureExtractor->_extractFeatures(image));
        }

        return histograms;

    }

public:
    MultiFeatureExtractor(std::vector <SingleFeatureExtractor*> featureExtractors) : featureExtractors(featureExtractors) {}
    // Extract features from an image (cv::Mat) into a set of feature vector (std::vector<cv::Mat>)
    std::vector<cv::Mat> extractFeatures(const cv::Mat& image){
        return _extractFeatures(image);
    }
};

/**
 * Feature Extractor that produces applies feature extraction methods on top of each other.
 */
class CompositeFeatureExtractor: public SingleFeatureExtractor{
    private:
        // The feature extractors to use to create intermediate feature vector
        // applied in order
        std::vector<SingleFeatureExtractor*> featureExtractors;
    public:

        CompositeFeatureExtractor(std::vector<SingleFeatureExtractor*> featureExtractors) : featureExtractors(featureExtractors) {}

        /**
         * Extracts the features from the image using the intermediate feature extractors and then the final feature extractor
         * @param image the input image (CV_8UC3)
         * @return the final feature vector
         */
        cv::Mat _extractFeatures(const cv::Mat& image){
            cv::Mat intermediateFeature = image;
            for (int i = 0; i < featureExtractors.size(); i++){
                SingleFeatureExtractor* featureExtractor = featureExtractors[i];
                intermediateFeature = featureExtractor->_extractFeatures(intermediateFeature);

            }

            return intermediateFeature;
            }
        
};

int getDepthValues(const cv::Mat&src, cv::Mat &dst);

class ForegroundExtractor : public SingleFeatureExtractor{
    private:
        // threshold below which a pixel is considered foreground
        int threshold = 128;
    public:
        ForegroundExtractor(int threshold = 128) : threshold(threshold) {}
        /**
         * Extracts the foreground by settubg all background pixels to black (0,0,0)
         * Uses depth anything to find depth rating of each pixle (0-255 with 0 reperesenting furthest and 255 representing closest),
         * and then blacks out pixles that are above a given threshold
         * @param image - the input image
         * @returns 3 channel rgb image with background blackde out.
         * 
         */
        cv::Mat _extractFeatures(const cv::Mat& image){
            cv::Mat depthImage;
            if(getDepthValues(image, depthImage) != 0){
                std::cout << "Error getting depth values" << std::endl;
                exit(1);
            }
            cv::Mat dst = image.clone(); // copy all values
            for (int i = 0; i< image.rows ; i++){
                const cv::Vec3b* row = image.ptr<cv::Vec3b>(i);
                unsigned char* depthRow = depthImage.ptr<unsigned char>(i);
                cv::Vec3b* dstRow = dst.ptr<cv::Vec3b>(i);
                for (int j = 0; j < image.cols; j++){
                    if (depthRow[j] < threshold){
                        dstRow[j] = cv::Vec3b(255,0,255);
                    }
                }
            }



            return dst;

        }

};

class ResnetFeatureExtractor : public FeatureExtractor{
    private:
        std::string resnetEmbeddingsFilePath;
        std::map<std::string, cv::Mat> resnetEmbeddingsMap;

    public:

        ResnetFeatureExtractor(std::string resnetEmbeddingsFilePath = DEFAULT_RESNET_EMBEDDINGS_FILE_PATH)
        {
            this->resnetEmbeddingsFilePath = resnetEmbeddingsFilePath;

            // preload the resnet embeddings into a map
            std::ifstream file(resnetEmbeddingsFilePath);
            if (!file.is_open()) {
                std::cerr << "Error opening file: " + resnetEmbeddingsFilePath << std::endl;
                exit(-1);
            }
            std::string line;
            while (std::getline(file, line)) {
                std::stringstream ss(line);
                std::string value;
                std::vector<std::string> row;

                while (std::getline(ss, value, ',')) {  // Split by comma
                    row.push_back(value);
                }

                // convert the string to a vector of floats
                std::vector<float> embeddings;
                for (int i = 1; i < row.size(); i++){
                    embeddings.push_back(std::stof(row[i]));
                }
                // the true flag is used to copy the embeddings to avoid dangling pointer issue where embedding gets corrupted
                cv::Mat resnetEmbeddings = cv::Mat(embeddings, true);
                if (resnetEmbeddings.empty()){
                    std::cerr << "Error creating cv::Mat from embeddings, embeddings empty" << std::endl;
                    exit(-1);
                }

                if(norm(resnetEmbeddings) == 0){
                    std::cerr << "Embeddings have zero norm" << std::endl;
                    exit(-1);
                }

                // no need for copy as assignment increases the refernce count of cv::Mat so the memory is not deallocated
                this->resnetEmbeddingsMap[row[0]] = resnetEmbeddings;
            
            }
            file.close();
        }



        /**
         * Unimplemented method
         */
        std::vector<cv::Mat> extractFeatures(const cv::Mat& image) override
        {
            std::cerr << "Not implemented" << std::endl;
            exit(-1);
        }

        /**
         * Extracts the resnet embeddings from the image file path
         * @param imagePath the path to the image file
         * @return the resnet embeddings (cv::Mat) (1,512) for the image
         */
        std::vector<cv::Mat> extractFeaturesFromFile(const std::string& imagePath) override{
        
            // Read the embeddings from the file
            std::vector<cv::Mat> resnetEmbeddings;

            //get the image name
            std::size_t found = imagePath.find_last_of("/\\");
            std::string imageName;
            if (found != std::string::npos){
                imageName = imagePath.substr(found+1);
            }else{
                imageName = imagePath; // if no path is found, just use the image name as this means there is no parent directory in the path
            }

 

            // check if the image is in the map
            if (resnetEmbeddingsMap.find(imageName) == resnetEmbeddingsMap.end()){
                std::cerr << "Embeddings not found for image: " + imageName + " in file: " + resnetEmbeddingsFilePath << std::endl;
                exit(-1);
            }
            
            //retrieve the embeddings from the map
            resnetEmbeddings.push_back(resnetEmbeddingsMap[imageName]);
            return resnetEmbeddings;
        }   

};



class FFTExtractor : public SingleFeatureExtractor{


    private:
        // whether or not to isolate the high frequency components (zeroing out the low frequency components)
        bool isolateHighFrequency = true;
    public:
        FFTExtractor(bool isolateHighFrequency = true) : isolateHighFrequency(isolateHighFrequency) {}

    /**
     * Extracts the magnitude of the fourier transform of the image
     * 
     * Code snipttes here are taken from Bruce Maxwell's fourierTransform.cpp demo
     * 
     * @param image the input image (CV_8UC3)
     * @return the magnitude of the fourier transform (CV_8U), centered at the origin
     */
    cv::Mat _extractFeatures(const cv::Mat& image){
        cv::Mat grey;
        cv::Mat paddedGrey;
        cv::cvtColor( image, grey, cv::COLOR_BGR2GRAY );
        // The fast DFT wants images to be nicely sized
        int m = cv::getOptimalDFTSize( grey.rows );
        int n = cv::getOptimalDFTSize( grey.cols );

        cv::copyMakeBorder( grey, paddedGrey, 0, m - grey.rows, n - grey.cols, cv::BORDER_CONSTANT, 0 );

        cv::Mat planes[] = { cv::Mat_<float>(paddedGrey), cv::Mat::zeros(paddedGrey.size(), CV_32F ) };
        cv::Mat complex;
        cv::Mat fft;
        cv::merge( planes, 2, complex );
        // take the discrete Fourier transform of the image
        cv::dft( complex, fft ); 

        // in order to visualize the spectrum, we compute the magnitude of the complex number and take the log
        cv::Mat mag;
        mag.create(fft.size(), CV_32F );
        // compute the magnitude and the log
        for(int i=0;i<fft.rows;i++) {
            float *data = fft.ptr<float>(i);
            float *mptr = mag.ptr<float>(i);
            for(int j=0;j<fft.cols;j++) {
            float x = data[j*2];
            float y = data[j*2 + 1];
            mptr[j] = log( 1 + sqrt(x*x + y*y) ); // get the log of the magnitude
            }
        }

        cv::normalize( mag, mag, 0, 1, cv::NORM_MINMAX );

        // reorganize the quadrants to be centered on the middle of the image
        int cx = mag.cols/2;
        int cy = mag.rows/2;


        cv::Mat q0(mag, cv::Rect( 0, 0, cx, cy ) ); // x, y, width, height
        cv::Mat q1(mag, cv::Rect( cx, 0, cx, cy ) );
        cv::Mat q2(mag, cv::Rect( 0, cy, cx, cy ) );
        cv::Mat q3(mag, cv::Rect( cx, cy, cx, cy ) );
        // flips q0 and q3

        cv::Mat tmp;
        q0.copyTo(tmp);
        q3.copyTo(q0);
        tmp.copyTo(q3);

        // flips q1 and q2
        q1.copyTo(tmp);
        q2.copyTo(q1);
        tmp.copyTo(q2);

        // mag now contains the magnitude of the Fourier transform, centered at the origin


        // zero out the center of the image if we are isolating the high frequency components
        if (isolateHighFrequency){
        
            int tlx = mag.cols / 4;
            int tly = mag.rows / 4;
            int width = mag.cols / 2;
            int height = mag.rows / 2;
            // iterate over the center of the image(  rectangle with top left corner at (tlx, tly) and width and height as half the respective dimensions) and set the values to 0

            for (int i = tly; i < tly + height; i++){
                float* row = mag.ptr<float>(i);
                for (int j = tlx; j < tlx + width; j++){
                    row[j] = 0;
                }
            }

        }

        // resize the image to 256x256
        cv::resize(mag, mag, cv::Size(256, 256));
        


        return mag;
    }
};

// Map of feature extractors to their names
extern std::map<std::string, FeatureExtractor*> featureExtractorMap;