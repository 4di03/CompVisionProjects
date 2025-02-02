/**
 * 
 */
#include <opencv2/opencv.hpp>
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

// ideally these two would be implemented with composition, but for times sake I will use inheritance



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

    public:
        Histogram3D(int numBins) : numBins(numBins) {}

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

                    unsigned char bIndex = b/binSize;
                    unsigned char gIndex = g/binSize;
                    unsigned char rIndex = r/binSize;


                    // Increment the histogram
                    float prevVal = hist.at<float>(bIndex,gIndex,rIndex);
                    int idx[] = {bIndex, gIndex, rIndex};
                    hist.at<float>(idx)++;
                    float postVal = hist.at<float>(bIndex,gIndex,rIndex);



                }
            }
           
            hist = hist / (image.rows * image.cols);

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
    private:
        int numBins = 8;
    public:
        TextureExtractor(int numBins) : numBins(numBins) {}

        /** 
         * Generates a sobel gradient magnitude histogram (3D) from the input image.
         * @param src - the input image (CV_8UC3)
         * @param numBins - the number of bins to use for the histogram
         * @return the gradient magnitude histogram (CV_8U) with dimensions (numBins, numBins, numBins)
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
            
            Histogram3D hist3D(numBins);

            return hist3D._extractFeatures(mag);
        }   
};



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
        // make both histograms with a 8 bins
        Histogram2D hist2D(8);
        Histogram3D hist3D(8);

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

// Map of feature extractors to their names
extern std::map<std::string, FeatureExtractor*> featureExtractorMap;