
#include <opencv2/opencv.hpp>


template <typename FeatureVectorType>
class FeatureExtractor {
public:
    // Extract features from an image (cv::Mat) into a feature vector (cv::Mat)
    virtual FeatureVectorType extractFeatures(const cv::Mat& image) = 0;
};


class CenterSquareFeatureExtractor : public FeatureExtractor<cv::Mat>{
    private:
        // The size of the center square
        int size;

    public:
        CenterSquareFeatureExtractor(int size) : size(size) {}


        /**
         * Extract the center square of the image with dimensions size x size
         */
        cv::Mat extractFeatures(const cv::Mat& image) override
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


class Histogram3D : public FeatureExtractor<cv::Mat>{

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
        cv::Mat extractFeatures(const cv::Mat& image) override
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


class Histogram2D : public FeatureExtractor<cv::Mat>{

    private:
        int numBins = 256;
    public:
        Histogram2D(int numBins) : numBins(numBins) {}

        /**
         * Generate a 2D chromaticity histogram of the image (R and G channels).
         * @param image the input image (CV_8UC3)
         * @ return the 2D histogram (CV_8U) with dimension (numBins, numBins). Only the top left triangle is filled as the histogram values are normalized such that r+g+b=1
         */
        cv::Mat extractFeatures(const cv::Mat& image) override{


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


class MultiHistogram: public FeatureExtractor<std::vector<cv::Mat>>{

    public:

        /**
         * Computes a 3D RGB histogram and a 2D chromaticity histogram of the image.
         * Both are normalized
         * @param image the input image (CV_8UC3)
         * @return  a vector of 3D histograms (CV_8U) with dimension (numBins, numBins, numBins)
         */
        std::vector<cv::Mat >extractFeatures(const cv::Mat& image) override
        {

            Histogram2D hist2D(8);
            Histogram3D hist3D(256);

            std::vector<cv::Mat> histograms;
            histograms.push_back(hist2D.extractFeatures(image));
            histograms.push_back(hist3D.extractFeatures(image));

            return histograms;

        }
};