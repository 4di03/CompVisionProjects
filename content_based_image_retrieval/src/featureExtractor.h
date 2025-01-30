
#include <opencv2/opencv.hpp>

class FeatureExtractor{
public:
    // Extract features from an image (cv::Mat) into a feature vector (cv::Mat)
    virtual cv::Mat extractFeatures(const cv::Mat& image) = 0;
};


class CenterSquareFeatureExtractor : public FeatureExtractor{
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


class Histogram3D : public FeatureExtractor{

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