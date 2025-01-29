
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