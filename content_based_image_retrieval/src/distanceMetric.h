
#include <opencv2/opencv.hpp>
#include <cmath>
class DistanceMetric
{
public: 
    // Compute the distance(double) between two feature vectors (represented as cv::Mat)
    // a smaller distance means the two feature vectors are more similar
    virtual double distance(const cv::Mat& a, const cv::Mat& b) const = 0;

};



class SSDDistance : public DistanceMetric
{
public:

    /**
     * Compute the sum of squared differences between two feature vectors (3-channel images).
     * 
     */
    double distance(const cv::Mat& a, const cv::Mat& b) const override
    {
        // Check if the two feature vectors have the same size
        if (a.size() != b.size())
        {
            throw std::invalid_argument("Feature vectors have different sizes");
        }

        // Compute the sum of squared differences
        double ssd = 0;
        for (int i = 0; i < a.rows; i++)
        {   
            const cv::Vec3b* aRow = a.ptr<cv::Vec3b>(i);
            const cv::Vec3b* bRow = b.ptr<cv::Vec3b>(i);
            for (int j = 0; j < a.cols; j++)
            {
                ssd += pow(aRow[j][0] - bRow[j][0], 2) + pow(aRow[j][1] - bRow[j][1], 2) + pow(aRow[j][2] - bRow[j][2], 2);
            }
        }

        return ssd;
    }
};


