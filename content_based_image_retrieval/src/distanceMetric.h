
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
                int blueDiff = aRow[j][0] - bRow[j][0];
                int greenDiff = aRow[j][1] - bRow[j][1];
                int redDiff = aRow[j][2] - bRow[j][2];
                ssd += blueDiff * blueDiff + greenDiff * greenDiff + redDiff * redDiff;
            }
        }

        return ssd;
    }
};


class HistogramIntersection : public DistanceMetric
{
public:

    /**
     * Compute the histogram intersection between two feature vectors (3D histograms).
     * @param a the first feature vector (3D histogram). This should be normalized
     * @param b the second feature vector (3D histogram). This should be normalized
     * 
     */
    double distance(const cv::Mat& a, const cv::Mat& b) const override
    {
        // Check if the two feature vectors have the same size
        if (a.size() != b.size())
        {
            throw std::invalid_argument("Feature vectors have different sizes");
        }

        // Compute the histogram intersection
        double intersection = 0;

        int size[3] = {a.size[0], a.size[1], a.size[2]};  // Correct way
        for (int i = 0; i < size[0]; i++)
        {   
            for (int j = 0; j < size[1]; j++)
            {
                for (int k = 0; k < size[2]; k++)
                {   
                    int idx[] = {i, j, k};
                    float minVal = std::min(a.at<float>(idx), b.at<float>(idx));
                    intersection += minVal;

                    
                }

            }
        }


        return 1 - intersection;
    }
};