
#include <opencv2/opencv.hpp>
#include <cmath>


template <typename FeatureVectorType>
class DistanceMetric {
public: 
    // Compute the distance(double) between two feature vectors (represented as cv::Mat)
    // a smaller distance means the two feature vectors are more similar
    virtual double distance(const FeatureVectorType& a, const FeatureVectorType& b) const = 0;

};



class SSDDistance : public DistanceMetric<cv::Mat>
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


class HistogramIntersection : public DistanceMetric<cv::Mat>
{
public:

    /**
     * Compute the histogram intersection between two feature vectors (ND histograms).
     * @param a the first feature vector (ND histogram). This should be normalized
     * @param b the second feature vector (ND histogram). This should be normalized
     * 
     */
    double distance(const cv::Mat& a, const cv::Mat& b) const override
    {
        // Check if the two feature vectors have the same size
        if (a.size() != b.size())
        {
            throw std::invalid_argument("Feature vectors have different sizes");
        }

        // Compute histogram intersection
        double intersection = 0.0;

        // Iterate over all elements in the "flattened" N-dimensional space
        cv::MatConstIterator_<float> itA = a.begin<float>(), itB = b.begin<float>();
        cv::MatConstIterator_<float> endA = a.end<float>();
        for (; itA != endA; ++itA, ++itB) {
            intersection += std::min(*itA, *itB);
        }

        return 1 - intersection;  // Convert similarity to distance metric
    }
};


class MultiHistogramIntersection : public DistanceMetric<std::vector<cv::Mat>>
{

public:

    /**
     * Compute the summed histogram intersection between two feature vectors (3D histograms).
     * In this summing, all vectors are weighted equally.
     * @param a the first feature vector (3D histogram). This should be normalized
     * @param b the second feature vector (3D histogram). This should be normalized
     * 
     */
    double distance(const std::vector<cv::Mat>& a, const std::vector<cv::Mat>& b) const override
    {      
        double distance = 0;
        HistogramIntersection histIntersection = HistogramIntersection();
        for(int i = 0; i < a.size(); i++)
        {
            if (a[i].size() != b[i].size())
            {
                throw std::invalid_argument("Feature vectors have different sizes");
            }
            distance += histIntersection.distance(a[i], b[i]);

        }

        return distance;
    }
};