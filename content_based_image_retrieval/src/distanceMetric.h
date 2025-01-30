
#include <opencv2/opencv.hpp>
#include <cmath>


class DistanceMetric {
public: 
    // Compute the distance(double) between two feature vectors (represented as cv::Mat)
    // a smaller distance means the two feature vectors are more similar
    virtual double distance(const std::vector<cv::Mat>& a, const std::vector<cv::Mat>& b) const = 0;

};


class MultipleDistanceMetric : public DistanceMetric {
private:

    // computes the distance between multiple feature vectors
    virtual double _distance(const std::vector<cv::Mat>&, const std::vector<cv::Mat>&) const = 0;
public:
    // Compute the distance(double) between two sets feature vectors (represented as std::vector<cv::Mat>)
    // a smaller distance means the two feature vectors are more similar
    double distance(const std::vector<cv::Mat>& a, const std::vector<cv::Mat>& b) const{
        if (a.size() != b.size())
        {
            throw std::invalid_argument("Feature vectors have different sizes");
        }
        for (int i = 0; i < a.size(); i++)
        {
            if (a[i].size() != b[i].size())
            {
                throw std::invalid_argument("Individual Feature vectors have different sizes");
            }
        }
        if (a.size() <= 1)
        {
            throw std::invalid_argument("Must be single mat");
        }    
        
        return _distance(a, b);
        
        }


};

class SingleDistanceMetric : public DistanceMetric {
private:

    // computes the distance between two feature vectors
    virtual double _distance(const cv::Mat&, const cv::Mat&) const = 0;
public:
    // Compute the distance(double) between two feature vectors (represented as cv::Mat)
    // a smaller distance means the two feature vectors are more similar
    double distance(const std::vector<cv::Mat>& a, const std::vector<cv::Mat>& b) const{
        if (a.size() != b.size())
        {
            throw std::invalid_argument("Feature vectors have different sizes");
        }
        if (a[0].size() != b[0].size())
        {
            throw std::invalid_argument("Individual Feature vectors have different sizes");
        }
        if (a.size() > 1)
        {
            throw std::invalid_argument("Must be single mat");
        }    
        
        return _distance(a[0], b[0]);
        
        }


};



class SSDDistance : public SingleDistanceMetric
{
public:

    /**
     * Compute the sum of squared differences between two feature vectors (3-channel images).
     * 
     */
    double _distance(const cv::Mat& a, const cv::Mat& b) const override
    {

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


class HistogramIntersection : public SingleDistanceMetric
{
public:

    /**
     * Compute the histogram intersection between two feature vectors (ND histograms).
     * @param a the first feature vector (ND histogram). This should be normalized
     * @param b the second feature vector (ND histogram). This should be normalized
     * 
     */
    double _distance(const cv::Mat& a, const cv::Mat& b) const override
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


class MultiHistogramIntersection : public MultipleDistanceMetric
{

public:

    /**
     * Compute the summed histogram intersection between multiple histogram feature vectors.
     * In this summing, all vectors are weighted equally.
     * @param a the first feature vector (3D histogram). This should be normalized
     * @param b the second feature vector (3D histogram). This should be normalized
     * 
     */
    double _distance(const std::vector<cv::Mat>& a, const std::vector<cv::Mat>& b) const override
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