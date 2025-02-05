/*
  Adithya Palle
  February 4, 2025

  Header exposing various distance metrics for comparing feature vectors, as well as a map of strings to distance metrics.
  */

#include <opencv2/opencv.hpp>
#include <cmath>
#pragma once

/**
 * Interface for computing the distance between two feature vectors.
 */
class DistanceMetric {
public: 
    // Compute the distance(double) between two feature vectors (represented as cv::Mat)
    // a smaller distance means the two feature vectors are more similar
    // we take in a vector of cv::Mat as the feature vectors can be one or more
    virtual double distance(const std::vector<cv::Mat>& a, const std::vector<cv::Mat>& b) const = 0;

};

/**
 * Interface for computing the distance between multiple feature vectors.
 */
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

        return _distance(a, b);
        
        }


};

/**
 * Interface for computing the distance between two feature vectors.
 */
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


template <typename ImageDataType>// data type for numbers in the image
class SSDDistance : public SingleDistanceMetric
{
public:

    /**
     * Compute the sum of squared differences between two feature vectors (n-dimension images).
     * @param a the first feature vector (n-channel N-dimension matrix)
     * @param b the second feature vector (n-channel N-dimension matrix)
     */
    double _distance(const cv::Mat& a, const cv::Mat& b) const override
    {

        // Ensure both images have the same dimensions and number of channels
        if (a.size() != b.size() || a.type() != b.type()) {
            throw std::invalid_argument("Input matrices must have the same size and type");
        }

        double ssd = 0.0;
            
        // Iterate over all elements in the flattened multi-dimensional space
        cv::MatConstIterator_<ImageDataType> itA = a.begin<ImageDataType>(), itB = b.begin<ImageDataType>();
        cv::MatConstIterator_<ImageDataType> endA = a.end<ImageDataType>();
        
        // take difference of elements along each channel and sum them up
        for (; itA != endA; ++itA, ++itB) {
            double diff = static_cast<double>(*itA) - static_cast<double>(*itB);
            ssd += diff * diff;
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
     * @return the histogram intersection between the two feature vectors
     */
    double _distance(const cv::Mat& a, const cv::Mat& b) const override
    {
        // Check if the two feature vectors have the same size
        if (a.size() != b.size())
        {
            throw std::invalid_argument("Feature vectors have different sizes");
        }

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
     * @returns the summed histogram intersection between the two feature vectors
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

            //  get distance of individual histograms
            distance += histIntersection._distance(a[i], b[i]);

        }

        return distance;
    }
};

double safeAcos(double x);

class CosineDistance : public SingleDistanceMetric{

    private:

    /**
     * Computes 1 - cosine similarity between two feature vectors (1,512) (resnet embeddings).
     * Uses the formula: 
     * D = 1 - cos(theta)
     * 
     * a.b = |a||b|cos(theta)
     * theta = arccos(a . b / |a||b|)
     *
     * 
     * @param a the first feature vector (1,512)
     * @param b the second feature vector (1,512)
     * @return the cosine distance between the two feature vectors
     */    
     double _distance(const cv::Mat& a, const cv::Mat& b) const{

        if (norm(a) == 0 || norm(b) == 0)
        {
            std::cerr << "Feature vectors have zero norm" <<std::endl;
            exit(-1);
        }

        double tmp = a.dot(b) / (norm(a) * norm(b));

        double theta = safeAcos(tmp);

        if (std::isnan(theta))
        {
            std::cout << "Error computing cosine distance" << std::endl;
            exit(-1);
        }

        return 1 - cos(theta);
     }

};


class IOU : public SingleDistanceMetric{

    private:
        /**
         * Computes the intersection over union between two feature vectors (histograms).
         * @param a the first feature vector (histogram)
         * @param b the second feature vector (histogram)
         * @return 1 - the intersection over union between the two feature vectors, which is a distance metric as 0 implies intersection == union which means a perfect match.
         */
        double _distance(const cv::Mat& a, const cv::Mat& b) const override{
                    // Check if the two feature vectors have the same size
        if (a.size() != b.size())
        {
            throw std::invalid_argument("Feature vectors have different sizes");
        }

        double intersection = 0.0;
        double unionArea = 0.0;



        // Iterate over all elements in the "flattened" N-dimensional space
        cv::MatConstIterator_<float> itA = a.begin<float>(), itB = b.begin<float>();
        cv::MatConstIterator_<float> endA = a.end<float>();
        for (; itA != endA; ++itA, ++itB) {
            intersection += std::min(*itA, *itB);
            unionArea += std::max(*itA, *itB);
        }

        return 1 - (intersection/unionArea);  // Convert similarity to distance metric
        }


};


// Map of distance metrics
extern std::map<std::string, DistanceMetric*> distanceMetricMap ;
