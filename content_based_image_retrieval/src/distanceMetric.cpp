/**
 * Adithya Palle
 * February 4, 2025
 * 
 * Header that implements supporting functions for distance metrics, as well as a map of strings to distance metrics.
 */

#include "distanceMetric.h"
/**
 * Arccosine function that clamps the input to [-1, 1] to avoid precision errors.
 * I noticed that sometims the result of a float division is slightly greater than 1, which causes the acos function to return nan.
 * This function clamps the input to [-1, 1] to avoid this issue.
 * @param x the input value
 * @return the arccosine of the input value
 */
double safeAcos(double x) {
    if (x > 1.0) return 0;
    if (x < -1.0) return M_PI;
    return acos(x);
}

std::map<std::string, DistanceMetric*> distanceMetricMap = {
    {"SSD_uchar", new SSDDistance<uchar>()},
    {"SSD_float", new SSDDistance<float>()},
    {"HistogramIntersection", new HistogramIntersection()},
    {"MultiHistogramIntersection", new MultiHistogramIntersection()},
    {"CosineDistance", new CosineDistance()},
    {"IOU", new IOU()},
};