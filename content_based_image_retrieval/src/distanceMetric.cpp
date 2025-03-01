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

// Map of distance metrics
std::map<std::string, DistanceMetric*> distanceMetricMap = {
    {"SSD_uchar", new SSDDistance<uchar>()}, // for task 1
    {"SSD_float", new SSDDistance<float>()}, // for extension
    {"HistogramIntersection", new HistogramIntersection()}, // for task 2
    {"MultiHistogramIntersection", new MultiHistogramIntersection()}, // for tasks 3 and 4
    {"CosineDistance", new CosineDistance()}, // for task 5 and 6
    {"IOU", new IOU()}, // for task 7
};