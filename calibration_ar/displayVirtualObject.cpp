/**
 * Adithya Palle
 * February 28, 2025
 * 
 * Main file for program that displays a virtual object above the chessboard pattern in the video feed.
 */
#include "calibration.h"
#define EXPECTED_FRAME_WIDTH 1280
#define EXPECTED_FRAME_HEIGHT 720

/**
 * Represents an object as a list of its points
 * and a list of pairs of indices to those points 
 * which represents lines in the object's structure.
 */
struct VirtualObject{
    std::vector<cv::Vec3f> points;
    std::vector<std::pair<int, int>> lines;
};
/**
 * Creates a virtual sword object that sits 2 z-units above the origin
 */
VirtualObject createSwordObject(){

    VirtualObject sword;


    // Define key 3D points of the sword
    sword.points = {
        // Handle (3D rectangular prism along Z-axis)
        {-0.3,  0,  1.5},  // 0: Bottom-left front
        { 0.3,  0,  1.5},  // 1: Bottom-right front
        {-0.3,  0,  3},    // 2: Top-left back
        { 0.3,  0,  3},    // 3: Top-right back
        {-0.3, -0.5,  1.5},// 4: Bottom-left front (depth)
        { 0.3, -0.5,  1.5},// 5: Bottom-right front (depth)
        {-0.3, -0.5,  3},  // 6: Top-left back (depth)
        { 0.3, -0.5,  3},  // 7: Top-right back (depth)

        // Crossguard (3D, extends along X & Z)
        {-1.5,  0,  2.8},  // 8: Left crossguard front
        { 1.5,  0,  2.8},  // 9: Right crossguard front
        {-1.5,  0,  3.2},  // 10: Left crossguard back
        { 1.5,  0,  3.2},  // 11: Right crossguard back

        // Blade (3D, extends along Z)
        {-0.5,  0,  3},   // 12: Blade base left front
        { 0.5,  0,  3},   // 13: Blade base right front
        {-0.5,  0,  7},   // 14: Blade tip left front
        { 0.5,  0,  7},   // 15: Blade tip right front
        { 0,    0,  8},   // 16: Blade tip center

        {-0.5, -0.3,  3}, // 17: Blade base left back
        { 0.5, -0.3,  3}, // 18: Blade base right back
        {-0.5, -0.3,  7}, // 19: Blade tip left back
        { 0.5, -0.3,  7}, // 20: Blade tip right back
        { 0,   -0.3,  8}  // 21: Blade tip center back
    };

    // Define lines connecting points to form the sword's 3D structure
    sword.lines = {
        // Handle (rectangular prism along Z)
        {0, 1}, {0, 2}, {1, 3}, {2, 3}, {4, 5}, {4, 6}, {5, 7}, {6, 7},
        {0, 4}, {1, 5}, {2, 6}, {3, 7},

        // Crossguard
        {8, 9}, {10, 11}, {8, 10}, {9, 11}, {8, 2}, {9, 3}, {10, 6}, {11, 7},

        // Blade (rectangular prism extending in +Z)
        {12, 13}, {12, 14}, {13, 15}, {14, 16}, {15, 16}, // Front face
        {17, 18}, {17, 19}, {18, 20}, {19, 21}, {20, 21}, // Back face
        {12, 17}, {13, 18}, {14, 19}, {15, 20}, {16, 21}  // Connect front & back
    };

    return sword;
    
}
/**
 * Displays a virtual object in the video feed.
 * @param object The object represented by a VirtualObject struct.
 * @param img The image to display the coordinate axes on.
 * @param cameraMatrix The camera matrix.
 * @param distCoeffsMat The distortion coefficients matrix.
 * @param rvec The rotation vector.
 * @param tvec The translation vector.
 */
void addVirtualObjectToImage(const VirtualObject& object, cv::Mat& img,  const cv::Mat& cameraMatrix, const cv::Mat& distCoeffsMat, const cv::Mat&  rvec, const cv::Mat&  tvec){
    std::vector<cv::Point2f> imagePoints;
    cv::projectPoints(object.points, rvec, tvec, cameraMatrix, distCoeffsMat, imagePoints);

    if (imagePoints.size() != object.points.size()){
        std::cerr << "Error: Number of image points does not match number of object points" << std::endl;
        return;
    }

    // draw lines between the points to form the object
    for (int i = 0; i < object.lines.size(); i++){
        std::pair<int, int> line = object.lines[i];
        cv::line(img, imagePoints[line.first], imagePoints[line.second], cv::Scalar(220, 70, 70), 2); 
    }
    return;

}




/**
 * Displays a virtual object above the chessboard pattern in the video feed.
 * @param argc The number of command line arguments.
 * @param argv The command line arguments. The first is the path to the calibration yaml file.
 */
int main(int argc, char** argv){

    if (argc != 2){
        std::cerr << "Usage: ./get_camera_pos <path_to_calibration_yaml>" << std::endl;
        return -1;
    }

    std::string calibrationYamlPath = argv[1];

    cv::VideoCapture cap(0);

    if (!cap.isOpened())
    {
        std::cerr << "Error opening video stream" << std::endl;
        return -1;
    }



    cv::Mat frame;

    int frameCols = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frameRows = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    if (frameCols != EXPECTED_FRAME_WIDTH || frameRows != EXPECTED_FRAME_HEIGHT){
        std::cerr << "Error: Camera resolution is not " << EXPECTED_FRAME_WIDTH << "x" << EXPECTED_FRAME_HEIGHT << std::endl;
        return -1;
    }


    // load camera matrix and distortion coefficients from yaml
    cv::Mat cameraMatrix;
    std::vector<double> distCoeffs;
    cv::FileStorage fs(calibrationYamlPath, cv::FileStorage::READ);

    if (!fs.isOpened()){
        std::cerr << "Error opening calibration yaml file" << std::endl;
        return -1;
    }

    fs["cameraMatrix"] >> cameraMatrix;

    if (cameraMatrix.empty()){
        std::cerr << "Error reading camera matrix from yaml" << std::endl;
        return -1;
    }

    fs["distCoeffs"] >> distCoeffs;

    if (distCoeffs.empty()){
        std::cerr << "Error reading distortion coefficients from yaml" << std::endl;
        return -1;
    }

    
    fs.release();

    cv::Mat distCoeffsMat(1, distCoeffs.size(), CV_64F, distCoeffs.data());

    std:: cout << "Camera Matrix: " << cameraMatrix << std::endl;
    std:: cout << "Distortion Coefficients: " << distCoeffsMat << std::endl;



    VirtualObject object = createSwordObject(); // creates a virtual sword object



    while (true){
        cap >> frame;



        if (frame.empty())
        {
            std::cerr << "Error reading frame" << std::endl;
            return 1;
        }

        std::vector<cv::Point2f> corners;
        bool patternFound = extractCorners(frame, corners, PATTERN_SIZE);

        // use solvePNP to get the camera's translation and rotation relative to the chessboard

        if (patternFound)
        {
            std::vector<cv::Vec3f> worldPoints = calculateWorldPoints(PATTERN_SIZE);
            cv::Mat rvec, tvec;
            cv::solvePnP(worldPoints, corners, cameraMatrix, distCoeffsMat, rvec, tvec);

            addVirtualObjectToImage(object, frame, cameraMatrix, distCoeffsMat, rvec, tvec);
            
        }

        cv::imshow("Frame", frame);
        cv::waitKey(1);

    }
}