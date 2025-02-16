/**
 * Adithya Palle
 * Feb 15 2025
 * CS 5330 - Project 3 : Real-time 2D Object Recognition
 * 
 * Entrypoint for code that runs the simple version of cleanup (naive dilation and erosion) on the image, 
 * and then runs the optimized version of cleanup (grassfire algorithm) on the image.
 * Compares the two results based on both speed and accuracy. Writes the resulting masks to the test_output folder.
 * 
 * Note that while this program tests the entire cleanup algorithm (dilation and erosion), I tested the dilation and erosion functions thoroguhly separately, but that is not shown here for brevity.
 */


#include <iostream>
#include <opencv2/opencv.hpp>
#include "thresholding.h"
#include <random>
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#define MASK_SIZE 100
#define PROP_BG 0.3 // proportion of background pixels in the mask
#define NUM_TESTS 100


cv::Mat generateRandomMask(int rows = MASK_SIZE, int cols = MASK_SIZE){

    cv::Mat mask = cv::Mat::zeros(rows,cols, CV_8UC1);
    // Random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0, 1.0); // generates float between 0 and 1

    // Fill the mask randomly
    for (int i = 0; i < mask.rows; ++i) {
        for (int j = 0; j < mask.cols; ++j) {
            // we want to fill the mask with PROP_BG% background and (1-PROP_BG)% foreground so that the foreground is sparse, and we can see the effect of the dilation and erosion

            float prop = dist(gen);
            mask.at<unsigned char>(i, j) = prop < PROP_BG ? 0 : 255;

        }
    }
    return mask;
}
int main(){


    // generate an arbitrary mask (255 is background, 0 is foreground)

    cv::Mat mask = generateRandomMask(5000,5000);


    cv::Mat cleaned;

    

    // run the simple version of cleanup 5 times (time it)
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_TESTS; i++){
        cleaned = cleanupSimple(mask,1,5);
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed_seconds = end-start;
    std::cout << "Time taken for simple cleanup: " << elapsed_seconds.count() << "s" << std::endl;

    // run the optimized version of cleanup 5 times (time it)
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_TESTS; i++){
        cleaned = cleanupGrassfire(mask, 1, 5);
    }
    end = std::chrono::high_resolution_clock::now();

    elapsed_seconds = end-start;
    std::cout << "Time taken for grassfire cleanup: " << elapsed_seconds.count() << "s" << std::endl;

    mkdir("test_output", 0777);
    for (int a = 0; a < 10 ; a++){
        for (int b = 0; b<10; b++){
            mask = generateRandomMask();

            // mask = (cv::Mat_<uchar>(3,3) <<
            // 0, 0, 0,
            // 0, 255, 0,
            // 0, 0, 0); 

            cv::Mat simpleResult = cleanupSimple(mask , a, b);
            cv::Mat grassfireResult = cleanupGrassfire(mask, a, b);


            if(simpleResult.rows != grassfireResult.rows || simpleResult.cols != grassfireResult.cols){
                printf("Results differ for numErosionIterations = %d, numDilationIterations = %d\n", a, b);
                printf("Rows and columns differ\n");
                exit(1);
            }
            // compare the two results
            int numDifferentPixels = 0;
            for (int i = 0; i < simpleResult.rows; i++){
                const unsigned char* simpleRow = simpleResult.ptr<unsigned char>(i);
                const unsigned char* grassfireRow = grassfireResult.ptr<unsigned char>(i);
                for (int j = 0; j < simpleResult.cols; j++){
                    if (simpleRow[j] != grassfireRow[j]){
                        numDifferentPixels++;
                    }
                }
            }   
            if (numDifferentPixels >0){
                printf("Results differ for numErosionIterations = %d, numDilationIterations = %d\n", a, b);
                printf("Number of different pixels: %d\n", numDifferentPixels);

                // print the mask
                std::cout << "Mask" << std::endl;
                for (int i = 0; i < mask.rows; i++){
                    for (int j = 0; j < mask.cols; j++){
                        std::cout << static_cast<int>(mask.at<unsigned char>(i,j)) << " ";
                    }
                    std::cout << std::endl;
                }

                // print the simple result
                std::cout << "Simple result" << std::endl;
                for (int i = 0; i < simpleResult.rows; i++){
                    for (int j = 0; j < simpleResult.cols; j++){
                        std::cout << static_cast<int>(simpleResult.at<unsigned char>(i,j)) << " ";
                    }
                    std::cout << std::endl;
                }

                // print the grassfire result
                std::cout << "Grassfire result" << std::endl;
                for (int i = 0; i < grassfireResult.rows; i++){
                    for (int j = 0; j < grassfireResult.cols; j++){
                        std::cout << static_cast<int>(grassfireResult.at<unsigned char>(i,j)) << " ";
                    }
                    std::cout << std::endl;
                }




                exit(1);

            }else{
                // save the results (named by method and a,b) to test_output folder

                //make a directory for (a,b) if it doesn't exist
                std::string dirName = "test_output/" + std::to_string(a) + "_" + std::to_string(b);
                mkdir(dirName.c_str(), 0777);

                std::string maskName = dirName + "/mask.png";
                std::string simpleName = dirName + "/simple.png";
                std::string grassfireName = dirName + "/grassfire.png";

                cv::imwrite(maskName, mask);
                cv::imwrite(simpleName, simpleResult);
                cv::imwrite(grassfireName, grassfireResult);

            }
        }
    }

    std::cout << "The two results are the same" << std::endl;







    return 0;


}