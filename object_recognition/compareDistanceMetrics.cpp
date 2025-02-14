/**
 * Adithya Palle
 * Feb 13 2025
 * CS 5330 - Project 3 : Real-time 2D Object Recognition
 * 
 * Entrypoint for program that allows you to classify objects in a directory using a known image database.
 */
#include "classify.h"

/**
 * Runs classification on objects in a directory , given a known image database, and puts the results in the output directory.
 * @param argc The number of command line arguments.
 * @param argv The command line arguments. The first argument is the path to the directory containing image features.
 */
int main(int argc, char** argv){
    
    if (argc < 3){
        std::cout << "Usage: ./classify <path_to_known_db> <path_to_unknown_images>" << std::endl;
        return -1;
    }

    std::string imageDBPath = argv[1];
    std::string unknownImageDirPath = argv[2];

    printf("Loading known features from %s\n", imageDBPath.c_str());
    printf("Classifying images in %s\n", unknownImageDirPath.c_str());

    ObjectFeatures knownFeatures = loadKnownFeatures(imageDBPath);

    if (knownFeatures.size() == 0){
        std::cout << "Error: No known features found" << std::endl;
        exit(-1);
    }

    printf("Loaded %d known features\n", knownFeatures.size());


    std::vector<FilePath> unknownImgPaths = getFilePathsFromDir(unknownImageDirPath, {".jpeg", "jpg", ".JPG", ".png", ".ppm", ".tif"});

    mkdir(PREDICTIONS_FOLDER, 0777);

    if (unknownImgPaths.size() == 0){
        std::cout << "Error: No images found to classify" << std::endl;
        exit(-1);
    }

    printf("Found %d images to classify\n", static_cast<int>(unknownImgPaths.size()));

    DistanceMetric* metrics[] = {new ScaledEuclideanDistance(), new CosineDistance(), new ChebyshevDistance(), new SimpleEuclideanDistance(), new ManhattanDistance()};

    for (DistanceMetric* metric : metrics){
        predict(unknownImgPaths, knownFeatures, *metric, std::string(PREDICTIONS_FOLDER) + "/" + metric->getName());
        delete metric; // free memory

    }

    return 0;

}