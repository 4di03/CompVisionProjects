

#include <iostream>
#include <stdio.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#include "utils.h"
#define MAX_FILE_NAME_LENGTH 256


/**
 * Gets the FilePaths from a directory.
 * @param dirPath The path to the directory.
 * @return A vector of FilePaths in the directory.
 */
std::vector<FilePath> getFilePathsFromDir(std::string dirPath){
    DIR* dirp = opendir(dirPath.c_str());
    if (dirp == NULL) {
        printf("Cannot open directory %s\n", dirPath.c_str());
        exit(-1);
    }
    char buffer[MAX_FILE_NAME_LENGTH]; // buffer to store the full path name of the image
    // loop over all the files in the image file listing
    struct dirent *dp;  
    std::vector<FilePath> filepaths;

    std::vector<FilePath> knownFeatures;
    while( (dp = readdir(dirp)) != NULL ) {
        // check if the file is an image
        if( strstr(dp->d_name, ".features") ) {
            // Implicit conversion (creates a copy)
            FilePath filePath = FilePath{dirPath, dp->d_name};
            filepaths.push_back(filePath);
            
        }
    }
    return filepaths;
}