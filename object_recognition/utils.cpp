/**
 * Adithya Palle
 * Feb 15 2025
 * CS 5330 - Project 3 : Real-time 2D Object Recognition
 * 
 * Utility file for miscellaneous utility functions.
 */

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
 * Gets the FilePaths from a directory for  files with the {extensions}.
 * @param dirPath The path to the directory.
 * @extension The list of extensions to look for.
 * @return A vector of FilePaths in the directory.
 */
std::vector<FilePath> getFilePathsFromDir(std::string dirPath, const std::vector<std::string>& extensions){
    DIR* dirp = opendir(dirPath.c_str());
    if (dirp == NULL) {
        printf("Cannot open directory %s\n", dirPath.c_str());
        exit(-1);
    }
    char buffer[MAX_FILE_NAME_LENGTH]; // buffer to store the full path name of the file
    struct dirent *dp;  
    std::vector<FilePath> filepaths;

    std::vector<FilePath> knownFeatures;
    while( (dp = readdir(dirp)) != NULL ) {
        for(std::string extension : extensions){
            if (strstr(dp->d_name, extension.c_str())){
                FilePath filePath = {dirPath, dp->d_name};
                filepaths.push_back(filePath);
                break;
            }
        }
    }
    return filepaths;
}