/**
 * Adithya Palle
 * Feb 15 2025
 * CS 5330 - Project 3 : Real-time 2D Object Recognition
 * 
 * Header file for miscellaneous utility functions.
 */

#include <iostream>
#pragma once

// dataclass that encapsulate all essential information about a file path
struct FilePath{
    std::string directoryPath;
    std::string fileName;


    /**
     * Gets the name of the file without the extension.
     */
    std::string getName(){
        return fileName.substr(0, fileName.find_last_of("."));
    }

    /**
     * Gets the full path of the file.
     */
    std::string getFullPath(){
        return directoryPath + "/" + fileName;
    }

    /**
     * Gets the extension of the file.
     */
    std::string getExtension(){
        return fileName.substr(fileName.find_last_of("."));
    }
};



std::vector<FilePath> getFilePathsFromDir(std::string dirPath, const std::vector<std::string>& extensions);

