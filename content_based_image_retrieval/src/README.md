## Group Members
Adithya Palle

## Local Setup

MacOS Sequoia 15.2 - Arm64 
Visual Studio Code
Cmake 3.31.3

To compile the code. Simple run the following commands from root. Make sure you have cmake installed. 
```
mkdir build
cd build
cmake ..
make
```

The executable binaries should then appear in the build folder.

To run the matching application:

```
cd build
./img_retrieval <target image path> <directory of image dataset> <method of computing features> <distance metric> <N>
```
Running this command will output the top and bottom N matches to a folder of the name `tmp_output` in the `build` directory, where the images will be indexed according to their position in the ascending-sorted list of matches based on distance.

Please see the maps in `distanceMetric.cpp` and `featureExtractor.cpp` to see what each feature and distance method map to and for which tasks in the project they are for.

To compare the 4 key methods + the extension in the report for a given image, run :
```
cd build
./compare_methods <target image path> <directory of image dataset> <N>
```

This will create 4 seperate output folders in the `build` directory similar to `tmp_output` above name appropriately for each method.
Please look at `compare_methods.cpp` to see which output folder corresponds to which method if you are confused.

If you'd like to test the extension seperately, the name of the method is "EdgeUniformity" and the distance metric is "SSD_float"

## Time Travel

No time travel days will be used.
