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
./get_matches <target image path> <directory of images path> <method of computing features> <distance metric> <N>
```

Please see the maps in `distanceMetric.h` and `featureExtractor.h` to see what each feature
and distance method map to.

Running this commandwill output the top N matches to a folder of the name `tmp_output` in the `build` directory, where the images are indexed by their index in the ascending-sorted list of matches based on distance.

To compare the 4 key methods in the report for a given image, run :
```
cd build
./compare_methods <target image path> <directory of images path> <N>
```

This will create 4 seperate output folders in the `build` directory similar to `tmp_output` above name appropriately for each method.

## Time Travel

No time travel days will be used.


### Todos
- finsih resport stuff for 5 and 6 ( may need to make script to streamline comparison)
- finnish task 7  
- finish extension
- clean up comments in teh code, explain reason for std::vector<cv::Mat> return type in comments
- add headers to each file
- reread report + code
- upload stuff then verify submission is good

