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

Make sure to update the location of the onnxruntime installation in the CMakeLists.txt before running the above command. 
In the current file, it is set to be installed in depthAnything/<onnx_folder_name>, but please update if it is installed elsewhere in your system.
The executable binaries should then appear in the build folder.

To run the video display application:

```
cd build
./img_display <target image path> <directory of images path> <method of computing features> <distance metric> <N>
```
Please see the maps in `distanceMetric.h` and `featureExtractor.h` to see what each feature
and distance method map to.


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

