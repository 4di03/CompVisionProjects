## Group Members
Adithya Palle

## Local Setup
MacOS Sequoia 15.2 - Arm64 
Visual Studio Code
Cmake 3.31.3

To compile the code. Simple run the following commands from root.
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
build/video_display
```

## Time Travel

No time travel days will be used.