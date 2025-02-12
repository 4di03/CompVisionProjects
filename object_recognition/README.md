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
./image_dir_or <image_path>
./image_dir_or <directory of iamges>
./real_time_or # press the N key to save a feature vector for the frame
```
Running this command will output the resulting images and features to the `output` folder created in the same directory it is called.
get image path> <directory of image dataset> <N>

### Training

If you'd like to extract features' simply run `./real_time_or`, and type the 'n' key after clicking on the features Image window. Follow the prompt and this will save a feature vector in the `image_features` directory labeled according to your input.

## Time Travel

No time travel days will be used.


I decided to implement thresholding from scratch, including the kmeans implementation. The other methods were also implemented mostly from scratch.
Furthermore, I opt to build a real-time object recognizer (`real_time_or` binary), but have also included a binary (`image_or`) that can be used for running object recognition on an individual image for testing purposes.


The 5 objects I chose to identify are my wallet, my nail clipper, a key, a quarter, and a nintendo switch controller



Todos:
Task 1 
    - put images of segmentation in report
