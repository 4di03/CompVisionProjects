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
./image_or <image_path>
./image_dir_or <directory of iamges> --save_features
./real_time_or # press the N key to save a feature vector for the frame
./classify <path_to_feature_db> <path_to_images_db>
./compareDistanceMetrics <path_to_feature_db> <path_to_images_db>
./test_cleanup
```

Please see `CMakeLists.txt` to see which files correspond to which executables (look for "add_executable").


Running `./image_dir_or` this command will output the resulting images and features to the `image_features` folder (if run with the --save_features flag) created in the same directory it is called.
get image path> <directory of image dataset> <N>

Running `./classify` or `./compareDistanceMetrics` will output predictions to the created `predictions` folder. You can then run `confusion_matrix.ipynb` with the PREDICTIONS_FOLDER_PATH updated accordingly


### Training

If you'd like to extract features' simply run `./real_time_or`, and type the 'n' key after clicking on the features Image window. Follow the prompt and this will save a feature vector in the `image_features` directory labeled according to your input.

If you'd like to run classification with scaled euclidean distance, simply run:

`./classify <path_to_known_db> <path_to_unknown_images>`

The output will be placed in `predictions/default` relative to where the code was ran.


If you'd like to run classification with all 5 distance metrics, run:

`./compareDistanceMetrics <path_to_known_db> <path_to_unknown_images>`

The output will be placed in `predictions/<distance_method_name>` relative to where the code was ran.

### Testing Extensions

If you'd like to test the grassfire algorithm , simply run `./test_cleanup` and you can see
the results in the created `test_output` folder, as well as results in the terminal log.

### Demo

Please follow this [link](https://drive.google.com/file/d/1bUER57udyDUkTDl73tSUmmATqzM8e60z/view?usp=sharing) to see a demo of the application run on a real time video feed.


## Time Travel

No time travel days will be used.
