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
# Task 1
./draw_corners # simply draws the corners on the chessboard
# Tasks 2 + 3
./calibrate # draws corners on detected chessboard and you can press 's' to save calibration images and you can press 'c' once you have at least 5 to generate a calibration matrix to calibration_data.yaml
# Tasks 4 + 5
./get_camera_pos <path_to_calibration_yaml> # displays the rotation and translation of the camera relative to the chessboard, given a file with the camera matrix in it, also displays the 3d axes

# Tasks 6
./display_virtual_object <path_to_calibration_yaml> # displays a virtual object on the chessboard using the camera matrix from ./calibrate that is saved to a yml file
```

### Demo

Here is the [link](https://drive.google.com/file/d/1NfKYKOSyyKJFOculv36Y6CwRYzJjoWFY/view?usp=sharing) to a demo of ./get_camera_pos which shows how the translation and rotation
vectors change based on the change in rotation and translation of the chessboard.


## Time Travel

No time travel days will be used.
