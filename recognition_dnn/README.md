## Group Members
Adithya Palle

## Local Setup

MacOS Sequoia 15.3 - Arm64 , Apple M1
Visual Studio Code
Python 3.10.16

I've added a `requirements.txt` file with all the requirements necessary. In a conda environment or python virtual environment, simply run:
`pip install -r requirements.txt` to install all relevant dependencies. 

An additional dependency that requires a homebrew install of `graphviz` is [torchview](https://github.com/mert-kurttutan/torchview), a package used
to visualize the network.



The python files can then be run as follows:

```
# Task 1A
# Visualizes the first 6 images from the test MNIST dataset
python visualize_data.py

# Task 1B
# visualizes the network architecture, saving it to `network_diagram.png`
python visualize_network.py

# Task 1C
# trains a network to recognize digits and saves it to output_file_path
python train_digit_recognizer.py <output_file_path> 

# Task 1E
# runs the network on the first 10 examples in the MNIST test set
# note that this only works with models of the default architecture (the one created in train_digit_recognizer.py)
python run_digit_recognizer.py <model_file_path>

# Task 1F 
# run the network on a directory of handwritten digits , showing an example of the processed digits and then the network's prediciton on them
python run_on_handwritten_digits.py <image_directory_path> <model_file_path>


# Task 2A + 2B
# Visualizes the filters learned in the first layer of the neural network for digit recognition, and applies them to the first image in the test set
# note that you will need to close the first plot (task 2A) to see the second one (task 2B)
python visualize_network_filters.py <model_file_path>


# Task 3
# applies transfer learning to the digit recognizer model to recognizing greek letters using the data in {training_data_directory}
# tests results on the greek letters in {test_data_directory}
python transfer_learning_greek.py <model_file_path> <training_data_directory> <test_data_directory>


# Task 4
# runs hyperparameter tuning using a json input file to find the model that performs the best in terms of performance with the least samples seen (on test data)
# runs on the MNIST digit dataset
python find_best_model.py <parameters_json_file> <output_file_path>

# Extension
# tunes a digit detector model specified by {model_file_path} with hand  gesture images
# and runs predictions on them. The hand gesture images should be in {image_directory_path}
python detect_hand_gestures.py <image_directory_path> <model_file_path>


```

For task 3, I've included a folder `handwritten_greek.zip` which
I used as the `test_data_directory` when evaluating my transfer learning model. 

For task 4, the parameter's I used for the randomized grid search
in my experiment are given in `training_parameters.json`. This file
can be used as the {parameters_json_file} argument for `find_best_model.py`

For the extensions, I've included a folder `hand_gestures_train.zip` which I used 
as the {training_data_directory} and a folder `hand_gestures_test.zip` which I used
as the {test_data_directory}


## Time Travel

No time travel days will be used.

## TODOS:

- finish exnteiosn
- finsih report (overview, reflection)
- review code, project doc, report
- final project proposal
- submit 