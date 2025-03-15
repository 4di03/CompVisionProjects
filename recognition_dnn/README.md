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
# visualizes the network architecture
python visualize_network.py

# Task 1C
# trains a network to recognize digits and saves it to output_file_path
python train_digit_recognizer.py <output_file_path> 

# Task 1E
# runs the network on the first 10 examples in the MNIST test set
python run_digit_recognizer.py <model_file_path>

# Task 1F
# runs the network on a directory of handwritten digits 
python run_on_handwritten_digits.py <image_directory_path> <model_file_path>



# Task 2A + 2B
# Visualizes the filters learned in the first layer of the neural network for digit recognition, and applies them to the first image in the test set
python visualize_network_filters.py <model_file_path>
```

### Demo


## Testing Extension


## Time Travel

No time travel days will be used.

## TODOS:

- review task 1
- do report stuff for task 1
