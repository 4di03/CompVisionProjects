"""
Adithya Palle
Mar 17 2025
CS 5330 - Project 5 : Recognition using Deep Networks

This file takes in a json file with parameters and an output file path and runs hyperparameter tuning on the model to find the
best model based on loss and training time. The best model is saved to the output file path
"""
import random
import sys
import time
from train import TrainingParams
from typing import List, Tuple, TypeAlias
import torch
from torchvision import datasets, transforms
from train import train_on_dataset
import json
from network import ModelingParameters, DigitDetectorNetwork


MAX_PARAM_SPACE_SIZE = 50 # max configurations to try

# modeling config consists of the model and the training parameters
ModelingConfig : TypeAlias = Tuple[DigitDetectorNetwork, TrainingParams]

def get_param_space(json_file_path : str) -> List[ModelingConfig]:
    """
    Gets a list of training parameter and model configurations based on the parameters in the json file

    The json must have structure:
    {
    "model_params":{
    "dropout_rate": [val1, val2, val3],
    "pooling_kernel_size": [val1, val2, val3],
    "conv_kernel_size": [val1, val2, val3],
    "num_conv_filters": [val1, val2, val3],
    },
    "training_params":{
    "n_epochs": [val1, val2, val3],
    "batch_size": [val1, val2, val3],
    }
    }

    where param1, param2, etc. are the names of the parameters that are in constructors

    Args:
        json_file_path: path to the json file
    Returns:
        List of tuples with the first element being the model and the second element being the training parameters
        Can be used for grid search or random search or any other hyperparameter tuning method
    """

    json_obj = json.load(open(json_file_path, 'r'))
    param_space = []
    model_params = json_obj["model_params"]
    for dr in model_params["dropout_rate"]:
        for pks in model_params["pooling_kernel_size"]:
            for cvs in model_params["conv_kernel_size"]:
                for ncf in model_params["num_conv_filters"]:
                    for ne in json_obj["training_params"]["n_epochs"]:
                        for bs in json_obj["training_params"]["batch_size"]:
                            try:
                                mp = ModelingParameters(dropout_rate=dr, pooling_kernel_size=pks, conv_kernel_size=cvs, num_conv_filters=ncf)
                                model = DigitDetectorNetwork(mp)
                            except ValueError as e:
                                print(f"Skipping configuration with dropout rate {dr}, pooling kernel size {pks}, convolutional kernel size {cvs}, and number of convolutional filters {ncf}")
                                print(e)
                                continue
                            params = TrainingParams(n_epochs=ne, batch_size=bs)
                            param_space.append((model, params))
    return param_space
    




def get_best_model(training_times : List[float], losses : List[float], models : List[Tuple[ModelingConfig]]) -> Tuple[ModelingConfig, float, float]:
    """
    Gets the best model based on the raw scores
    Applies min-max scaling to the raw scores and then adds the loss and training time and takes
    to get a score that weights both loss and training time equally
    Then returns the model with the lowest score (meaning least combined loss and training time)
    Args:
        raw_scores: list of tuples with the first element being the raw score and the second element being the model and parameters
    Returns:
        The best model and parameters
        The loss of the best model
        The training time of the best model
    """
    min_loss = min(losses)
    max_loss = max(losses)
    # normalize the losses such that the smallest loss is 0 and the largest loss is 1
    norm_losses = [(l - min_loss) / (max_loss - min_loss) for l in losses]
    min_time = min(training_times)
    max_time = max(training_times)
    # normalize the training times such that the smallest time is 0 and the largest time is 1
    norm_times = [(t - min_time) / (max_time - min_time) for t in training_times]

    scores = [l + t for l, t in zip(norm_losses, norm_times)]
    # finds the smallest joint sum
    best_index = scores.index(min(scores))
    return models[best_index], losses[best_index], training_times[best_index]

def main(argv):
    """
    Takes in a json file with parameters and an output file path and runs hyperparameter tuning on the model to find the
    best model based on loss and training time
    Args:
        argv: command line arguments. The first is the path to the json file and the second is the output file path
    Returns:
        None
    """
    if len(argv) != 3:
        print("Usage: python find_best_model.py <json_file_path> <output_file_path>")
        return

    json_file_path = argv[1]
    output_file_path = argv[2]


    param_space = get_param_space(json_file_path)

    if len(param_space) > MAX_PARAM_SPACE_SIZE:
        # randomly pick MAX_PARAM_SPACE_SIZE configurations to try
        param_space = random.sample(param_space, MAX_PARAM_SPACE_SIZE)

    print(f"Running hyperparameter tuning on {len(param_space)} configurations")

    # download training data
    print("Downloading data...")
    training_data = datasets.MNIST(root='data', train=True, download=True, transform=transforms.ToTensor())

    # download test data
    test_data = datasets.MNIST(root='data', train=False, download=True, transform=transforms.ToTensor())
    print("Data downloaded.")
    # run training for each configuration
    training_times = []
    losses = []
    for model, params in param_space:


        start = time.time()
        _, test_loss_data = train_on_dataset(training_data, test_data, model, params)
        end = time.time()
        training_time = end - start

        training_times.append(training_time)
        losses.append(test_loss_data.get_final_loss())


    # get the best model
    best_model, best_model_loss, best_model_training_time = get_best_model(training_times, losses, param_space)


    # print best loss and training time
    print(f"Best model's NLL loss: {best_model_loss}")
    print(f"Best model's training time: {best_model_training_time}")

    # save the best model
    model, params = best_model
    torch.save(model.state_dict(), output_file_path)
    print(f"Best model saved to {output_file_path}")
    print(f"Best model parameters: {params} {model.model_params}")


    
    return

if __name__ == "__main__":
    main(sys.argv)