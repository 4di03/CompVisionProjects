"""
Adithya Palle
Mar 14 2025
CS 5330 - Project 5 : Recognition using Deep Networks

This file uses transfer learning on a model that was pre-trained to recognize digits to recognize Greek characters alpha, beta, and gamma.
It prints out the architectuer of the model, plots the training error, and tests the model on a directory of handwritten Greek characters.
"""
import sys
from network import DigitDetectorNetwork
import torch
import torchvision
from network import train_network
from visualize import visualize_loss, visualize_predictions
from run_on_handwritten_digits import get_predictions

IMAGE_SIZE = 128  



def main(argv):
    """
    Uses transfer learning on a model that was pre-trained to recognize digits to recognize Greek characters alpha, beta, and gamma.
    It prints out the architectuer of the model, plots the training error, and tests the model on a directory of handwritten Greek characters.
    Args:
        argv: command line arguments. The first is the path to the model file for the digit model, the 2nd is the path to the directory of training images of greek letters, 
        the third is the path to the directory of test images of greek letters (handwritten)
    """
    if len(argv) != 4:
        print("Usage: python transfer_learning_greek.py <model_file_path> <training_data_directory> <test_data_directory>")
        return
    
    model_file_path = argv[1]
    training_data_directory = argv[2]
    test_data_directory = argv[3]




if __name__ == "__main__":
    main(sys.argv)
