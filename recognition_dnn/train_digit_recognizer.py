"""
Adithya Palle
Mar 14 2025
CS 5330 - Project 5 : Recognition using Deep Networks

This file is the entrypoint for the program that trains a deep network to recognize digits and saves it to a file.
"""
# import statements
import sys
from typing import List
import torch
from torch import nn
from torchvision import datasets
import matplotlib.pyplot as plt
from torchvision import transforms
from visualize import visualize_loss
from network import DigitDetectorNetwork
from network import train_network



            


def main(argv : List[str]) -> None:
    """
    Reads the file path to save the trained network to from the command line arguments and trains a network to recognize digits, saving it to the specified file.

    Args:
        argv: list of command line arguments. The first argument is the file path to save the trained network to.
    Returns:
        None
    """
    # handle any command line arguments in argv
    if len(argv) != 2:
        print("Usage: python train_digit_recognizer.py <output_file_path>")
        return
    output_file_path = argv[1]
    
    # download training data
    print("Downloading data...")
    training_data = datasets.MNIST(root='data', train=True, download=True, transform=transforms.ToTensor())

    # download test data
    test_data = datasets.MNIST(root='data', train=False, download=True, transform=transforms.ToTensor())
    print("Data downloaded.")


    # create data loaders for mini-batch training
    batch_size = 64
    train_dataloader = torch.utils.data.DataLoader(training_data, batch_size=batch_size)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)


    # create the network
    network = DigitDetectorNetwork()

    # train the network in-place using the training data, visualizing the loss and accuracy on the test data as well
    train_loss_data, test_loss_data = train_network(network, train_dataloader, test_dataloader)

    # save the network to the specified file
    torch.save(network.state_dict(), output_file_path)
    print(f"Network saved to {output_file_path}")


    # plot the training and test loss
    visualize_loss(train_loss_data, test_loss_data)
                


    return

if __name__ == "__main__":
    main(sys.argv)