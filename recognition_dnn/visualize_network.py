"""
Adithya Palle
Mar 14 2025
CS 5330 - Project 5 : Recognition using Deep Networks

This file generates a diagram of the DigitDetectorNetwork neural network.
"""
import torch
from torchview import draw_graph
from network import DigitDetectorNetwork


def main():
    """ Generate a diagram of the DigitDetectorNetwork neural network, saving it to network_diagram.png"""

    # Create an instance of the network
    network = DigitDetectorNetwork()


    #visualize the network with torchview
    # shape is (1,1,28,28) because we are using one grayscale images of size 28x28
    model_graph = draw_graph(network, input_size = (1,1,28,28), save_graph= True, filename = "network_diagram")

if __name__ == "__main__":
    main()