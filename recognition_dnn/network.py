"""
Adithya Palle
Mar 14 2025
CS 5330 - Project 5 : Recognition using Deep Networks

This file defines the neural network for the digit recognition task.
"""
from torch import nn
import torch

class DigitDetectorNetwork(nn.Module):
    """
    Neural network for the digit recognition
    """

    def __init__(self):
        """
        Initializes the network with the following layers:
        Input - 28x28 grayscale image
        1. Convolutional Layer with 10 filters of size 5x5 - outputs 24x24x10 mat
        2. Max pooling layer with 2x2 window followed by ReLU activation - outputs 12x12x10 mat
        3. Convolutional Layer with 20 filters of size 5x5 - outputs 8x8x20 mat
        4. A dropout layer with 0.3 dropout rate
        5. Max pooling layer with 2x2 window followed by ReLU activation - outputs 4x4x20 mat 
        6. Flattening layer - outputs 320 vector
        7. Fully connected layer with 50 units and relu activation - outputs 50 vector
        8. Fully connected layer with 10 units and log_softmax activation - outputs 10 vector
        
        """

        super(DigitDetectorNetwork, self).__init__()


        # Define layers as fields
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5)  # in channel is 1 because we are using grayscale images, output layers defines the number of filters to be used
        self.pool1 = nn.MaxPool2d(kernel_size=2)  # reduces dimensionality
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=
                               20, kernel_size=5)  # output is 12x12x20
        self.dropout = nn.Dropout(p=0.3)
        self.pool2 = nn.MaxPool2d(kernel_size=2)  # output is 6x6x20
        self.relu2 = nn.ReLU()
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=320, out_features=50)
        self.fc2 = nn.Linear(in_features=50, out_features=10)  # output is 10 because we have 10 digits
        self.log_softmax = nn.LogSoftmax(dim=1)  # log_softmax for 0-1 values

        # Store layers in Sequential
        self.net = nn.Sequential(
            self.conv1,
            self.pool1,
            self.relu1,
            self.conv2,
            self.dropout,
            self.pool2,
            self.relu2,
            self.flatten,
            self.fc1,
            self.fc2,
            self.log_softmax
        )


    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """
        Computes a forward pass for the network.

        Args:
            x: input tensor
        Returns:
            output tensor
        """

        return self.net(x)