"""
Adithya Palle
Mar 14 2025
CS 5330 - Project 5 : Recognition using Deep Networks

This file defines the neural network for the digit recognition task, as well as the modeling parameters dataclass for the network.
"""
from torch import nn
import torch
import dataclasses


@dataclasses.dataclass
class ModelingParameters:
    """
    Parameters for building the model
    dropout_rate: the dropout rate after the 2nd convolutional layer (the probability that a neuron will be dropped)
    pooling_kernel_size: the pooling kernel size in both pooling layers
    conv_kernel_size: the convolutional kernel size  in both convolutional layers
    num_conv_filters: the number of convolutional filters in both convolutional layers
    """
    dropout_rate:float
    pooling_kernel_size:int
    conv_kernel_size:int
    num_conv_filters : int


class DigitDetectorNetwork(nn.Module):
    """
    Neural network for the digit recognition
    """

    def __init__(self, model_params:ModelingParameters = None):
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

        if model_params is None:
            # Define layers as fields
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5)  # in channel is 1 because we are using grayscale images, output layers defines the number of filters to be used
            # output is 24x24x10
            
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
            self.model_params = None
        else:
            self.init_from_params(model_params)
            self.model_params = model_params

        # compile the network
        self.compile_network()


    
    def init_from_params(self, 
                    model_params:ModelingParameters):
        """
        Initalizes the model with the given parameters

        The model will have 2 convolutional layers, each of which are followed by a max pooling layer and a ReLU activation function.
        The model will also have 2 fully connected layers, the first with 50 units and the second with 10 units.

        However, the dropout rate used after the 2nd convolutional layer, 
        the pooling kernel size after each conv layer,
        the convolutional kernel size of each convolutional layer, 
        and the number of convolutional filters in each layer can be specified

        Args:
            model_params: the parameters to use for the model
        Returns:
            None
        """
        super(DigitDetectorNetwork, self).__init__()
        conv_kernel_size = model_params.conv_kernel_size
        pooling_kernel_size = model_params.pooling_kernel_size
        num_conv_filters = model_params.num_conv_filters
        dropout_rate = model_params.dropout_rate
        if conv_kernel_size % 2 == 0:
            raise ValueError("conv_kernel_size must be odd")
        
        if pooling_kernel_size <= 0 or conv_kernel_size <= 0 or num_conv_filters <= 0:
            raise ValueError("pooling_kernel_size, conv_kernel_size, and num_conv_filters must be greater than 0")


        # Define layers as fields
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=num_conv_filters, kernel_size=conv_kernel_size)  # in channel is 1 because we are using grayscale images, output layers defines the number of filters to be used
        self.pool1 = nn.MaxPool2d(kernel_size=pooling_kernel_size)  # reduces dimensionality
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(in_channels=num_conv_filters, out_channels=
                               num_conv_filters, kernel_size=conv_kernel_size)  
        self.dropout = nn.Dropout(p=dropout_rate)
        self.pool2 = nn.MaxPool2d(kernel_size=pooling_kernel_size) 
        self.relu2 = nn.ReLU()
        
        self.flatten = nn.Flatten()


        # calculate number of in_features for the first fully connected layer
        orig_image_size = 28
        conv1_output_size = (orig_image_size - (conv_kernel_size) + 1) # because conv_kernel_size//2 is reduced from all sides due to valid convolution, but this is the same as subtracting by (conv_kernel_size - 1) since it is odd
        pool1_output_size = conv1_output_size // pooling_kernel_size # pooling divides the size by the pooling kernel size
        conv2_output_size = (pool1_output_size - (conv_kernel_size) + 1)

        if conv2_output_size <= 0:
            raise ValueError("First pooling output is too large for the 2nd convolutional kernel size")

        pool2_output_size = conv2_output_size // pooling_kernel_size # pooling divides the size by the pooling kernel size

        if pool2_output_size <= 0:
            raise ValueError("Final pooling resulting in invalid output")

        num_linear_units = pool2_output_size * pool2_output_size * num_conv_filters # since the last convolutional layer had {num_conv_filters} layers, and the image has {pool2_output_size} x {pool2_output_size} dimensions

        # add linear layers
        self.fc1 = nn.Linear(in_features=num_linear_units, out_features=50)
        self.fc2 = nn.Linear(in_features=50, out_features=10)  # output is 10 because we have 10 digits
        self.log_softmax = nn.LogSoftmax(dim=1)  # log_softmax for 0-1 values


    def compile_network(self):
        """
        Recompiles the network with the layers in the correct order.
        Use this after modifying the layers in the network.
        """
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