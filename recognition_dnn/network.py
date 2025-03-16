"""
Adithya Palle
Mar 14 2025
CS 5330 - Project 5 : Recognition using Deep Networks

This file defines the neural network for the digit recognition task, as well as training and testing functions.
"""
from torch import nn
import torch
from typing import List
import dataclasses
from visualize import PlotData

N_EPOCHS = 5 # number of epochs to train the network
IMAGE_SIZE = 28 # image size for MINIST DIGIT images
BATCH_PLOT_INTERVAL = 50 # record train loss every {BATCH_PLOT_INTERVAL} samples



def get_total_avg_loss(network : nn.Module, dataloader : torch.utils.data.DataLoader, loss_function) -> float:
    """
    Computes the average loss on the entire dataset using the given network and data loader.

    Args:
        network: the network to use
        dataloader: the data loader to use
        loss_function: the loss function
    Returns:
        float: the loss on the entire dataset, averaged over all batches
    """
    total_batches = len(dataloader)

    if total_batches == 0:
        raise ValueError("Dataloader is empty")
    total_loss = 0
    network.eval()  # set network to evaluation mode

    with torch.no_grad():
        for test_images, test_labels in dataloader:
            test_output = network(test_images)
            total_loss += loss_function(test_output, test_labels).item()

    return total_loss / total_batches   


def train_network(network : nn.Module, 
                  train_dataloader : torch.utils.data.DataLoader, 
                  test_dataloader : torch.utils.data.DataLoader = None,
                  n_epochs = N_EPOCHS) -> None:
    """
    Trains the network using the training data and visualizes the loss and accuracy on the test data.
    Args:
        network: the network to train
        train_dataloader: the data loader for the training data
        test_dataloader: the data loader for the test data
    Returns:
        PlotData: the training loss data
        PlotData: the test loss data

    """

    # Define negative log likelihood loss
    loss_function = nn.NLLLoss()

    # Define optimizer
    optimizer = torch.optim.SGD(network.parameters(), lr=0.01, momentum=0.9)

    train_losses = []
    train_samples_seen = [] # epoch progress for each sample

    test_losses = []
    test_samples_seen = []  # epoch progress for each sample in test data


    seen_so_far = 0 # keep track of the total number of training samples trained on so far (double counts for repeated epochs)



    # record total average loss before training
    train_losses.append(get_total_avg_loss(network, train_dataloader, loss_function))
    train_samples_seen.append(seen_so_far)
    if test_dataloader is not None:
        test_losses.append(get_total_avg_loss(network, test_dataloader, loss_function))
        test_samples_seen.append(seen_so_far)

    
    print("Begin training...")
    # Train the network
    for epoch_index in range(n_epochs):  # loop over the dataset multiple times
        network.train()  # set network to training mode
        for batch_index, (train_images, train_labels) in enumerate(train_dataloader):
            seen_so_far += len(train_images)
            optimizer.zero_grad()  # zero the gradients for safety


            output = network(train_images)  # forward pass on training data

            train_loss = loss_function(output, train_labels)  # calculate loss
            train_loss.backward()  # backpropagation
            optimizer.step()  # update weights


            if batch_index % BATCH_PLOT_INTERVAL == 0:
                # store the loss and epoch progress
                train_losses.append(train_loss.item())
                train_samples_seen.append(seen_so_far)

        if test_dataloader is not None:
            # predict on test data after each epoch
            test_losses.append(get_total_avg_loss(network, test_dataloader, loss_function))
            test_samples_seen.append(seen_so_far)

        print(f"Epoch {epoch_index + 1} completed.")



    return PlotData(x=train_samples_seen, y=train_losses), PlotData(x=test_samples_seen, y=test_losses) if test_dataloader is not None else None

                


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

        # compile the network
        self.compile_network()

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