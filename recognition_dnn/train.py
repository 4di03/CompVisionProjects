"""
Adithya Palle
Mar 14 2025
CS 5330 - Project 5 : Recognition using Deep Networks

This file the function for the entire training pipeline on the MNIST digit dataset, including training and evaluation.
"""
import torch.utils.data as data
import torch
from typing import Tuple
import dataclasses
from visualize import PlotData
from network import get_total_avg_loss
N_EPOCHS = 5 # number of epochs to train the network
BATCH_PLOT_INTERVAL = 50 # record train loss every {BATCH_PLOT_INTERVAL} samples


@dataclasses.dataclass
class TrainingParams:
    """
    Dataclass to store training parameters
    """
    batch_size: int
    n_epochs: int = N_EPOCHS


def train_network(network : torch.nn.Module, 
                  train_dataloader : torch.utils.data.DataLoader, 
                  test_dataloader : torch.utils.data.DataLoader = None,
                  n_epochs = N_EPOCHS) -> Tuple[PlotData, PlotData]:
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
    loss_function = torch.nn.NLLLoss()

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


def train_on_dataset(training_data : data.Dataset, 
                     test_data : data.Dataset, 
                     model : torch.nn.Module, 
                     training_params : TrainingParams) ->  Tuple[PlotData, PlotData]:
    """
    Trains the neural network on the given data with the given model and training parameters.
    Args:
        training_data: the training data
        test_data: the test data
        model: the model to train
        training_params: the training parameters
    Returns:
        PlotData: the training loss data
        PlotData: the test loss data

    """
    train_dataloader = torch.utils.data.DataLoader(training_data, batch_size=training_params.batch_size)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=training_params.batch_size)

    # train the network in-place using the training data, visualizing the loss and accuracy on the test data as well
    return train_network(model, train_dataloader, test_dataloader, n_epochs = training_params.n_epochs)
