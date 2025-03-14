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
import dataclasses
N_EPOCHS = 5 # number of epochs to train the network
IMAGE_SIZE = 28 # image size for MINIST DIGIT images
BATCH_PLOT_INTERVAL = 50 # record train loss every {BATCH_PLOT_INTERVAL} samples

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
        
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5)  # output is 12x12x20
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
@dataclasses.dataclass
class PlotData:
    """
    Dataclass to store x and y values for plotting
    """
    x: List[float] # x values
    y: List[float] # y values


def get_total_avg_loss(network : DigitDetectorNetwork, dataloader : torch.utils.data.DataLoader, loss_function) -> float:
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


def train_network(network : DigitDetectorNetwork, train_dataloader : torch.utils.data.DataLoader, test_dataloader : torch.utils.data.DataLoader) -> None:
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
    test_losses.append(get_total_avg_loss(network, test_dataloader, loss_function))
    test_samples_seen.append(seen_so_far)

    
    print("Begin training...")
    # Train the network
    for epoch_index in range(N_EPOCHS):  # 10 epochs
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
        
        # predict on test data after each epoch
        test_losses.append(get_total_avg_loss(network, test_dataloader, loss_function))
        test_samples_seen.append(seen_so_far)

        print(f"Epoch {epoch_index + 1} completed.")



    return PlotData(x=train_samples_seen, y=train_losses), PlotData(x=test_samples_seen, y=test_losses)

                



            


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



                
    # Plot training and test loss against epoch progress
    plt.plot(train_loss_data.x, train_loss_data.y, label='Train Loss')

    # plot test as scatter plot
    plt.scatter(test_loss_data.x, test_loss_data.y, label='Test Loss', color='red')
    plt.title('Training and Test Loss')

    plt.xlabel('Epoch Progress')
    plt.ylabel('Negative Log Likelihood Loss')
    plt.legend()
    plt.show()


    return

if __name__ == "__main__":
    main(sys.argv)