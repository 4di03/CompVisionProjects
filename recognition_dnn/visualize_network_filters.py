"""
Adithya Palle
Mar 14 2025
CS 5330 - Project 5 : Recognition using Deep Networks

This file generates plots of the learned filters in the first convolutional layer of the DigitDetectorNetwork neural network, and applies them to the first image in the test set for the MNIST dataset.
"""
import sys
import torch
import matplotlib.pyplot as plt
from network import DigitDetectorNetwork
import numpy as np
import cv2
from torchvision import datasets
from torchvision import transforms
def main(argv):
    """
    Generate plots of the learned filters in the first convolutional layer of the DigitDetectorNetwork neural network, 
    and apply them to the first image in the test set for the MNIST dataset.
    Args:
        argv: command line arguments. The first is the path to the model file
    """
    if len(argv) != 2:
        print("Usage: python visualize_network_filters.py <model_file_path>")
        return

    model = DigitDetectorNetwork()
    model.load_state_dict(torch.load(argv[1]))

    # Get the weights of the first convolutional layer
    filters = model.conv1.weight.data

    # plot the 10 filters
    fig, axes = plt.subplots(3, 4, figsize=(10, 12))  


    # Find the min and max values across all filters for consistent color mapping
    vmin = np.min([filters[i].squeeze().numpy() for i in range(10)])
    vmax = np.max([filters[i].squeeze().numpy() for i in range(10)])

    for i in range(10):
        row, col = divmod(i, 4)
        img = axes[row, col].imshow(filters[i].squeeze(), cmap='coolwarm', vmin=vmin, vmax=vmax)
        axes[row, col].set_xticks([])  # Remove x-axis ticks
        axes[row, col].set_yticks([])  # Remove y-axis ticks
        axes[row,col].set_title(f"Filter {i}")
        axes[row, col].axis('off')  # Alternatively, turn off the entire axis

    # Hide unused subplots (for cases where you have fewer than 12 filters)
    for j in range(10, 12):
        row, col = divmod(j, 4)
        axes[row, col].axis('off')


    # Add a color bar aligned with the entire figure
    cbar_ax = fig.add_axes([0.92, 0.2, 0.02, 0.6])  # Position of color bar
    fig.colorbar(img, cax=cbar_ax)

    fig.suptitle("Visualization of CNN Filters", fontsize=16, fontweight="bold")
    plt.show()


    # Apply the filters to the first image in the test set and visualize the results
    test_set = datasets.MNIST(root='./data', train=False, download=True, transform = transforms.ToTensor())
    img, label = test_set[0]
    
    # apply a filter with opencv filter2D
    img = img.numpy().squeeze()
    filtered_imgs = []
    for i in range(10):
        filtered_img = cv2.filter2D(img, -1, filters[i].squeeze().numpy())
        filtered_imgs.append(filtered_img)
    
    fig, axes = plt.subplots(5,4, figsize=(8, 8))
    for i in range(0,20,2):
        row, col = divmod(i, 4)
        print(row, col)
        axes[row, col].imshow(filters[i//2].squeeze(), cmap='gray')
        axes[row, col].axis('off')
        axes[row, col+1].imshow(filtered_imgs[i//2], cmap='gray')
        axes[row, col+1].axis('off')

    
    plt.suptitle("Application of CNN Filters to Image", fontsize=16, fontweight="bold")
    plt.show()






if __name__ == "__main__":
    main(sys.argv)