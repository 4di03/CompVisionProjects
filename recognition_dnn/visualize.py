"""
Adithya Palle
Mar 14 2025
CS 5330 - Project 5 : Recognition using Deep Networks


Functions to visualize the predictions made by the model
"""

from typing import List, Tuple
import torch
import matplotlib.pyplot as plt

def visualize_predictions(images: List[torch.Tensor], preds : List[int] = None, labels : List[int] = None, plot_shape : Tuple[int,int]=(2,5)):
    """
    Visualizes the predictions made by the model
    Args:
        images: the images
        preds: the predicted labels
        labels: the actual labels
        plot_shape: the shape of the plot. The product of this tuple is the number of images to plot
    Returns:
        None
    """
    num_images = plot_shape[0] * plot_shape[1]
    if num_images != len(images):
        raise ValueError(f"Number of images ({len(images)}) does not match the plot shape ({plot_shape})")
    
    # Plot the first 10 images in a 5-row, 2-column format
    fig, axes = plt.subplots(*plot_shape, figsize=(10,10))  # Adjust figsize for better layout

    for i in range(num_images):
        row, col = divmod(i, plot_shape[1])  # Compute row and column indices
        axes[row, col].imshow(images[i].squeeze(), cmap='gray')
        pred_str = f"Pred: {preds[i]}," if preds is not None else ""
        label_str = f"Actual: {labels[i]}" if labels is not None else ""
        axes[row, col].set_title(f"{pred_str} {label_str}")
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.show()