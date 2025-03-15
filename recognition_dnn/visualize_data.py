"""
Adithya Palle
Mar 14 2025
CS 5330 - Project 5 : Recognition using Deep Networks

This file generates a plot of the first six example digits from the test set for the MNISt dataset.
"""
import matplotlib.pyplot as plt
from torchvision import datasets

def main():
    """
    Generate a plot of the first six example digits from the test set for the MNISt dataset.
    """

    # Load the test set
    test_set = datasets.MNIST(root='./data', train=False, download=True)

    # Plot the first six images
    fig, axes = plt.subplots(1, 6, figsize=(10, 3))
    for i in range(6):
        img, label = test_set[i]
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f"Label: {label}")
        axes[i].axis('off')
    plt.show()

if __name__ == "__main__":
    main()