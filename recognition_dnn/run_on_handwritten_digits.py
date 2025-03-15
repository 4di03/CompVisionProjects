"""
Adithya Palle
Mar 14 2025
CS 5330 - Project 5 : Recognition using Deep Networks

This file reads a directory of handwritten digit images, processes them, and uses a trained neural network to recognize the digits.
"""
import sys
import numpy as np
from network import DigitDetectorNetwork
import torch
import cv2
from typing import List
from visualize import visualize_predictions
import os

def load_images(image_directory:str) -> List[cv2.Mat]:
    """
    Loads images from a directory
    Args:
        image_directory: the directory containing the images
    Returns:
        List[cv2.Mat]: the loaded images
    """
    
    if not os.path.isdir(image_directory):
        raise ValueError(f"Directory {image_directory} does not exist")
    
    images = []

    for file in os.listdir(image_directory):
        if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg"):
            image = cv2.imread(os.path.join(image_directory, file), cv2.IMREAD_GRAYSCALE)
            images.append(image)
    return images

def process_images(images:List[cv2.Mat]) -> List[torch.Tensor]:
    """
    Since the images are expected to be black on white, we invert the colors and resize them to 28x28, and conver to tensor
    Args:
        images: the images to process
    Returns:
        List[torch.Tensor]: the processed images
    """

    processed_images = []

    for image in images:
        # Invert the colors
        image = cv2.bitwise_not(image)

        # Resize the image to 28x28
        image = cv2.resize(image, (28, 28))

        # Convert to tensor
        image = torch.tensor(image).float()

        # Normalize the image
        image = image / 255.0

        # Add a batch dimension
        image = image.unsqueeze(0)

        processed_images.append(image)

    return processed_images


def get_predictions(model:DigitDetectorNetwork, images:List[torch.Tensor]) -> List[int]:
    """
    Get predictions from the model
    Args:
        model: the model to use
        images: the images to predict
    Returns:
        List[int]: the predicted digits
    """
    predictions = []
    batch = torch.stack(images) # put all images in a single batch, so we can process them all at once
    predictions = model(batch)
    predictions = torch.argmax(predictions, dim=1).tolist()
    return predictions

def main(argv):
    """
    Reads a directory of handwritten digit images, processes them, and uses a trained neural network to recognize the digits.
    Args:
        argv: command line arguments. The first is the path to the directory of images, the second is the path to the model file
    Returns:
        None
    """


    if len(argv) != 3:
        print("Usage: python run_on_handwritten_digits.py <image_directory> <model_file_path>")
        return

    image_directory = argv[1]
    model_file_path = argv[2]

    # Load the model
    model = DigitDetectorNetwork()
    model.load_state_dict(torch.load(model_file_path))

    # Load the images
    images = load_images(image_directory)

    if len(images) == 0:
        print(f"No images found in directory {image_directory}")
        return

    # Process the images
    processed_images = process_images(images)


    # visualize the processed images
    visualize_predictions(processed_images, None, None, (2,5))

    # Get predictions
    predictions = get_predictions(model, processed_images)


    visualize_predictions(images, predictions,None, (2,5))


if __name__ == "__main__":
    main(sys.argv)