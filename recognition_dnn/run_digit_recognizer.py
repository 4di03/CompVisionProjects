"""
Adithya Palle
Mar 14 2025
CS 5330 - Project 5 : Recognition using Deep Networks

This file reads a digit recognizer model from a file and runs it on the MNIST test set
"""
from network import DigitDetectorNetwork
import sys
import torch
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
from visualize import visualize_predictions
def main(argv):
    """
    Reads a digit recognizer model from a file and runs it on the MNIST test set
    Args:
        argv: command line arguments. The first is the path to the model file
    Returns:
        None
    """

    if len(argv) != 2:
        print("Usage: python run_digit_recognizer.py <model_file_path>")
        return

    model_file_path = argv[1]

    # Load the model
    model = DigitDetectorNetwork()
    model.load_state_dict(torch.load(model_file_path))

    # Load the test set
    test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
    

    # get predictions on first 10 test images
    preds = []
    labels = []
    images = []
    for i in range(10):
        img, label = test_set[i]
        img = img.unsqueeze(0)  # add a batch dimension
        predictions = model(img)
        prediction = torch.argmax(predictions, dim=1).item() # get the index of the max value on first dimensions since output is (1,10)
        formatted_preds = ", ".join([f"{x:.4f}" for x in predictions.squeeze().tolist()])

        print(f"Image {i}, All predictions: , [{formatted_preds}], Prediction: {prediction}, Actual label: {label}")
        images.append(img)
        labels.append(label)
        preds.append(prediction)
    

    # visualize the predictions
    visualize_predictions(images, preds, labels)



if __name__ == "__main__":
    main(sys.argv)