"""
Adithya Palle
Mar 18 2025
CS 5330 - Project 5 : Recognition using Deep Networks

This file uses transfer learning on a model that was pre-trained to recognize digits to recognize  hand gestures.
It prints out the architecture of the model, plots the training error, and tests the model on a directory of hand gestures.
"""
import sys
from transfer_learning import run_transfer_learning, Transform , IMAGE_SIZE
import torchvision 
import torch

RGB_LOWER_BOUND = torch.tensor([0/255, 80/255, 31/255])
RGB_UPPER_BOUND = torch.tensor([255/255, 150/255, 255/255])
# greek data set transform
class HandGestureTransform(Transform):


    def __call__(self, x : torch.Tensor) -> torch.Tensor:
        """
        Preprocesses the image to be compatible with the digit model
        Resizes the image to 28x28 and sets values between (0, 85, 31)and (255, 120, 255) as white, everything else as black,
        and converts the image to grayscale
        Args:
            x: the image to preprocess
        Returns:
            the preprocessed image
        """
        x = torchvision.transforms.functional.affine( x, 0, (0,0), 28/IMAGE_SIZE, 0 )
        x = torchvision.transforms.functional.center_crop( x, (28, 28) )

        # set values between (0, 85, 31) and (255, 120, 255) as white, everything else as black
        mask = ((x >= RGB_LOWER_BOUND.view(3, 1, 1)) & 
                (x <= RGB_UPPER_BOUND.view(3, 1, 1)))
        
        
        x = torch.where(mask.all(dim=0), 
                        torch.tensor(1.0, dtype=x.dtype, device=x.device), 
                        torch.tensor(0.0, dtype=x.dtype, device=x.device))

        return x.unsqueeze(0)  # add a batch dimension

    
def main(argv):
    """
    Uses transfer learning on a model that was pre-trained to recognize digits to recognize Greek characters alpha, beta, and gamma.
    It prints out the architectuer of the model, plots the training error, and tests the model on a directory of handwritten Greek characters.
    Args:
        argv: command line arguments. The first is the path to the model file for the digit model, the 2nd is the path to the directory of training images of greek letters, 
        the third is the path to the directory of test images of greek letters (handwritten)
    """
    if len(argv) != 4:
        print("Usage: python transfer_learning_greek.py <model_file_path> <training_data_directory> <test_data_directory>")
        return
    
    model_file_path = argv[1]
    training_data_directory = argv[2]
    test_data_directory = argv[3]
    # make no assumption about the mean and std of the images
    transform = torchvision.transforms.Compose( [torchvision.transforms.ToTensor(),
                                                                HandGestureTransform() ] )
    
    run_transfer_learning(model_file_path, 
                          training_data_directory, 
                          test_data_directory, 
                          transform,
                          n_classes = 3)


if __name__ == "__main__":
    main(sys.argv)
