"""
Adithya Palle
Mar 14 2025
CS 5330 - Project 5 : Recognition using Deep Networks

This file uses transfer learning on a model that was pre-trained to recognize digits to recognize Greek characters alpha, beta, and gamma.
It prints out the architectuer of the model, plots the training error, and tests the model on a directory of handwritten Greek characters.
"""
import sys
from transfer_learning import run_transfer_learning, Transform , IMAGE_SIZE
import torchvision 


# greek data set transform
class GreekTransform(Transform):

    def __init__(self, original_image_size = 128, shrink_target = 36):
        """
        Initializes the GreekTransform with the original image size
        Args:
            original_image_size: the size of the original image ( assumed to be square )
            shrink_target: the size to shrink the image to before applying a 28x28 center crop
        """
        self.original_image_size = original_image_size
        self.shrink_target = shrink_target
    def __call__(self, x):
        """
        Preprocesses the image to be compatible with the digit model
        Converts the image to grayscale, resizes it to 28x28, and inverts the colors
        Args:
            x: the image to preprocess
        Returns:
            the preprocessed image
        """
        x = torchvision.transforms.functional.rgb_to_grayscale( x )
        x = torchvision.transforms.functional.affine( x, 0, (0,0), self.shrink_target/self.original_image_size, 0 )
        x = torchvision.transforms.functional.center_crop( x, (28, 28) )
        return torchvision.transforms.functional.invert( x )
    
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
    greek_transform = GreekTransform(original_image_size=IMAGE_SIZE, shrink_target = 28)
    transform = torchvision.transforms.Compose( [torchvision.transforms.ToTensor(),
                                                                                       greek_transform,
                                                                                       torchvision.transforms.Normalize(
                                                                                           (0.1307,), (0.3081,) ) ] )
    run_transfer_learning(model_file_path, 
                          training_data_directory, 
                          test_data_directory, 
                          transform,
                          n_classes = 3)


if __name__ == "__main__":
    main(sys.argv)
