from abc import ABC, abstractmethod
import torchvision
import torch
from network import DigitDetectorNetwork, train_network
from visualize import visualize_loss, visualize_predictions
from run_on_handwritten_digits import get_predictions

IMAGE_SIZE = 128

# Neural Network for recognizing Greek characters
class GenericDetector(torch.nn.Module):


    def __init__(self, digit_model: DigitDetectorNetwork, n_classes :int = 3):
        """
        Initializes the GreekDetectorNetwork with the given digit model,
        by replacing the last layer of the digit model to recognize 3 classes.
        Warning: the digit model will be modified in-place.
        Args:
            digit_model: the model that was pre-trained to recognize digits
            n_classes: the number of classes to recognize
        """
        super(GenericDetector, self).__init__()
        # freeze the weights of the digit model that are already trained, so they are not updated during training
        for param in digit_model.parameters():
            param.requires_grad = False

        # Replace the last layer of the digit model to recognize 3 classes
        digit_model.fc2 = torch.nn.Linear(50, n_classes)
        digit_model.compile_network() # recompile the network after changing the last layer

        self.net = digit_model
    

    def forward(self, x):
        return self.net(x)
    
class Transform(ABC):
    @abstractmethod
    def __call__(self, x):
        """
        Preprocesses the image 
        Args:
            x: the image to preprocess (torch.Tensor)
        Returns:
            the preprocessed image (torch.Tensor)
        """
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
    



def get_dataloader_from_image_folder(directory, 
                                     greek_transform : GreekTransform,
                                     batch_size = 1):
    """
    Gets a DataLoader from an image folder, which has the following directory structure:
    directory
    ├── class1
    │   ├── img1.png
    │   ├── img2.png
    │   └── ...
    ├── class2
    │   ├── img1.png
    │   ├── img2.png
    │   └── ...
    └── ...
    Args:
        directory: the directory containing the images
        batch_size: the batch size for the DataLoader
    Returns:
        DataLoader: the DataLoader for the images in the directory
    """

    return torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder( directory,
                                          transform = torchvision.transforms.Compose( [torchvision.transforms.ToTensor(),
                                                                                       greek_transform,
                                                                                       torchvision.transforms.Normalize(
                                                                                           (0.1307,), (0.3081,) ) ] ) ),
        batch_size = batch_size,
        shuffle = True )

def run_transfer_learning(model_file_path : str, training_data_directory : str, test_data_directory : str, n_classes : int = 3):
    """
    Uses transfer learning on a model that was pre-trained to recognize digits to recognized n_classes from the images in the training_data_directory and test_data_directory
    Args:
        model_file_path: the path to the model file for the digit model
        training_data_directory: the path to the directory of training images
        test_data_directory: the path to the directory of test images

        Both training and test data directories should have the following structure:
        directory
            ├── class1
            │   ├── img1.png
            │   ├── img2.png
            │   └── ...
            ├── class2
            │   ├── img1.png
            │   ├── img2.png
            │   └── ...
            └── ...


        n_classes: the number of classes to recognize
    Returns:
        None
    """
    model = DigitDetectorNetwork()
    model.load_state_dict(torch.load(model_file_path))

    greek_model = GenericDetector(model, n_classes=n_classes) # transfer learning by replacing the last layer of the digit model to recognize 3 classes

    # print the architecture of the model
    print("Greek Detector Network Architecture:")
    print(greek_model)


    # DataLoader for the Greek data set
    greek_train = get_dataloader_from_image_folder(training_data_directory, GreekTransform(original_image_size=IMAGE_SIZE, shrink_target = 28), 5)

    greek_test = get_dataloader_from_image_folder(test_data_directory,GreekTransform(original_image_size=IMAGE_SIZE, shrink_target = 28), 1)


    train_loss, test_loss = train_network(model, greek_train, greek_test, n_epochs = 5)

    visualize_loss(train_loss, test_loss)
    

    # Get predictions on the test data

    test_images = []
    test_labels = []
    for img, label in greek_test:
        # img will have shape [1,1,28,28], we want to remove the batch dimension and convert it to [1,28,28]
        test_images.append(img.squeeze().unsqueeze(0))
        test_labels.append(label.item())

    predictions = get_predictions(greek_model, test_images)

    visualize_predictions(test_images, predictions,test_labels, (3,3))