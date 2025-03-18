from abc import ABC, abstractmethod
import torchvision
import torch
from network import DigitDetectorNetwork
from visualize import visualize_loss, visualize_predictions
from run_on_handwritten_digits import get_predictions
from train import train_network
import matplotlib.pyplot as plt
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

    



def get_dataloader_from_image_folder(directory, 
                                     transform : Transform,
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
                                          transform = transform),
        batch_size = batch_size,
        shuffle = True )

def run_transfer_learning(model_file_path : str, 
                          training_data_directory : str, 
                          test_data_directory : str, 
                          transform : Transform,
                          n_classes : int = 3):
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

        transform: the transform to apply to the images before training and prediction
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
    greek_train = get_dataloader_from_image_folder(training_data_directory,transform , 3)

    greek_test = get_dataloader_from_image_folder(test_data_directory,transform, 1)

    # visualize some of the train images
    img_batch, label_batch = next(iter(greek_train))
    img = img_batch[0]
    label = label_batch[0]

    print("L134", torch.max(img), torch.min(img))



    plt.imshow(img.squeeze().numpy(), cmap='gray')
    plt.title(f"Label: {label}")
    plt.show()
        



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