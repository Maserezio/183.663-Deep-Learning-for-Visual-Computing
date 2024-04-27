import os
import pickle
from typing import Tuple
import numpy as np

# from dlvc.datasets.dataset import  Subset, ClassificationDataset
from .dataset import Subset, ClassificationDataset


class CIFAR10Dataset(ClassificationDataset):
    '''
    Custom CIFAR-10 Dataset.
    '''

    def __init__(self, fdir: str, subset: Subset, transform=None):
        '''
        Loads the dataset from a directory fdir that contains the Python version
        of the CIFAR-10, i.e. files "data_batch_1", "test_batch" and so on.
        Raises ValueError if fdir is not a directory or if a file inside it is missing.

        The subsets are defined as follows:
          - The training set contains all images from "data_batch_1" to "data_batch_4", in this order.
          - The validation set contains all images from "data_batch_5".
          - The test set contains all images from "test_batch".

        Images are loaded in the order they appear in the data files
        and returned as uint8 numpy arrays with shape (32, 32, 3), in RGB channel order.
        '''

        self.classes = ('plane', 'car', 'bird', 'cat',
                        'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        ## TODO implement
        # See the CIFAR-10 website on how to load the data files
        if not os.path.isdir(fdir):
            raise ValueError(f"{fdir} is not a directory")

        self.transform = transform
        self.data = []
        self.labels = []

        if subset == Subset.TRAINING:
            batch_files = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4']
        elif subset == Subset.VALIDATION:
            batch_files = ['data_batch_5']
        elif subset == Subset.TEST:
            batch_files = ['test_batch']
        else:
            raise ValueError("Invalid subset")

        for batch_file in batch_files:
            file_path = os.path.join(fdir, batch_file)
            if not os.path.isfile(file_path):
                raise ValueError(f"File {file_path} is missing")

            with open(file_path, 'rb') as file:
                batch_data = pickle.load(file, encoding='latin1')  # or encoding='bytes'
                self.data.append(batch_data['data'])
                self.labels += batch_data['labels']

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        self.labels = np.array(self.labels)

        if self.data.shape[0] != len(self.labels):
            raise ValueError("Number of images and labels do not match")

    def __len__(self) -> int:
        '''
        Returns the number of samples in the dataset.
        '''
        ## TODO implement
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple:
        '''
        Returns the idx-th sample in the dataset, which is a tuple,
        consisting of the image and labels.
        Applies transforms if not None.
        Raises IndexError if the index is out of bounds.
        '''
        ## TODO implement
        if idx >= len(self.data):
            raise IndexError("Index out of bounds")

        img, label = self.data[idx], self.labels[idx]
        if self.transform:
            img = self.transform(img)

        return img, label

    def num_classes(self) -> int:
        '''
        Returns the number of classes.
        '''
        ## TODO implement
        return len(self.classes)
