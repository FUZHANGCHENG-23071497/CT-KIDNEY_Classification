# utils.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import numpy as np

class CT_KidneyDataset(Dataset):
    def __init__(self, csv_file, root_dir, num_classes=4, file_format=".jpg", transform=None):
        """
        Args:
            csv_file (str): Path to the csv file with image paths and labels.
            root_dir (str): Directory containing all images.
            num_classes (int, optional): Number of classes (default is 4).
            file_format (str, optional): File format for images (default is ".jpg").
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.file_format = file_format
        self.transform = transform
        self.num_classes = num_classes

        """Extract file paths and its labels"""
        self.image_paths = self.root_dir + self.data_frame["Class"].values + "/" + self.data_frame["image_id"].values + self.file_format
        self.labels = self.data_frame['target'].values  # Extract labels

    def __len__(self):
        """Returns the total number of images in the dataset"""
        return len(self.data_frame)

    def __getitem__(self, idx):
        """Loads and returns an image and its one-hot encoded label at the given index"""

        # Get the image path and label
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load the image
        image = Image.open(img_path).convert("RGB")  # Convert to RGB

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)
    
        # Convert label to one-hot encoding if num_classes > 1
        one_hot_label = torch.zeros(self.num_classes)
        one_hot_label[label] = 1
        
        return image, one_hot_label


class baseline_CNN(nn.Module):
    def __init__(self, num_classes=4):
        super(baseline_CNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)  # Layer 1
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1) # Layer 2
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Pooling layer (applied after conv layers)
        
        # Additional convolutional layers
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1) # Layer 3
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1) # Layer 4
        
        # Flatten layer
        self.flatten = nn.Flatten()  # Automatically flattens the feature maps
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 14 * 14, 512)  # Adjust input size based on input image dimensions
        self.fc2 = nn.Linear(512, 128)  # Layer 5
        self.fc3 = nn.Linear(128, num_classes)  # Layer 6 (output layer)
    
    def forward(self, x):
        # Block 1: Conv1 -> ReLU -> Pool
        x = self.pool(F.relu(self.conv1(x)))
        
        # Block 2: Conv2 -> ReLU -> Pool
        x = self.pool(F.relu(self.conv2(x)))
        
        # Block 3: Conv3 -> ReLU -> Pool
        x = self.pool(F.relu(self.conv3(x)))
        
        # Block 4: Conv4 -> ReLU -> Pool
        x = self.pool(F.relu(self.conv4(x)))
        
        # Flatten the feature maps for the fully connected layers
        x = self.flatten(x)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # No activation for the output layer (use softmax in loss function if needed)
        
        return x
