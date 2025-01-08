# utils.py

import torch, time
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import seaborn as sns

class CT_KidneyDataset(Dataset):
    def __init__(self, data_file, root_dir, num_classes=4, file_format=".jpg", transform=None):
        """
        Args:
            csv_file (str or pd.DataFrame): Path to the csv file with image paths and labels,
                                            or a pandas DataFrame containing the data.
            root_dir (str): Directory containing all images.
            num_classes (int): Number of classes for one-hot encoding.
            file_format (str): File format for the images (default is ".jpg").
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        if isinstance(data_file, str):
            self.data_frame = pd.read_csv(data_file)  # Load from CSV file
        elif isinstance(data_file, pd.DataFrame):
            self.data_frame = data_file  # Use the provided DataFrame directly
        else:
            raise ValueError("csv_file should be a file path (str) or a pandas DataFrame.")
        
        self.root_dir = root_dir
        self.file_format = file_format
        self.transform = transform
        self.num_classes = num_classes

        # Extract file paths and labels
        self.image_paths = (
            self.root_dir 
            + self.data_frame["Class"].astype(str).values 
            + "/" 
            + self.data_frame["image_id"].astype(str).values 
            + self.file_format
        )
        self.labels = self.data_frame['target'].values  # Extract labels

    def __len__(self):
        """Returns the total number of images in the dataset."""
        return len(self.data_frame)

    def __getitem__(self, idx):
        """Loads and returns an image and its one-hot encoded label at the given index."""
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
    
class baseline_CNN_v2(nn.Module):
    def __init__(self, num_classes=4):
        super(baseline_CNN_v2, self).__init__()

        # Convolutional layers with batch normalization
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)  # Layer 1
        self.bn1 = nn.BatchNorm2d(32)  # Batch normalization after conv1
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)  # Layer 2
        self.bn2 = nn.BatchNorm2d(32)  # Batch normalization after conv2
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Pooling layer after conv2
        
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)  # Layer 3
        self.bn3 = nn.BatchNorm2d(64)  # Batch normalization after conv3
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)  # Layer 4
        self.bn4 = nn.BatchNorm2d(64)  # Batch normalization after conv4
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Pooling layer after conv4

        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)  # Layer 5
        self.bn5 = nn.BatchNorm2d(128)  # Batch normalization after conv5|
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)  # Layer 6
        self.bn6 = nn.BatchNorm2d(128)  # Batch normalization after conv6|
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # Pooling layer after conv6

        self.conv7 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)  # Layer 7
        self.bn7 = nn.BatchNorm2d(256)  # Batch normalization after conv7|
        self.conv8 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)  # Layer 8
        self.bn8 = nn.BatchNorm2d(256)  # Batch normalization after conv8|
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # Pooling layer after conv8
        
        # Flatten layer
        self.flatten = nn.Flatten()  # Automatically flattens the feature maps
        
        # Fully connected layers with dropout
        self.fc1 = nn.Linear(256 * 14 * 14, 1024)  # Adjust input size based on input image dimensions
        self.dropout1 = nn.Dropout(0.5)  # Dropout after fc1
        
        self.fc2 = nn.Linear(1024, 512)  # Layer 5
        self.dropout2 = nn.Dropout(0.5)  # Dropout after fc2

        self.fc3 = nn.Linear(512, num_classes)  # Layer 6 (output layer)
    
    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(F.relu(self.bn2(self.conv2(x))))
        
        # Block 2
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool2(F.relu(self.bn4(self.conv4(x))))
        
        # Block 3
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool3(F.relu(self.bn6(self.conv6(x))))
        
        # Block 4
        x = F.relu(self.bn7(self.conv7(x)))
        x = self.pool4(F.relu(self.bn8(self.conv8(x))))
        
        # Flatten the feature maps for the fully connected layers
        x = self.flatten(x)
        
        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)  # No activation for the output layer (use softmax in loss function if needed)
        
        return x
    
def train_model(model, train_loader, test_loader, device = "cpu", learning_rate=0.001, weight_decay=1e-4, num_epochs=10):
    # Initialize lists to store metrics for each epoch
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    # Move model to the specified device
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    print("Total Epochs:", num_epochs)

    # Start the overall timer
    overall_start_time = time.time()

    for epoch in range(num_epochs):
        # Start timer for the current epoch
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)  # Raw logits, no softmax
            loss = criterion(outputs, labels.argmax(dim=1))  # Convert one-hot labels to class indices

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Accumulate loss
            running_loss += loss.item() * inputs.size(0)

            # Calculate training accuracy
            _, predicted = torch.max(outputs, 1)  # Get the index of the max log-probability
            correct_train += (predicted == labels.argmax(dim=1)).sum().item()  # Convert one-hot labels
            total_train += labels.size(0)

        # Calculate average training loss and accuracy
        epoch_train_loss = running_loss / total_train
        epoch_train_accuracy = correct_train / total_train
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_accuracy)

        # Evaluation phase
        model.eval()
        running_test_loss = 0.0
        correct_test = 0
        total_test = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)  # Raw logits
                loss = criterion(outputs, labels.argmax(dim=1))  # Convert one-hot labels to class indices

                # Accumulate testing loss
                running_test_loss += loss.item() * inputs.size(0)

                # Calculate validation accuracy
                _, predicted = torch.max(outputs, 1)
                correct_test += (predicted == labels.argmax(dim=1)).sum().item()  # Convert one-hot labels
                total_test += labels.size(0)

        # Calculate average validation loss and accuracy
        epoch_test_loss = running_test_loss / total_test
        epoch_test_accuracy = correct_test / total_test
        test_losses.append(epoch_test_loss)
        test_accuracies.append(epoch_test_accuracy)

        # Stop timer for the current epoch
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        # Format duration for current epoch
        hours = int(epoch_duration // 3600)
        minutes = int((epoch_duration % 3600) // 60)
        seconds = int(epoch_duration % 60)
        epoch_duration_str = f"{hours:02}:{minutes:02}:{seconds:02}"

        print(f"Epoch [{epoch + 1}/{num_epochs}], "
            f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_accuracy * 100:.2f}%, "
            f"Valid Loss: {epoch_test_loss:.4f}, Valid Acc: {epoch_test_accuracy * 100:.2f}% "
            f"(Duration: {epoch_duration_str})")

    # Stop the overall timer
    overall_end_time = time.time()
    total_duration = overall_end_time - overall_start_time

    # Format total training duration
    hours = int(total_duration // 3600)
    minutes = int((total_duration % 3600) // 60)
    seconds = int(total_duration % 60)
    total_duration_str = f"{hours:02}:{minutes:02}:{seconds:02}"

    print("Model Training Completed!")
    print(f"[Total Training Time] {total_duration_str}")

    results = {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'train_accuracies': train_accuracies,
        'test_accuracies': test_accuracies,
    }

    return results

def plot_training_results(results):
    # Extract metrics from the results dictionary
    train_losses = results['train_losses']
    test_losses = results['test_losses']
    train_accuracies = results['train_accuracies']
    test_accuracies = results['test_accuracies']
    
    epochs = range(1, len(train_losses) + 1)

    # Plot training vs testing losses
    plt.figure(figsize=(12, 5))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, test_losses, label='Testing Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training vs. Testing Loss')
    plt.legend()

    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(epochs, [acc * 100 for acc in train_accuracies], label='Training Accuracy')
    plt.plot(epochs, [acc * 100 for acc in test_accuracies], label='Testing Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Training vs. Testing Accuracy')
    plt.legend()

    # Show the plots
    plt.tight_layout()
    plt.show()

def predicted_label(model, image_path, transform=None, labels=None, device='cpu'):
    """
    Predict the label of an image using a CNN model.

    Parameters:
        model (torch.nn.Module): The trained CNN model.
        image_path (str): The path to the image to predict.
        transform (torchvision.transforms.Compose, optional): Custom transformation to apply to the image.
        labels (list of str, optional): List of class labels to map the output index to class names.
        device (str, optional): The device to use for computation ('cpu' or 'cuda').

    Returns:
        str or int: The predicted class label (if `labels` is provided) or the predicted class index.
    """
    
    # Set the model to evaluation mode
    model.eval()
    
    # Default transform if none is provided
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to the input size expected by the model
            transforms.ToTensor(),          # Convert to tensor
            transforms.Normalize(          # Normalize using the mean and std used during training
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            ),
        ])
    
    # Load and preprocess the image
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    
    # Move the image tensor to the same device as the model
    image = image.to(device)
    
    # Forward pass through the model
    with torch.no_grad():  # Disable gradient computation for inference
        output = model(image)
    
    # Get the predicted class (index of the max value in the output)
    _, predicted_idx = torch.max(output, 1)
    
    # If labels are provided, return the label name instead of index
    if labels:
        predicted_label = labels[predicted_idx.item()]
        return predicted_label
    else:
        return predicted_idx.item()  # Convert the tensor to a Python number

def plot_confusion_matrix_and_roc_auc(model, dataloader, labels, model_name, device='cpu', normalize=False):
    """
    Computes and plots the confusion matrix and ROC AUC curve for a model on a given dataset (dataloader).

    Parameters:
        model (torch.nn.Module): The trained model.
        dataloader (torch.utils.data.DataLoader): The dataloader containing the test data.
        labels (list): List of class labels.
        model_name (str): Name of the model to display in the titles.
        device (str): The device to use ('cpu' or 'cuda').
        normalize (bool): If True, normalize the confusion matrix by dividing each row by the sum of that row.
    """
    model.eval()  # Set the model to evaluation mode
    y_true = []
    y_pred = []
    y_probs = []  # Store probabilities for ROC AUC

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            # For multiclass classification, use argmax to get the predicted class
            _, predicted = torch.max(outputs, 1)  # Get the predicted class label (argmax)
            y_probs.extend(outputs.softmax(dim=1).cpu().numpy())  # Get probabilities for ROC AUC

            # If targets are one-hot encoded, convert them to class indices using argmax
            targets = torch.argmax(targets, 1)  # Convert one-hot to class labels
            
            y_true.extend(targets.cpu().numpy())  # True labels
            y_pred.extend(predicted.cpu().numpy())  # Predicted labels
    
    # Convert both y_true and y_pred to 1D arrays (flatten them)
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(labels)))
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Normalize by row

    # Plot the confusion matrix and ROC AUC curve
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Subplot 1: Confusion Matrix
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap='Blues', 
                xticklabels=labels, yticklabels=labels, cbar=False, ax=axes[0], annot_kws={"size": 16})
    axes[0].set_title(f'Confusion Matrix ({model_name})', fontsize=16)
    axes[0].set_xlabel('Predicted Labels', fontsize=14)
    axes[0].set_ylabel('True Labels', fontsize=14)
    
    # Subplot 2: ROC AUC Curve
    y_probs = np.array(y_probs)
    n_classes = len(labels)
    fpr = {}
    tpr = {}
    roc_auc = {}
    
    # One-vs-all ROC curves for each class
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true == i, y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute macro-average ROC curve using one-vs-rest binary format
    y_true_binary = label_binarize(y_true, classes=np.arange(n_classes))
    fpr["macro"], tpr["macro"], _ = roc_curve(y_true_binary.ravel(), y_probs.ravel())
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    # Plot ROC AUC curves
    axes[1].plot(fpr["macro"], tpr["macro"], label=f'Macro-average ROC (AUC = {roc_auc["macro"]:.2f})', linestyle='--', color='blue')
    for i in range(n_classes):
        axes[1].plot(fpr[i], tpr[i], label=f'Class {labels[i]} (AUC = {roc_auc[i]:.2f})')
    
    axes[1].plot([0, 1], [0, 1], 'k--', lw=1)  # Diagonal line
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_title(f'ROC AUC Curve ({model_name})', fontsize=16)
    axes[1].set_xlabel('False Positive Rate', fontsize=14)
    axes[1].set_ylabel('True Positive Rate', fontsize=14)
    axes[1].legend(loc='lower right')
    
    plt.tight_layout()
    plt.show()
    
def save_model(model, filepath):
    """
    Save the PyTorch model to the specified file path.

    Args:
    - model (torch.nn.Module): The model to save.
    - filepath (str): The file path where the model will be saved.
    """
    # Save the model's state dictionary (weights)
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")

def load_model(model_class, filepath, device="cpu", model_args=None):
    """
    Load a PyTorch model from the specified file path.

    Args:
    - model_class (torch.nn.Module): The class of the model to load.
    - filepath (str): The file path from which to load the model.
    - device (str): The device on which to load the model ("cpu" or "cuda").
    - model_args (dict, optional): Arguments to pass to the model class constructor.

    Returns:
    - model (torch.nn.Module): The loaded model with the saved weights.
    """
    if model_args is None:
        model_args = dict()  # Safely create a new dictionary here

    # Initialize the model with additional arguments
    model = model_class(**model_args)

    # Load the model weights into the initialized model
    model.load_state_dict(torch.load(filepath, map_location=device))
    model.to(device)  # Move the model to the desired device
    model.eval()  # Set the model to evaluation mode
    
    print(f"Model loaded from {filepath}")
    
    return model

def display_tensor_image(tensor_image):
    """
    Display an image from a PyTorch tensor using matplotlib.

    Args:
        tensor_image (torch.Tensor): The image in a PyTorch tensor format (C, H, W).
    """
    # Ensure the tensor is in the right format (C, H, W -> H, W, C)
    image_np = tensor_image.permute(1, 2, 0).numpy()  # Convert from CHW to HWC
    
    # If the tensor values are between [0, 1], convert to [0, 255]
    if image_np.max() <= 1.0:
        image_np = (image_np * 255).astype(np.uint8)
    
    # Display the image using matplotlib
    plt.imshow(image_np)
    plt.axis('off')  # Hide the axes
    plt.show()