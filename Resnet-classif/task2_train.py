import pandas as pd
import matplotlib.pyplot as plt
import os
import random
from PIL import Image

from sklearn.model_selection import KFold
from tqdm import tqdm

import torch
import numpy as np
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torch.optim import Adam
import torch.nn.functional as F


df = pd.read_csv("data_aait/Task2/task2/train_data/annotations.csv")

class ImageDataset(Dataset):
  def __init__ (self, root_dir, csv_file = None, transform=None):
    """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
    """
    self.root_dir = root_dir
    self.transform = transform

    # Load annotations for labeled images
    if csv_file:
      if os.path.exists(csv_file):
          self.images = pd.read_csv(csv_file)

  def __len__(self):
      return len(self.images)

  def __getitem__(self, idx):
      img_info = self.images.iloc[idx]
      img_name = img_info['renamed_path']

      img_path = os.path.join(self.root_dir, img_name)
      label = img_info['label_idx']

      image = Image.open(img_path).convert('RGB')

      if self.transform:
          image = self.transform(image)
      return image, label


def augment():
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(p=0.5),  # Random flip left-right
        transforms.ColorJitter(brightness=0.5),  # Random brightness
        transforms.ColorJitter(contrast=0.5),     # Random contrast
        transforms.ToTensor()

    ])
    return transform

def augment2():
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomVerticalFlip(p=0.5),    # Random flip up-down
        transforms.ColorJitter(hue=0.1),         # Random hue
        transforms.ColorJitter(saturation=0.5),   # Random saturation
        transforms.ToTensor()
    ])
    return transform

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Create the dataset
dataset = ImageDataset(
    root_dir='data_aait/Task2/',
    csv_file='data_aait/Task2/task2/train_data/annotations.csv',
    transform = augment()
)

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

num_classes = 100


class CustomResNet(nn.Module):
    def __init__(self, resnet_base, additional_layers):
        super(CustomResNet, self).__init__()
        self.resnet_base = resnet_base
        self.additional_layers = additional_layers

    def forward(self, x):
        x = self.resnet_base(x)
        x = self.additional_layers(x)
        return x

def plot_and_save_metrics(train_loss, train_accuracy, val_loss, val_accuracy, fold, save_path):
    """Plot and save the training and validation metrics."""
    epochs = range(1, len(train_loss) + 1)
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracy, label='Train Accuracy')
    plt.plot(epochs, val_accuracy, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.suptitle(f'Fold {fold + 1}')
    plt.savefig(os.path.join(save_path, f't2_training_metrics_fold_{fold+1}.png'))
    plt.close()



num_folds = 5
# Define the k-fold cross-validation
kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)


for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
    # Creating data loaders for each fold
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
    val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, sampler=train_subsampler)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=32, sampler=val_subsampler)
  
    print(f"Fold {fold + 1}/{num_folds}")  
    
    resnet = models.resnet50(pretrained=True)
    num_ftrs = resnet.fc.in_features
    resnet = nn.Sequential(*list(resnet.children())[:-1])
    resnet = resnet.to(device)

    additional_layers = nn.Sequential(
    nn.Flatten(),
    nn.BatchNorm1d(num_ftrs),
    nn.Linear(num_ftrs, 128),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.BatchNorm1d(128),
    nn.Linear(128, num_classes),
    nn.Softmax(dim=1)   
    )

    # freeze all layers
    for param in resnet.parameters():
            param.requires_grad = False

    # Create the full model
    model = CustomResNet(resnet, additional_layers).to(device)
    criterion = nn.CrossEntropyLoss()
    
    trainable_parameters = filter(lambda p: p.requires_grad, model.parameters())

    optimizer = Adam(trainable_parameters, lr=1e-4)

    # Initialize metrics

    val_loss_all = []
    val_accuracy_all = []

    train_loss_all = []
    train_accuracy_all = []

    num_epochs = 20
    print("First training (Resnet frozen)")
    for epoch in range(num_epochs):
            # Training Phase
            model.train()  # Set the model to training mode
            running_loss = 0.0
            correct = 0
            total = 0
            # Wrap the training loader with tqdm for a progress bar
            train_loader_with_progress = tqdm(train_loader, desc=f'Fold {fold + 1}, Epoch {epoch + 1}')
            for i, data in enumerate(train_loader_with_progress):
                inputs, labels = data[0].to(device), data[1].to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Update progress bar description with current loss
                train_loader_with_progress.set_postfix(loss=(running_loss / (i + 1)))

            train_accuracy = 100 * correct / total
            train_accuracy_all.append(train_accuracy)
            train_loss_all.append(running_loss / len(train_loader_with_progress))

            # Print statistics
            print(f'Fold {fold + 1}, Epoch {epoch + 1}, Loss: {running_loss / len(train_loader_with_progress)}, Accuracy: {train_accuracy}')
            # Validation Phase after training is complete
            model.eval()
            val_running_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for data in val_loader:
                    inputs, labels = data[0].to(device), data[1].to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_running_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            val_loss = val_running_loss / len(val_loader)
            val_accuracy = 100 * correct / total

            val_loss_all.append(val_loss)
            val_accuracy_all.append(val_accuracy)
            print(f'Fold {fold + 1}, Validation Loss: {val_loss}, Accuracy: {val_accuracy}%')

    # Unfreeze some layers
    layer_count = 0
    ct = 0
    for child in model.children():
        ct += 1
        if ct > 9:
            for param in child.parameters():
                param.requires_grad = True
    
    trainable_parameters = filter(lambda p: p.requires_grad, model.parameters())

    optimizer = Adam(trainable_parameters, lr=1e-4)

    num_epochs = 50
    print("Train again ufreezing layers")
    for epoch in range(num_epochs):
        # Training Phase
        model.train()  # Set the model to training mode
        running_loss = 0.0
        correct = 0
        total = 0
        # Wrap the training loader with tqdm for a progress bar
        train_loader_with_progress = tqdm(train_loader, desc=f'Fold {fold + 1}, Epoch {epoch + 1}')
        for i, data in enumerate(train_loader_with_progress):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Update progress bar description with current loss
            train_loader_with_progress.set_postfix(loss=(running_loss / (i + 1)))

        train_accuracy = 100 * correct / total
        train_accuracy_all.append(train_accuracy)
        train_loss_all.append(running_loss / len(train_loader_with_progress))

        # Print statistics
        print(f'Fold {fold + 1}, Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}, Accuracy: {train_accuracy}')

        # Validation Phase after training is complete
        model.eval()
        val_running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for data in val_loader:
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss = val_running_loss / len(val_loader)
        val_accuracy = 100 * correct / total

        val_loss_all.append(val_loss)
        val_accuracy_all.append(val_accuracy)
        print(f'Fold {fold + 1}, Validation Loss: {val_loss}, Accuracy: {val_accuracy}%')

    plot_and_save_metrics(train_loss_all, train_accuracy_all, val_loss_all, val_accuracy_all, fold, 'results_aait/frozen_unfrozen_t2')

    print('Finished Training')

    model_save_path = 'models_aait/'

    # Save the model after training and validation
    torch.save(model.state_dict(), os.path.join(model_save_path, f'model_frozen_unfrozen_{fold}_t2.pth'))
