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
import csv

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CustomResNet(nn.Module):
    def __init__(self, resnet_base, additional_layers):
        super(CustomResNet, self).__init__()
        self.resnet_base = resnet_base
        self.additional_layers = additional_layers

    def forward(self, x):
        x = self.resnet_base(x)
        x = self.additional_layers(x)
        return x


resnet = models.resnet50(pretrained=True)
num_ftrs = resnet.fc.in_features
resnet = nn.Sequential(*list(resnet.children())[:-1])
resnet = resnet.to(device)


num_classes = 100


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

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])


class ImageDataset_valid(Dataset):
  def __init__ (self, root_dir, transform=None):
    """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
    """
    self.root_dir = root_dir
    self.transform = transform

    self.images = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]

  def __len__(self):
      return len(self.images)

  def __getitem__(self, idx):
      img_name = self.images[idx]
      img_path = os.path.join(self.root_dir, img_name)

      image = Image.open(img_path).convert('RGB')

      if self.transform:
          image = self.transform(image)
      return image, img_name

print("Creating Dataloader")
validation_dataset = ImageDataset_valid(root_dir='data_aait/Task2/task2/val_data', transform=transform)
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=32, shuffle=False)

print("Loading model")
model = CustomResNet(resnet, additional_layers)
model.load_state_dict(torch.load('models_aait/model_frozen_unfrozen_0_t2.pth'))
model.to(device)
model.eval()

predictions = []
filenames = []
with torch.no_grad():  # This will disable gradient computation during inference
    for inputs, fname in tqdm(validation_loader, desc="Running Inference"):
        inputs = inputs.to(device)
        outputs = model(inputs)

        predictions.extend(outputs.cpu().numpy())
        filenames.extend(fname)

pred_classes = [p.argmax() for p in predictions]

print("Saving Results")
with open('results_aait/predictions_t2/val.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['sample', 'label'])
    for fname, pred in zip(filenames, pred_classes):
        writer.writerow([fname, pred])






