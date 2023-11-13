import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import sigmoid
from torch_geometric.nn import SAGEConv
from torch.utils.data import DataLoader 
from matplotlib.image import imsave

class AConvNet(nn.Module):
    def __init__(self, num_classes: int = 10, dropout: float = 0.5) -> None:
        super().__init__()
        #_log_api_usage_once(self)
        self.classifier = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=6),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Conv2d(128, 10, kernel_size=3, stride=3)
        )

        self.head = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128)
        )

        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.classifier(x)
        x = torch.flatten(x, 1)
        x = self.logsoftmax(x)
        return x

def load_checkpoint(device, checkpoint_path):
    torch.manual_seed(0)
    checkpoint = torch.load(checkpoint_path)
    model = AConvNet()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    epoch_loss = checkpoint['epoch_loss']
    return model, optimizer, epoch, epoch_loss

def load_model(device, model_path):
    torch.manual_seed(0)
    model = AConvNet()
    #model.load_state_dict(torch.load(model_path))
    model = torch.load(model_path)
    print("Loaded the parameters for the model from %s"%model_path)
    model.to(device)
    return model

def load_aconvnet(device, path):
    model, _, _, _ = load_checkpoint(device, path)
    model.eval()
    return model
    #return load_model(device, path)


