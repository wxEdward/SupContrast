from typing import Any, cast, Dict, List, Optional, Union

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim

from torch import sigmoid
from torch_geometric.nn import SAGEConv
from torch.utils.data import DataLoader 
from matplotlib.image import imsave

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
        self.batch_norm3 = nn.BatchNorm2d(out_channels*self.expansion)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.batch_norm1(self.conv1(x)))

        x = self.relu(self.batch_norm2(self.conv2(x)))

        x = self.conv3(x)
        x = self.batch_norm3(x)

        #downsample if needed
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        #add identity
        x+=identity
        x=self.relu(x)

        return x

class Block(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Block, self).__init__()


        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
      identity = x.clone()

      x = self.relu(self.batch_norm2(self.conv1(x)))
      x = self.batch_norm2(self.conv2(x))

      if self.i_downsample is not None:
          identity = self.i_downsample(identity)
      print(x.shape)
      print(identity.shape)
      x += identity
      x = self.relu(x)
      return x




class ResNet(nn.Module):
    def __init__(self, ResBlock, layer_list, num_classes, num_channels=3):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size = 3, stride=2, padding=1)

        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=64)
        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=128, stride=2)
        self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=256, stride=2)
        self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*ResBlock.expansion, num_classes)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        x = self.logsoftmax(x)
        return x

    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        ii_downsample = None
        layers = []

        if stride != 1 or self.in_channels != planes*ResBlock.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes*ResBlock.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes*ResBlock.expansion)
            )

        layers.append(ResBlock(self.in_channels, planes, i_downsample=ii_downsample, stride=stride))
        self.in_channels = planes*ResBlock.expansion

        for i in range(blocks-1):
            layers.append(ResBlock(self.in_channels, planes))

        return nn.Sequential(*layers)


def ResNet50(num_classes=10, channels=1):
    return ResNet(Bottleneck, [3,4,6,3], num_classes, channels)


def load_checkpoint(device, checkpoint_path):
    torch.manual_seed(0)
    checkpoint = torch.load(checkpoint_path)
    model = ResNet50()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    epoch_loss = checkpoint['epoch_loss']
    return model, optimizer, epoch, epoch_loss

def load_model(device, model_path):
    torch.manual_seed(0)
    model = ResNet50()
    model = torch.load(model_path)
    print("Loaded the parameters for the model from %s"%model_path)
    model.to(device)
    return model

def new_model(device, cfg="A"):
    torch.manual_seed(0)
    model = ResNet50()
    model.to(device)
    return model

def load_images(device):
    import scipy.io
    #train_data = scipy.io.loadmat('./binversion/train.mat')
    #test_data = scipy.io.loadmat('./binversion/test.mat')
    
    train_data = scipy.io.loadmat('/data/tian/MSTAR/dataset88/train88.mat')
    test_data = scipy.io.loadmat('/data/tian/MSTAR/dataset88/test88.mat')

    X_train, y_train = np.array(train_data['train_data']), np.array(train_data['train_label'])
    X_test, y_test =  np.array(test_data['test_data']), np.array(test_data['test_label'])

    y_train = y_train.squeeze()
    y_test = y_test.squeeze()

    X_train_image = X_train
    X_test_image = X_test

    X_train_image = (X_train_image - X_train_image.min(axis=(1,2)).reshape([-1,1,1]))/X_train_image.ptp(axis=(1,2)).reshape([-1,1,1])
    print(np.min(X_train_image), np.max(X_train_image))
    X_train_image = X_train_image[:, np.newaxis, :, :]
    X_train_image = torch.from_numpy(X_train_image).to(device)
    X_train_image = X_train_image.type(torch.float32)
    
    X_test_image = (X_test_image - X_test_image.min(axis=(1,2)).reshape([-1,1,1]))/X_test_image.ptp(axis=(1,2)).reshape([-1,1,1])
    print(np.min(X_test_image), np.max(X_test_image))
    X_test_image = X_test_image[:, np.newaxis, :, :]
    X_test_image = torch.from_numpy(X_test_image).to(device)
    X_test_image = X_test_image.type(torch.float32)
   
    if y_train.min().item == 1 and y_train.max().item() == 10:
        train_label = torch.from_numpy(y_train).to(device) - 1
    else:
        train_label = torch.from_numpy(y_train).to(device)
    train_label = train_label.type(torch.int64)
    
    if y_test.min().item == 1 and y_test.max().item() == 10:
        test_label = torch.from_numpy(y_test).to(device) - 1
    else:
        test_label = torch.from_numpy(y_test).to(device)
    test_label = test_label.type(torch.int64)

    #y_attack_target = np.ones(y_test.shape)
    #y_attack_target = np.where(y_test != y_attack_target, y_attack_target, y_attack_target+1)
    #test_attack_target = torch.from_numpy(y_attack_target).to(device) - 1
    #test_attack_target = test_attack_target.type(torch.int64)
    
    print(X_train_image.size(), train_label.size())

    train_data = [[X_train_image[i], train_label[i]] for i in range(train_label.size()[0])]
    test_data = [[X_test_image[i], test_label[i]] for i in range(test_label.size()[0])]

    train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False)

    return train_dataloader, test_dataloader

    #return X_train_image, X_test_image, train_label, test_label #, test_attack_target

if __name__ == '__main__':
    device = 'cpu'
    train_dataloader, test_dataloader = load_images(device)
   
    train_features, train_labels = next(iter(train_dataloader))
    print(train_features.size(), train_labels.size())
    
    #train_features = next(iter(train_dataloader))
    #print(train_features.size())

    img = train_features[0].squeeze()
    label = train_labels[0]
    
    print(img.size(), label.size())

    imsave('example.png', img, cmap='gray')


