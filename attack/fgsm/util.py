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

def new_model(device):
    torch.manual_seed(0)
    model = AConvNet()
    model.to(device)
    return model

def load_images(device):
    import scipy.io
    #train_data = scipy.io.loadmat('./binversion/train.mat')
    #test_data = scipy.io.loadmat('./binversion/test.mat')
    
    train_data = scipy.io.loadmat('/data/tian/MSTAR/mat/train88.mat')
    test_data = scipy.io.loadmat('/data/tian/MSTAR/mat/test88.mat')
    
    #train_data = scipy.io.loadmat('/data/tian/MSTAR/dataset88/train88.mat')
    #test_data = scipy.io.loadmat('/data/tian/MSTAR/dataset88/test88.mat')

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


