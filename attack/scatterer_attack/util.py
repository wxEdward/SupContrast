import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import sigmoid
from torch_geometric.nn import SAGEConv
from matplotlib.image import imsave

def set_all_seeds(seed):
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True

def setup_model(device, model_path):
    torch.manual_seed(0)
    model = Net()
    model.load_state_dict(torch.load(model_path))
    print("Loaded the parameters for the model from %s" % model_path)
    model.to(device)
    return model


def load_images(device):
    import scipy.io
    train_data = scipy.io.loadmat('./binversion/train.mat')
    test_data = scipy.io.loadmat('./binversion/test.mat')

    X_train, y_train = np.array(train_data['train_data']), np.array(train_data['train_label'])
    X_test, y_test =  np.array(test_data['test_data']), np.array(test_data['test_label'])

    X_train_image = X_train
    X_test_image = X_test

    X_train_image = (X_train_image - X_train_image.min(axis=(1,2)).reshape([-1,1,1]))/X_train_image.ptp(axis=(1,2)).reshape([-1,1,1])
    print(np.min(X_train_image), np.max(X_train_image))
    X_train_image = torch.from_numpy(X_train_image).to(device)
    X_train_image = X_train_image.type(torch.float32)
    
    X_test_image = (X_test_image - X_test_image.min(axis=(1,2)).reshape([-1,1,1]))/X_test_image.ptp(axis=(1,2)).reshape([-1,1,1])
    print(np.min(X_test_image), np.max(X_test_image))
    X_test_image = torch.from_numpy(X_test_image).to(device)
    X_test_image = X_test_image.type(torch.float32)
    
    train_label = torch.from_numpy(y_train).to(device) - 1
    train_label = train_label.type(torch.int64)
    test_label = torch.from_numpy(y_test).to(device) - 1
    test_label = test_label.type(torch.int64)

    y_attack_target = np.ones(y_test.shape)
    y_attack_target = np.where(y_test != y_attack_target, y_attack_target, y_attack_target+1)
    test_attack_target = torch.from_numpy(y_attack_target).to(device) - 1
    test_attack_target = test_attack_target.type(torch.int64)
    
    print(X_test_image.shape, test_label.shape, test_attack_target.shape)

    return X_train_image, X_test_image, train_label, test_label, test_attack_target


def load_test_images(device):
    from scipy.io import loadmat

    X_test = np.load('./dataset88/X_test_100.npy')
    y_test = np.load('./dataset88/y_test_100.npy')
    #n_test = np.load('./dataset88/n_test_100.npy')
    #n_test = loadmat('./dataset88/n_test_100.mat')
    n_test = loadmat('./dataset88/n_test_100_obj.mat')
    #n_test = loadmat('./dataset88/n_test_100_nz.mat')

    mag_test = np.load('./dataset88/mag_test_100.npy')

    print(np.min(X_test), np.max(X_test))

    X_test = torch.from_numpy(X_test).to(device)
    X_test = X_test.type(torch.float32)
    
    y_test = torch.from_numpy(y_test).to(device)
    y_test = y_test.type(torch.int64)
   
    #n_test = torch.from_numpy(n_test).to(device)
    #n_test = n_test.type(torch.int8)

    print(X_test.size(), y_test.size(), len(n_test.keys()), mag_test.shape)

    return X_test, y_test, n_test, mag_test
