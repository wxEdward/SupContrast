import numpy as np
import torch
from torch.utils.data import DataLoader 

def load_adv_images(device, adv_path, gt_path):
    import scipy.io

    X_adv = np.load(adv_path)
    y_gt = np.load(gt_path)

    X_adv = X_adv[:, np.newaxis, :, :]
    X_adv = torch.from_numpy(X_adv).to(device)
    X_adv = X_adv.type(torch.float32)
    
    y_gt = torch.from_numpy(y_gt).to(device)# - 1
    y_gt = y_gt.type(torch.int64)

    print(X_adv.size(), y_gt.size())

    test_data = [[X_adv[i], y_gt[i]] for i in range(y_gt.size()[0])]
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False)
    return test_dataloader

