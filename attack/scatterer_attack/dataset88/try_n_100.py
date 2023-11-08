import torch
import scipy.io
import numpy as np
from matplotlib.image import imsave


def notation_overlay(test_notation, batch):
    overlays = torch.zeros([batch, 88, 88])
    for i in range(batch):
        notation = test_notation[str(i)]
        for coor in notation:
            overlays[i][coor[0], coor[1]] = 1
    return overlays


data = scipy.io.loadmat('n_test_100_obj.mat')

overlay = notation_overlay(data, 100).numpy()


overlay = overlay.reshape([10, 10, 88, 88])

out = np.zeros([10*88, 10*88])

for i in range(10):
    for j in range(10):
        tl = i*88, j*88
        out[tl[0]:tl[0]+88, tl[1]:tl[1]+88] = overlay[i,j,:,:]


imsave('overlay_100.png', out)

print(overlay.min(), overlay.max())


