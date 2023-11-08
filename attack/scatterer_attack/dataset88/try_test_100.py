import torch
import scipy.io
import numpy as np
from matplotlib.image import imsave


imgs = np.load('X_test_100.npy')

assert(imgs.shape == (100, 88, 88))

imgs = imgs.reshape([10, 10, 88, 88])

out = np.zeros([10*88, 10*88])

for i in range(10):
    for j in range(10):
        tl = i*88, j*88
        out[tl[0]:tl[0]+88, tl[1]:tl[1]+88] = imgs[i,j,:,:]

imsave('test_100.png', out)



