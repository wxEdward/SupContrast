import torch
import scipy.io
import numpy as np
from matplotlib.image import imsave


imgs = np.load('X_test_100.npy')

assert(imgs.shape == (100, 88, 88))

img = imgs[5]

imsave('test_id6.png', img)



