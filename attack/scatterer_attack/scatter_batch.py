# batched version of scatter model
import torch
import numpy as np
from torch import pi, exp, sinc, sin, cos, tan, arctan, sqrt
from scipy.signal.windows import taylor
from torch.fft import ifft2 as ifft2
from matplotlib import pyplot as plt
import torch.nn.functional as F
import torch_geometric

from matplotlib.image import imsave
from time import time

from util import load_images, setup_model


#torch.cuda.is_available()
#device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
#device = "cpu"

# light speed
c = 299792458

# SAR imaging parameters
# [fc, B, phi_m, m, n, m_star, n_star, W(.)]
fc = 9.6
B = 0.59
phi_m = 0.051
m = 85
n = 85
m_star = 128
n_star = 128
Wx = taylor(m, sll=35).reshape([m,1])
Wy = taylor(n, sll=35).reshape([1,n])
#W = np.tile(Wx, (1,n)) * np.tile(Wy, (m,1))
#W = torch.from_numpy(W).to(device)
W = None

eta_x = (m-1)/(m_star-1)
eta_y = (n-1)/(n_star-1)
px = c*eta_x/(2*B)
py = c*eta_y/(4*fc*np.sin(phi_m/2))

# fx axis & fy axis
max_fy = fc*np.sin(phi_m/2)
fx = np.arange(fc-B/2, fc+B/2, B/n).reshape([1,n,1])
fy = np.arange(-max_fy, max_fy, 2*max_fy/m).reshape([1,1,m])


'''
                 0   1   2    3      4      5     6
 parameters are: A, xp, yp, alpha, gammap, Lp, phi_barp
 @param theta: torch tensor [batch, 7] for a batch of 7 parameters
'''
def E_i(theta, batch, device):
    #print(theta.size())
    x = np.tile(fx, (batch, 1, m))
    y = np.tile(fy, (batch, n, 1))
    x = torch.from_numpy(x).to(device).permute(1,2,0)
    y = torch.from_numpy(y).to(device).permute(1,2,0)
    #print(x.size(), y.size())
    t0 = 1j*sqrt(x**2+y**2)/fc
    #print(t0.size(), theta[:,3].size())
    t1 = t0 ** theta[:,3]
    #t1 = (1j*sqrt(x**2+y**2)/fc)**theta[:,3]
    t2 = exp(-y*theta[:,4]/fc)
    t3 = exp(-1j*4*pi/c*(px*theta[:,1]*x+py*theta[:,2]*y))
    t4 = sinc(pi*sqrt(x**2+y**2)/(2*np.sin(phi_m/2)*fc)*theta[:,5]*eta_y*
                sin(arctan(y/x)-theta[:,6]*phi_m/2))
    e = theta[:,0] * t1 * t2 * t3 * t4
    e = e.permute(2,0,1)
    assert(e.size() == (batch, m, n))
    return e

'''
 @param params: torch tensor [batch, N, 7] for a batch of N scatters and 7 parameters each
'''
def E(param, batch, device):
    #print(param.size())
    e = np.zeros([batch, m, n]).astype(np.complex128)
    e = torch.from_numpy(e).to(device)
    for i in range(param.size()[1]):
        e += E_i(param[:,i,:], batch, device)
    return e

'''
 @return: image [batch, m_star, n_star]
'''
def getImage(e, device):
    global W
    if W is None:
        W = np.tile(Wx, (1,n)) * np.tile(Wy, (m,1))
        W = torch.from_numpy(W).to(device)

    # Step 1: multiply e(fx, fy) by a window function W(fx, fy)
    e = e * W
    
    # Step 2: zero-pad it from [m, n] to [m_star, n_star]
    e = F.pad(e, [0, n_star-n, 0, m_star-m, 0, 0], "constant", 0)
    
    # Step 3: apply 2D-IDFT
    I = ifft2(e)

    # Step 4: get amplitude of complex
    img = torch.abs(I)
    return img[:,:88,:88]

if __name__ == '__main__':
    param = torch.randn(2, 3, 7)
    img = getImage(E(param, 2, 'cpu'), 'cpu')
    print(img.size())
