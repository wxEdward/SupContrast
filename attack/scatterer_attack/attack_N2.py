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

from util import set_all_seeds, load_images, load_test_images, setup_model
from scatter_batch import E_i, E, getImage

from gnn import load_gnn
from vgg11 import load_vgg11
from alexnet2 import load_alexnet
from squeezenet import load_squeezenet
from densenet121 import load_densenet121
from mobilenetv2 import load_mobilenetv2
from resnet50 import load_resnet50
from shufflenetv2 import load_shufflenetv2
from aconvnet import load_aconvnet


import argparse

model_names = ["gnn", "vgg11", "alexnet", "squeezenet", "densenet121", "mobilenetv2", "resnet50", "shufflenetv2", "aconvnet"]

model_paths = {"gnn": "./models/gnn-checkpoint-380.pt",
               "vgg11": "./models/vgg11-checkpoint-420.pt",
               "alexnet": "./models/alexnet-checkpoint-150.pt",
               "squeezenet": "./models/squeezenet-checkpoint-50.pt",
               "densenet121": "./models/densenet121-checkpoint-180.pt",
               "mobilenetv2": "./models/mobilenetv2-checkpoint-510.pt",
               "resnet50": "./models/resnet50-checkpoint-110.pt",
               "shufflenetv2": "./models/shufflenetv2-checkpoint-550.pt",
               "aconvnet": "./models/aconvnet-checkpoint-150.pt"}

model_loads = {"gnn": load_gnn,
               "vgg11": load_vgg11,
               "alexnet": load_alexnet,
               "squeezenet": load_squeezenet,
               "densenet121": load_densenet121,
               "mobilenetv2": load_mobilenetv2,
               "resnet50": load_resnet50,
               "shufflenetv2": load_shufflenetv2,
               "aconvnet": load_aconvnet}

parser = argparse.ArgumentParser()
parser.add_argument('--surrogate', help='Surrogate model')
parser.add_argument('--debug', action='store_true', help='Debug mode')
parser.set_defaults(debug=False)
FLAGS = parser.parse_args()

assert(FLAGS.surrogate in model_names)
print("Surrogate model: {}".format(FLAGS.surrogate))

surrogate_path = model_paths[FLAGS.surrogate]
load_surrogate = model_loads[FLAGS.surrogate]

torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "cpu"

'''
 [A, xp, yp, alpha, gammap, Lp, phi_barp]
 @param N: number of scatters
 @param theta_min: min values of parameters
 @param theta_max: max values of parameters
 @example theta_min = [0, 0, 0, -1, 0, 0, -1]
 @example theta_max = [10, 87, 87, 1, 2, 5, 1] # why 87 not 127?
 @return: torch tensor [N, 7] for N scatters and 7 parameters each
 @return: freeze vector [7] binary vector representing frozen parameters

mode
   tau=[1, 1, 1, 0, 1, 0, 0]
0: alpha=1, L=0, phi=0
1: alpha=0.5, L=0, phi=0
2: alpha=0, L=0, phi=0
3: alpha=-1, L=0, phi=0

   tau=[1, 1, 1, 0, 0, 1, 1]
4: alpha=1, gamma=0
5: alpha=0.5, gamma=0
6: alpha=0, gamma=0
7: alpha=-0.5, gamma=0
'''
def initParam(batch, N, theta_min, theta_max, notation, device):
    mode = np.random.randint(0, 8, size=(batch, N))
    param = torch.rand((batch, N, 7), device=device)

    # control initial x and y
    theta_min = torch.from_numpy(theta_min).to(device)
    theta_max = torch.from_numpy(theta_max).to(device)
    
    param = param * (theta_max - theta_min) + theta_min

    coord_id = np.random.choice(np.arange(0, notation.shape[0]), size=(batch, N), replace=True)

    for b in range(batch):
        for i in range(N):
            param[b,i,1] = notation[coord_id[b][i]][0]
            param[b,i,2] = notation[coord_id[b][i]][1]

    tau = torch.empty(batch, N, 7)
    for b in range(batch):
        for i in range(N):
            if mode[b,i] <= 3:
                param[b,i,5] = 0
                param[b,i,6] = 0
                tau[b,i] = torch.tensor([1, 1, 1, 0, 1, 0, 0])
            else:
                param[b,i,4] = 0
                tau[b,i] = torch.tensor([1, 1, 1, 0, 0, 1, 1])

            if mode[b,i] == 0 or mode[b,i] == 4:
                param[b,i,3] = 1
            elif mode[b,i] == 1 or mode[b,i] == 5:
                param[b,i,3] = 0.5
            elif mode[b,i] == 2 or mode[b,i] == 6:
                param[b,i,3] = 0
            elif mode[b,i] == 3:
                param[b,i,3] = -1
            elif mode[b,i] == 7:
                param[b,i,3] = -0.5
    
    tau = tau.to(device)

    param.requires_grad = True
    return param, tau

'''
    params: [batch, N, 7]
    return: [batch] -- average gaussian score for each batch of N scatterers
'''
def gaussian(params, notation, batch, N, device, sigma=0.4):
    assert(params.size() == (batch, N, 7))
    score = torch.zeros([batch, N], device=device)
    for coor in notation:
        score += torch.exp(-0.5 * ((params[:,:,1] - coor[0]) ** 2 + (params[:,:,2] - coor[1]) ** 2) / (sigma ** 2)) / (2*torch.pi)
    #score = torch.fmin(score, 0.2 * sigma)
    score = torch.clamp(score, max=0.2*sigma)
    return torch.mean(score, axis=1)

'''
 @param model: GNN/CNN model
 @param device: device
 @param X_image: image
 @param y_gt: ground truth
 @param edgearray: edges

 @param N: number of scatters
 TODO: @param batch: batch size (TODO; for now batch=1)
 @param theta_min, theta_max: ranges of parameters
 @param vth: confidence threshold
 @param lambd: stepsize changing rate
 @param n_max: max iterations
 @param S0: initial mean stepsize (e.g., [0.05, 0.5, 0.5, 0, 0.01, 0.025, 0.01])
'''
def test(model, device, X_image, y_gt, notation, overlay, mag, batch, N, theta_min, theta_max, vth, lambd, lambd_gaussian, n_max, S0):
    assert(mag.shape == (2, ))
    print(mag[0], mag[1])
    assert(mag[0] < mag[1])
    
    std = (theta_max - theta_min)/200
    std = np.tile(std.reshape([1,-1]), [N, 1])
    std = torch.from_numpy(std).to(device)
    S0 = np.tile(S0.reshape([1, 1,-1]), [batch, N, 1])
    S0 = torch.from_numpy(S0).to(device)

    param, freeze = initParam(batch, N, theta_min, theta_max, notation, device)#.to(device)

    theta_min = np.tile(theta_min.reshape([1,-1]), [N, 1])
    theta_min = torch.from_numpy(theta_min).to(device)
    theta_max = np.tile(theta_max.reshape([1,-1]), [N, 1])
    theta_max = torch.from_numpy(theta_max).to(device)

    #freeze = torch.from_numpy(np.array([1, 1, 1, 0, 1, 0, 1])).to(device)

    iteration = 0

    x_max = X_image.max().item()

    # getImage returns [batch, 88, 88], X_adv will be of the same size
    X_adv = X_image + getNormImage(getImage(E(param, batch, device), device), mag) #* overlay
    X_adv = torch.clamp(X_adv, 0, 1)
    X_adv = X_adv.float()
    X_adv = X_adv[:,None,:,:] # X_adv resized to [batch, 1, 88, 88]
#    print(X_adv.size())
    output = model(X_adv)

#    print(output)        # output is of size [batch, 10]
    v = output[:, y_gt]  # confidence of ground truth.
    v = torch.exp(v)     # v is of size [batch, 1]

    min_v = v.min().item() # get the minimum of v
    min_param = param[v.argmin()]

    #output = output[:, None, :]  # size [batch, 1, 10]
#    print(output.size())
    
    y_gt_batch = y_gt.repeat(batch)
    #loss = F.nll_loss(output, y_gt_batch, reduction='none') + lambd_gaussian * gaussian(param, notation, batch, N, device)
    loss_pred = F.nll_loss(output, y_gt_batch, reduction='none') 
    loss_gaussian = lambd_gaussian * gaussian(param, notation, batch, N, device)
    loss = loss_pred + loss_gaussian
    total_loss = torch.sum(loss)  # prepare to backward for the entire batch
    
    #print(loss)
    #print(loss_pred)
    #print(loss_gaussian)

    #print(param, X_adv, loss)
    while iteration < n_max and v.min().item() > vth:
        iteration += 1
        
        #print(iteration)
        param.requires_grad = True

        # generate stepsize
        step = torch.normal(mean=S0, std=std)#.to(device)

        # backpropagate
        model.zero_grad()
        
        # compute gradient for the entire batch of sets of parameters
        total_loss.backward()

        param_grad = param.grad.data
        sign_param_grad = param_grad.sign()

        # compute and clip new theta
        delta_param = step * sign_param_grad

        param_new = torch.clamp(param + freeze*delta_param, theta_min, theta_max).detach()
        #param_new = param + freeze*delta_param
        param_new = param_new.detach()

        # reduce amplitude if a scatter's max pixel > 1            
        for i in range(N):
            img_batch = getNormImage(getImage(E_i(param_new[:,i,:], batch, device), device), mag) #* overlay
            for b in range(batch):
                if torch.max(img_batch[b]).item() > x_max:
                    param_new[b][i][0] = param_new[b][i][0] / (np.random.uniform() + param_new[b][i][0])

        param_new.requires_grad = False
        param.requires_grad = False
        # new image
        X_adv = X_image + getNormImage(getImage(E(param_new, batch, device), device), mag) #* overlay
        X_adv = torch.clamp(X_adv, 0, 1)
        X_adv = X_adv.float()
        X_adv = X_adv[:,None,:,:]
        output_new = model(X_adv)
        loss_new = F.nll_loss(output_new, y_gt_batch, reduction='none') + lambd_gaussian * gaussian(param_new, notation, batch, N, device)

        #print(param_new, X_adv, loss_new)
        
        '''
        for b in range(batch):
            if loss_new[b] > loss[b] or np.random.uniform() > 0.5:
                param[b] = param_new[b]
                # TODO: update mean stepsize
                if loss_new[b] > loss[b]:
                    S0[b] = lambd * S0[b] + (1-lambd) * torch.abs(delta_param[b])
        '''

        param = param_new
        S0 = lambd * S0 + (1-lambd) * torch.abs(delta_param)

        #loss = loss_new  # TODO: should we move it into the if-statement?
        
        param.requires_grad = True
        img_noise = getNormImage(getImage(E(param, batch, device), device), mag) #* overlay
        X_adv = X_image + img_noise
        X_adv = torch.clamp(X_adv, 0, 1)
        X_adv = X_adv.float()
        X_adv = X_adv[:,None,:,:]
        output = model(X_adv)

        # TODO: maybe wrong
        # Since no all param are updated, we compute the loss again to make sure the loss is aligned with the param
        #loss = F.nll_loss(output, y_gt_batch, reduction='none') + lambd_gaussian * gaussian(param, notation, batch, N, device)
        loss_pred = F.nll_loss(output, y_gt_batch, reduction='none') 
        loss_gaussian = lambd_gaussian * gaussian(param, notation, batch, N, device)
        loss = loss_pred + loss_gaussian
        total_loss = torch.sum(loss)

        v = output[:, y_gt]
        v = torch.exp(v)

        v = v.squeeze()
        mask = torch.where(torch.abs(lambd_gaussian * 0.2 * 0.4 - loss_gaussian) < 1e-4, 0, 1)
        v = v + mask

        if v.min().item() < min_v:
            min_v = v.min().item()
            min_param = param[v.argmin()]

    print(iteration)

    if FLAGS.debug:
        print(min_param)

    return min_param

def notation_overlay(test_notation):
    overlays = torch.zeros([100, 88, 88])
    for i in range(100):
        notation = test_notation[str(i)]
        for coor in notation:
            overlays[i][coor[0], coor[1]] = 1
    return overlays


def filter_param(param, notation, N):
    if param.dim() == 2:
        param = param[None, :, :]
        assert(param.size() == (1, N, 7))
    for batch in range(param.shape[0]):
        for i in range(N):
            xy = param[batch, i, 1:3].cpu().detach().numpy()
            xy = np.round(xy)  # TODO: is round ok? 
            if not xy.tolist() in notation.tolist():
    #        if not (xy == notation).any():
                #param[batch, i, 0] = 0
                param[batch, i, :] = 0
    return param

'''
def filter_param(param, notation, N):
    if param.dim() == 2:
        param = param[None, :, :]
    return param
'''

'''
 img: [batch, m_star, n_star]
 Entire batch is for the same original SAR image
'''
def getNormImage(img, mag):
    min_pixel = mag[0]
    max_pixel = mag[1]
    assert(min_pixel < max_pixel)
    img = (img - min_pixel) / (max_pixel - min_pixel)
    return img

def all():
    batch = 100 #1000
    N = 2
    theta_min = np.array([0, 0, 0, -1, 0, 0, -1])
    theta_max = np.array([10, 87, 87, 1, 2, 5, 1])
    lambd = 0.5
    lambd_gaussian = 1000
    vth = 0.1
    n_max = 90
    S0 = np.array([0.05, 0.5, 0.5, 0, 0.01, 0.025, 0.01])
    #S0 = np.array([0.05, 0.1, 0.1, 0, 0.01, 0.025, 0.01])

    surrogate_model = load_surrogate(device, surrogate_path)
    X_test_image, test_label, test_notation, test_mag = load_test_images(device)
    # test_notation is a dict with key '0', '1', ..., '99'
    test_label = test_label[:,None]
    overlays = notation_overlay(test_notation).to(device) # overlays: [100, 88, 88] with objects pixel of 1
                                                          # 100 is the number of images to attack
                                                          # NOT the batch size!

    X_adv_images = []
    X_adv_filtered_images = []
    gt = []
    params = []
    params_filtered = []
    progress = 0
    total = 0
    for i, (image, label) in enumerate(zip(X_test_image, test_label)):
        progress += 1

        print("Progress: {}/{}".format(progress, X_test_image.size()[0]))

        output_clear = surrogate_model(image[None,None,:,:])
        _, predicted_clear = torch.max(output_clear.data, 1)
        if predicted_clear.item() != label.item():
            # was wrong prediction before attacking
            continue
        total += 1
        gt.append(label.cpu().detach().numpy())

        param = test(surrogate_model, device, image, label, test_notation[str(i)], overlays[i], test_mag[i], batch=batch, N=N, theta_min=theta_min, theta_max=theta_max, vth=vth, lambd=lambd, lambd_gaussian=lambd_gaussian, n_max=n_max, S0=S0)

        # filter out any scatter if its [x,y] is not on the object
        # allow scatters if partially out of object but with centroid on the object
        # param size: [N, 7] -> [N_filtered, 7] or None
        param = param.detach()
        param.requires_grad = False
        param_arg = torch.clone(param)
        param_filtered = filter_param(param_arg, test_notation[str(i)], N)  

        if param.dim() == 2:
            param = param[None, :, :]
        assert(param.size() == (1, N, 7))
        assert(param_filtered.size() == (1, N, 7))
        
        X_adv = image + getNormImage(getImage(E(param, 1, device), device), test_mag[i])
        X_adv = torch.clamp(X_adv, 0, 1)
        X_adv = X_adv.float()
        X_adv = X_adv.squeeze()

        X_adv_filtered = image + getNormImage(getImage(E(param_filtered, 1, device), device), test_mag[i])
        X_adv_filtered = torch.clamp(X_adv_filtered, 0, 1).float().squeeze()

        # DEBUG
        if FLAGS.debug:
            print(param.size())
            print(X_adv.min(), X_adv.max())
            print(X_adv.size())
            #if X_adv.size()[0] != 1:
            #    X_adv = X_adv[None, :,:]
            X_adv = X_adv[None,None,:,:]
            output = surrogate_model(X_adv)
            print(torch.exp(output).cpu().detach().numpy())
            print(label.item(), output.argmax().item())

            X_adv_filtered = X_adv_filtered[None,None,:,:]
            output = surrogate_model(X_adv_filtered)
            print(torch.exp(output).cpu().detach().numpy())
            print(label.item(), output.argmax().item())          
        # END DEBUG
        
        X_adv = X_adv.squeeze().cpu().detach().numpy()
        X_adv_images.append(X_adv)

        X_adv_filtered = X_adv_filtered.squeeze().cpu().detach().numpy()
        X_adv_filtered_images.append(X_adv_filtered)

        param = param.squeeze()
        params.append(param.cpu().detach().numpy())

        param_filtered = param_filtered.squeeze()
        params_filtered.append(param_filtered.cpu().detach().numpy())

    X_adv_images = np.array(X_adv_images)
    X_adv_filtered_images = np.array(X_adv_filtered_images)

    params = np.array(params).reshape([-1, 7])
    params_filtered = np.array(params_filtered).reshape([-1, 7])

    gt = np.array(gt)
    suffix = FLAGS.surrogate
    np.save('adv_dataset_N2/adv_images_{}'.format(suffix), X_adv_images, allow_pickle=False)
    np.save('adv_dataset_N2/adv_images_obj_{}'.format(suffix), X_adv_filtered_images, allow_pickle=False)
    np.save('adv_dataset_N2/adv_gt_{}'.format(suffix), gt, allow_pickle=False)
    np.savetxt('adv_dataset_N2/params_{}.txt'.format(suffix), params, fmt='%.4f')
    np.savetxt('adv_dataset_N2/params_obj_{}.txt'.format(suffix), params_filtered, fmt='%.4f')

if __name__ == '__main__':
    set_all_seeds(0)
    all()
    #all_for_a_target()
