import torch
import numpy as np
from torch.fft import ifft2 as ifft2
import torch.nn.functional as F

from attacks.scatter_batch import E_i, E, getImage
from losses import SupConLoss

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
class OTSA():
    def __init__(self, temerature) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = SupConLoss(temperature=temerature)

    def initParam(self, batch, N, theta_min, theta_max, notation, device):
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
    def gaussian(self, params, notation, batch, N, device, sigma=0.4):
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
    def attack(self, model, criterion, device, X_image, y_gt, notation, overlay, batch, N, theta_min, theta_max, vth, lambd, lambd_gaussian, n_max, S0):
        '''
        assert(mag.shape == (2, ))
        print(mag[0], mag[1])
        assert(mag[0] < mag[1])
        
        '''

        std = (theta_max - theta_min)/200
        std = np.tile(std.reshape([1,-1]), [N, 1])
        std = torch.from_numpy(std).to(device)
        S0 = np.tile(S0.reshape([1, 1,-1]), [batch, N, 1])
        S0 = torch.from_numpy(S0).to(device)

        param, freeze = self.initParam(batch, N, theta_min, theta_max, notation, device)#.to(device)

        theta_min = np.tile(theta_min.reshape([1,-1]), [N, 1])
        theta_min = torch.from_numpy(theta_min).to(device)
        theta_max = np.tile(theta_max.reshape([1,-1]), [N, 1])
        theta_max = torch.from_numpy(theta_max).to(device)

        iteration = 0

        x_max = X_image.max().item()

        # getImage returns [batch, 88, 88], X_adv will be of the same size
        
        #X_image = X_image[None, :, :]
        print(X_image.size())
        X_image = X_image.repeat(batch,1,1)
        #X_image = X_image[:, None, :, :]
        X_adv = X_image + self.getNormImage(getImage(E(param, batch, device), device)) #* overlay
        X_adv = torch.clamp(X_adv, 0, 1)
        X_adv = X_adv.float()
        X_adv = X_adv[:,None,:,:]
        # print(X_adv.size())# X_adv resized to [batch, 1, 88, 88]
        ori_feat = model(X_image[:, None, :, :])
        #ori_feat = ori_feat[:,, :]
        adv_feat = model(X_adv)   # output is of size [batch, 10]
        # adv_feat = output[:,None,:]
        ##print(output.size())
        ori_adv_feat = torch.cat([ori_feat.unsqueeze(1), adv_feat.unsqueeze(1)], dim=1)
        y_gt_batch = y_gt.repeat(batch)
        loss_pred = criterion(ori_adv_feat) 
        loss_gaussian = lambd_gaussian * self.gaussian(param, notation, batch, N, device)
        loss = loss_pred + loss_gaussian
        total_loss = torch.sum(loss)  # prepare to backward for the entire batch
        

        # v = output[:, y_gt]  # confidence of ground truth.
        # v = torch.exp(v)     # v is of size [batch, 1]
        
        v = loss_pred.detach().clone()

        min_v = v.max().item() # get the minimum of v
        min_param = param[v.argmax()]
        
        while iteration < n_max:
            iteration += 1
            
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
            param_new = param_new.detach()

            # reduce amplitude if a scatter's max pixel > 1            
            for i in range(N):
                img_batch = self.getNormImage(getImage(E_i(param_new[:,i,:], batch, device), device)) #* overlay
                for b in range(batch):
                    if torch.max(img_batch[b]).item() > x_max:
                        param_new[b][i][0] = param_new[b][i][0] / (np.random.uniform() + param_new[b][i][0])

            param = param_new
            S0 = lambd * S0 + (1-lambd) * torch.abs(delta_param)

            param.requires_grad = True
            img_noise = self.getNormImage(getImage(E(param, batch, device), device)) #* overlay
            X_adv = X_image + img_noise
            X_adv = torch.clamp(X_adv, 0, 1)
            X_adv = X_adv.float()
            X_adv = X_adv[:,None,:,:]


             # print(X_adv.size())# X_adv resized to [batch, 1, 88, 88]
            ori_feat = model(X_image[:,None,:,:])
             #ori_feat = ori_feat[:,, :]
            adv_feat = model(X_adv)   # output is of size [batch, 10]
            # adv_feat = output[:,None,:]
            ##print(output.size())
            ori_adv_feat = torch.cat([ori_feat.unsqueeze(1), adv_feat.unsqueeze(1)], dim=1)
            y_gt_batch = y_gt.repeat(batch)
            loss_pred = criterion(ori_adv_feat) 
            
            #output_bz = output[None, :, :]

            # Since not all param are updated, we compute the loss again to make sure the loss is aligned with the param
            #loss_pred = criterion(output_bz, y_gt_batch) 
            loss_gaussian = lambd_gaussian * self.gaussian(param, notation, batch, N, device)
            loss = loss_pred + loss_gaussian
            total_loss = torch.sum(loss)
            #loss_gaussian: [batch, 1]
            #v: [batch, 1]
            v = loss_pred.detach().clone()
            #v = torch.exp(v)
            #v: [batch,]
            #v = v.squeeze()
            mask = torch.where(torch.abs(lambd_gaussian * 0.2 * 0.4 - loss_gaussian) < 1e-4, 0, 1)
            v = v + mask

            if v.min().item() < min_v:
                min_v = v.min().item()
                min_param = param[v.argmin()]

        print(iteration)
        """
        if FLAGS.debug:
            print(min_param)
        """
        

        return min_param

    def notation_overlay(self, test_notation):
        overlays = torch.zeros([100, 88, 88])
        for i in range(100):
            notation = test_notation[str(i)]
            for coor in notation:
                overlays[i][coor[0], coor[1]] = 1
        return overlays


    def filter_param(self, param, notation, N):
        if param.dim() == 2:
            param = param[None, :, :]
            assert(param.size() == (1, N, 7))
        for batch in range(param.shape[0]):
            for i in range(N):
                xy = param[batch, i, 1:3].cpu().detach().numpy()
                xy = np.round(xy)  # TODO: is round ok? 
                if not xy.tolist() in notation.tolist():
                #if not (xy == notation).any():
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
    def getNormImage(self, img):
        min_pixel = torch.min(img)
        max_pixel = torch.max(img)
        assert(min_pixel < max_pixel)
        img = (img - min_pixel) / (max_pixel - min_pixel)
        return img

    def generate(self, model, criterion,X_image, y_gt, overlays, batch=10, N=3, theta_min=np.array([0, 0, 0, -1, 0, 0, -1]), theta_max=np.array([10, 87, 87, 1, 2, 5, 1]), 
                 vth=0.1, lambd=0.5, lambd_gaussian=1000, n_max=20, S0=np.array([0.05, 0.5, 0.5, 0, 0.01, 0.025, 0.01])):
        #S0 = np.array([0.05, 0.1, 0.1, 0, 0.01, 0.025, 0.01])

        surrogate_model = model
        # test_notation is a dict with key '0', '1', ..., '99'
        y_gt = y_gt[:,None]

        overlays = overlays.to(device) # overlays: [100, 88, 88] with objects pixel of 1
                                                            # 100 is the number of images to attack
                                                            # NOT the batch size!
        X_adv_images = []
        X_adv_filtered_images = []
        gt = []
        params = []
        params_filtered = []
        progress = 0
        total = 0
        for i, (image, label) in enumerate(zip(X_image, y_gt)):
            progress += 1
            print("Progress: {}/{}".format(progress, X_image.size()[0]))
            """
            output_clear = surrogate_model(image[None,:,:])
            _, predicted_clear = torch.max(output_clear.data, 1)
            if predicted_clear.item() != label.item():
                # was wrong prediction before attacking
                continue
            """
            total += 1
            gt.append(label.cpu().detach().numpy())
            ol = overlays[i].cpu()
            notation = np.argwhere(ol==1)
            param = self.attack(surrogate_model,criterion, device, image, label, notation, overlays[i], batch=batch, N=N, theta_min=theta_min, theta_max=theta_max, vth=vth, lambd=lambd, lambd_gaussian=lambd_gaussian, n_max=n_max, S0=S0)

            # filter out any scatter if its [x,y] is not on the object
            # allow scatters if partially out of object but with centroid on the object
            # param size: [N, 7] -> [N_filtered, 7] or None
            param = param.detach()
            param.requires_grad = False
            param_arg = torch.clone(param)
            param_filtered = self.filter_param(param_arg, notation, N)  

            if param.dim() == 2:
                param = param[None, :, :]
            assert(param.size() == (1, N, 7))
            assert(param_filtered.size() == (1, N, 7))
            
            X_adv = image + self.getNormImage(getImage(E(param, 1, device), device))
            X_adv = torch.clamp(X_adv, 0, 1)
            X_adv = X_adv.float()
            X_adv = X_adv.squeeze()

            X_adv_filtered = image + self.getNormImage(getImage(E(param_filtered, 1, device), device))
            X_adv_filtered = torch.clamp(X_adv_filtered, 0, 1).float().squeeze()

            """
            
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
            """
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
        suffix = 'ACONV'
        np.save('adv_dataset_N3/adv_images_{}'.format(suffix), X_adv_images, allow_pickle=False)
        np.save('adv_dataset_N3/adv_images_obj_{}'.format(suffix), X_adv_filtered_images, allow_pickle=False)
        np.save('adv_dataset_N3/adv_gt_{}'.format(suffix), gt, allow_pickle=False)
        np.savetxt('adv_dataset_N3/params_{}.txt'.format(suffix), params, fmt='%.4f')
        np.savetxt('adv_dataset_N3/params_obj_{}.txt'.format(suffix), params_filtered, fmt='%.4f')
        return X_adv_images, X_adv_filtered_images

