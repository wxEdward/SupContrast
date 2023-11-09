import torch
import numpy as np
from losses import pairwise_similarity, NT_xent
import torch
import torch.nn.functional as F

def project(x, original_x, epsilon, _type='linf'):

    if _type == 'linf':
        max_x = original_x + epsilon
        min_x = original_x - epsilon

        x = torch.max(torch.min(x, max_x), min_x)
    else:
        raise NotImplementedError
    return x

class FGSM():
    def __init__(self, device):
        self.device = device
    '''
        Arguments:
            model: neural network model
            data: [batch, C, d1, d2, ...], original images
            label: [batch], ground truth labels
            criterion: loss function
            epsilon: L-infinity limit
            *args, **kwargs: additional arguments to `model`
    '''
    def generate(self, model, data, label, criterion, epsilon, *args, **kwargs):
        data = data.requires_grad_()
        data_min = data.min().item()
        data_max = data.max().item()

        output = model(data, *args, **kwargs)
        loss = criterion(output, label)
        model.zero_grad()
        loss.backward()

        sign_grad = data.grad.data.sign()
        perturbed = data + (epsilon * data_max) * sign_grad
        perturbed = torch.clamp(perturbed, data_min, data_max)
        return perturbed
    
    def inference(self, model, data, *args, **kwargs):
        output = model(data, *args, **kwargs)
        return output


class RepresentationAdv():

    def __init__(self, model, projector, epsilon, alpha, min_val, max_val, max_iters, _type='linf', loss_type='sim',
                 regularize='original'):

        # Model
        self.model = model
        self.projector = projector
        self.regularize = regularize
        # Maximum perturbation
        self.epsilon = epsilon
        # Movement multiplier per iteration
        self.alpha = alpha
        # Minimum value of the pixels
        self.min_val = min_val
        # Maximum value of the pixels
        self.max_val = max_val
        # Maximum numbers of iteration to generated adversaries
        self.max_iters = max_iters
        # The perturbation of epsilon
        self._type = _type
        # loss type
        self.loss_type = loss_type

    def get_loss(self, original_images, target, optimizer, weight, random_start=True):
        if random_start:
            rand_perturb = torch.FloatTensor(original_images.shape).uniform_(
                -self.epsilon, self.epsilon)
            rand_perturb = rand_perturb.float().cuda()
            x = original_images.float().clone() + rand_perturb
            x = torch.clamp(x, self.min_val, self.max_val)
        else:
            x = original_images.clone()

        x.requires_grad = True

        self.model.eval()
        self.projector.eval()
        batch_size = len(x)

        with torch.enable_grad():
            for _iter in range(self.max_iters):

                self.model.zero_grad()
                self.projector.zero_grad()

                if self.loss_type == 'mse':
                    loss = F.mse_loss(self.projector(self.model(x)), self.projector(self.model(target)))
                elif self.loss_type == 'sim':
                    inputs = torch.cat((x, target))
                    output = self.projector(self.model(inputs))
                    similarity, _ = pairwise_similarity(output, temperature=0.5, multi_gpu=False, adv_type='None')
                    loss = NT_xent(similarity, 'None')
                elif self.loss_type == 'l1':
                    loss = F.l1_loss(self.projector(self.model(x)), self.projector(self.model(target)))
                elif self.loss_type == 'cos':
                    loss = 1 - F.cosine_similarity(self.projector(self.model(x)),
                                                   self.projector(self.model(target))).mean()

                grads = torch.autograd.grad(loss, x, grad_outputs=None, only_inputs=True, retain_graph=False)[0]

                if self._type == 'linf':
                    scaled_g = torch.sign(grads.data)

                x.data += self.alpha * scaled_g

                x = torch.clamp(x, self.min_val, self.max_val)
                x = project(x, original_images, self.epsilon, self._type)

        self.model.train()
        self.projector.train()
        optimizer.zero_grad()

        if self.loss_type == 'mse':
            loss = F.mse_loss(self.projector(self.model(x)), self.projector(self.model(target))) * (1.0 / batch_size)
        elif self.loss_type == 'sim':
            if self.regularize == 'original':
                inputs = torch.cat((x, original_images))
            else:
                inputs = torch.cat((x, target))
            output = self.projector(self.model(inputs))
            similarity, _ = pairwise_similarity(output, temperature=0.5, multi_gpu=False, adv_type='None')
            loss = (1.0 / weight) * NT_xent(similarity, 'None')
        elif self.loss_type == 'l1':
            loss = F.l1_loss(self.projector(self.model(x)), self.projector(self.model(target))) * (1.0 / batch_size)
        elif self.loss_type == 'cos':
            loss = 1 - F.cosine_similarity(self.projector(self.model(x)), self.projector(self.model(target))).sum() * (
                        1.0 / batch_size)

        return x.detach(), loss