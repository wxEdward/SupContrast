import torch
import numpy as np

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

