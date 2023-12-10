import torch
import torch.nn.functional as F
from losses import pairwise_similarity, NT_xent

def project(x, original_x, epsilon, _type='linf'):

    if _type == 'linf':
        max_x = original_x + epsilon
        min_x = original_x - epsilon

        x = torch.max(torch.min(x, max_x), min_x)
    else:
        raise NotImplementedError

    return x


class FastGradientSignUntargeted():
    """
        Fast gradient sign untargeted adversarial attack, minimizes the initial class activation
        with iterative grad sign updates
    """

    def __init__(self, model, linear, epsilon=0.0314, alpha=0.007, min_val = 0.0, max_val=1.0, max_iters=7, _type='linf'):

        # Model
        self.model = model
        self.linear = linear
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

    def perturb(self, original_images, labels, reduction4loss='mean', random_start=True):
        # original_images: values are within self.min_val and self.max_val
        # The adversaries created from random close points to the original data
        if random_start:
            rand_perturb = torch.FloatTensor(original_images.shape).uniform_(
                -self.epsilon, self.epsilon)
            rand_perturb = rand_perturb.cuda()
            x = original_images.clone() + rand_perturb
            x = torch.clamp(x, self.min_val, self.max_val)
        else:
            x = original_images.clone()

        x.requires_grad = True

        self.model.encoder.eval()
        self.model.head.eval()

        if not self.linear == 'None':
            self.linear.eval()

        with torch.enable_grad():
            for _iter in range(self.max_iters):

                self.model.zero_grad()
                if not self.linear == 'None':
                    self.linear.zero_grad()

                if self.linear == 'None':
                    outputs = self.model.encoder(x)
                else:
                    outputs = self.linear(self.model.encoder(x))

                loss = F.cross_entropy(outputs, labels, reduction=reduction4loss)

                grad_outputs = None
                grads = torch.autograd.grad(loss, x, grad_outputs=grad_outputs, only_inputs=True, retain_graph=False)[0]

                if self._type == 'linf':
                    scaled_g = torch.sign(grads.data)

                x.data += self.alpha * scaled_g

                x = torch.clamp(x, self.min_val, self.max_val)
                x = project(x, original_images, self.epsilon, self._type)

        return x.detach()

class RepresentationAdv():

    def __init__(self, model, epsilon, alpha, min_val = 0.0, max_val=1.0, max_iters=7,  _type='linf', loss_type='sim', regularize='original'):

        # Model
        self.model = model
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
            x = torch.clamp(x,self.min_val, self.max_val)
        else:
            x = original_images.clone()

        x.requires_grad = True
        
        self.model.encoder.eval()
        self.model.head.eval()
        batch_size = len(x)
        
        with torch.enable_grad():
            for _iter in range(self.max_iters):

                self.model.encoder.zero_grad()
                self.model.head.zero_grad()

                if self.loss_type == 'mse':
                    loss = F.mse_loss(self.model(x),self.model(target))
                elif self.loss_type == 'ce':
                    loss = F.cross_entropy((self.model(x), self.model(target)))
                elif self.loss_type == 'sim':
                    inputs = torch.cat((x, target))
                    output = self.model(inputs)
                    similarity,_  = pairwise_similarity(output, temperature=0.5, multi_gpu=False, adv_type = 'None') 
                    loss  = NT_xent(similarity, 'None')
                elif self.loss_type == 'l1':
                    loss = F.l1_loss(self.model(x), self.model(target))
                elif self.loss_type =='cos':
                    loss = 1-F.cosine_similarity(self.model(x), self.model(target)).mean()

                grads = torch.autograd.grad(loss, x, grad_outputs=None, only_inputs=True, retain_graph=False)[0]

                if self._type == 'linf':
                    scaled_g = torch.sign(grads.data)
               
                x.data += self.alpha * scaled_g

                x = torch.clamp(x,self.min_val,self.max_val)
                x = project(x, original_images, self.epsilon, self._type)

        self.model.encoder.train()
        self.model.head.train()
        optimizer.zero_grad()

        if self.loss_type == 'mse':
            loss = F.mse_loss(self.model(x),self.model(target)) * (1.0/batch_size)
        elif self.loss_type == 'ce':
            loss = F.cross_entropy(self.model(x), self.model(target)) * (1.0/batch_size)
        elif self.loss_type == 'sim':
            if self.regularize== 'original':
                inputs = torch.cat((x, original_images))
            else:
                inputs = torch.cat((x, target))
            output = self.model(inputs)
            similarity, _  = pairwise_similarity(output, temperature=0.5, multi_gpu=False, adv_type = 'None')
            loss  = (1.0/weight) * NT_xent(similarity, 'None')  
        elif self.loss_type == 'l1':
            loss = F.l1_loss(self.model(x), self.model(target)) * (1.0/batch_size)
        elif self.loss_type == 'cos':
            loss = 1-F.cosine_similarity(self.model(x), self.model(target)).sum() * (1.0/batch_size)

        return x.detach(), loss
