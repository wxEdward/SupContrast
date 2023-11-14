from __future__ import print_function

import math
import numpy as np
from scipy.io import loadmat
import torch
import torch.optim as optim
from attacks.fgsm import RepresentationAdv
from attacks.otsa import OTSA
from torchlars import LARS


class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform, model, args):
        self.transform = transform
        self.model = model
        self.attack_1 = RepresentationAdv(model, epsilon=args.epsilon, alpha=args.alpha, min_val=args.min, max_val=args.max,
                                          max_iters=args.k, _type=args.attack_type, loss_type=args.loss_type,
                                          regularize = args.regularize_to)
        self.attack_2 = OTSA(args.temperture)

        model_params = []
        model_params += model.parameters()
        base_optimizer = optim.SGD(model_params, lr=args.lr, momentum=0.9, weight_decay=args.decay)
        self.optimizer = LARS(optimizer=base_optimizer, eps=1e-8, trust_coef=0.001)

    def __call__(self, x, y, overlays):
        augmented_x1 = self.transform(x)
        augmented_x2 = self.transform(x)
        adv_1, _ = self.attack_1.get_loss(original_images=augmented_x1, target = augmented_x2, optimizer=self.optimizer,
                                          weight=256, random_start=True)
        adv_2 = self.attack_2.generate(self.model, x, y, overlays)
        return [adv_1, adv_2]
 

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def set_optimizer(opt, model):
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    return optimizer


def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state


def load_train_images(device):

    data = loadmat('MSTAR/mat/train88.mat')
    masks = np.load("MSTAR/mat/train88_masks.npy")

    imgs = data['train_data']
    labels = data['train_label']

    # imgs = torch.from_numpy(imgs).to(device)
    # labels = torch.from_numpy(labels).to(device)
    masks = torch.from_numpy(masks).to(device)

    return imgs, labels, masks


def load_test_images(device):

    data = loadmat('MSTAR/mat/test88.mat')
    masks = np.load("MSTAR/mat/test88_masks.npy")

    imgs = data['test_data']
    labels = data['test_label']

    # imgs = torch.from_numpy(imgs).to(device)
    # labels = torch.from_numpy(labels).to(device)
    masks = torch.from_numpy(masks).to(device)

    return imgs, labels, masks