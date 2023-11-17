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
        adv_1, _ = self.attack_1.get_loss(original_images=augmented_x1, target = augmented_x1, optimizer=self.optimizer,
                                          weight=256, random_start=True)
        adv_2 = self.attack_2.generate(self.model, augmented_x2, y, overlays)
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
    
    y_train = labels
    y_train = y_train.squeeze()

    X_train_image = imgs

    X_train_image = (X_train_image - X_train_image.min(axis=(1, 2)).reshape([-1, 1, 1])) / X_train_image.ptp(
        axis=(1, 2)).reshape([-1, 1, 1])
    print(np.min(X_train_image), np.max(X_train_image))
    X_train_image = X_train_image[:, np.newaxis, :, :]
    X_train_image = torch.from_numpy(X_train_image).to(device)
    X_train_image = X_train_image.type(torch.float32)

    if y_train.min().item == 1 and y_train.max().item() == 10:
        train_label = torch.from_numpy(y_train).to(device) - 1
    else:
        train_label = torch.from_numpy(y_train).to(device)
    train_label = train_label.type(torch.int64)

    return X_train_image, train_label, masks

def load_test_images(device):

    data = loadmat('MSTAR/mat/test88.mat')
    masks = np.load("MSTAR/mat/test88_masks.npy")

    imgs = data['test_data']
    labels = data['test_label']

    X_test_image = imgs
    y_test = labels
    y_test = y_test.squeeze()

    X_test_image = (X_test_image - X_test_image.min(axis=(1, 2)).reshape([-1, 1, 1])) / X_test_image.ptp(
        axis=(1, 2)).reshape([-1, 1, 1])
    print(np.min(X_test_image), np.max(X_test_image))
    X_test_image = X_test_image[:, np.newaxis, :, :]
    X_test_image = torch.from_numpy(X_test_image).to(device)
    X_test_image = X_test_image.type(torch.float32)

    if y_test.min().item == 1 and y_test.max().item() == 10:
        test_label = torch.from_numpy(y_test).to(device) - 1
    else:
        test_label = torch.from_numpy(y_test).to(device)
    test_label = test_label.type(torch.int64)

    return X_test_image, test_label, masks




def set_loader(model, device):

    augment = TwoCropTransform(train_transform, model, opt)
    X_train_augmented = augment(X_train_image, train_label, musk_train)
    X_test_augmented = augment(X_test_image,test_label, musk_test)

    train_data = [[X_train_augmented[i], train_label[i]] for i in range(train_label.size()[0])]
    test_data = [[X_test_augmented [i], test_label[i]] for i in range(test_label.size()[0])]

    #normalize = transforms.Normalize(mean=mean, std=std)

    train_dataloader = DataLoader(train_data, batch_size=opt.batch_size,
                                  shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=opt.batch_size,
                                 shuffle=False)

    return train_dataloader, test_dataloader
