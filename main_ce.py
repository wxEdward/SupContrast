from __future__ import print_function

import os
import sys
import argparse
import time
import math

# import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

from util import AverageMeter
from util import adjust_learning_rate, warmup_learning_rate, accuracy
from util import set_optimizer, save_model
from networks.resnet_big import SupCEResNet
from util import load_train_images,load_test_images
import numpy as np
from networks.aconvnet import FullAConvNet

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=500,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.2,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='350,400,450',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100'], help='dataset')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')

    opt = parser.parse_args()

    # set the path according to the environment
    opt.data_folder = './adv_dataset/'
    opt.model_path = './save/SupCon/models/final'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = 'SupCE_{}_lr_{}_decay_{}_bsz_{}_trial_{}'.\
        format(opt.model, opt.learning_rate, opt.weight_decay,
               opt.batch_size, opt.trial)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def set_loader(opt, device):
    ori_train_X, ori_train_y, _ = load_train_images(device)
    ori_test_X, ori_test_y, _ = load_test_images(device)
    print(ori_train_X.shape)
    if opt.model == 'aconv':
        fgsm_test_X = np.load('adv_dataset_ce/aconv_pgd_test.npy')
        otsa_test_X = np.load('adv_dataset_ce/aconv_otsa_test.npy')
    if opt.model == 'resnet':
        fgsm_test_X = np.load('adv_dataset_ce/resnet_pgd_test.npy')
        otsa_test_X = np.load('adv_dataset_ce/resnet_otsa_test.npy')

    fgsm_test_X = torch.from_numpy(fgsm_test_X).to(device)
    otsa_test_X = torch.from_numpy(otsa_test_X).to(device)
    otsa_test_X = otsa_test_X.unsqueeze(1)

    train_data = [[ori_train_X[i], ori_train_y[i]] for i in range(ori_train_y.size()[0])]
    train_dataloader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True)

    test_X = torch.cat([ori_test_X, fgsm_test_X, otsa_test_X], dim=1)
    test_data = [[test_X[i], ori_test_y[i]] for i in range(ori_test_y.size()[0])]
    test_dataloader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=True)

    return train_dataloader, test_dataloader

def set_model(opt):
    model = None
    if opt.model == 'resnet':
        model = SupCEResNet(name='resnet50')
    if opt.model == 'aconv':
        model = FullAConvNet()

    model = SupCEResNet()
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)

    model.load_state_dict(torch.load('save/SupCon/models/final/SupCE_resnet_lr_0.2_decay_0.0001_bsz_64_trial_0/ckpt_epoch_30.pth')['model'])

    criterion = torch.nn.CrossEntropyLoss()
    '''

    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)
    '''
    if torch.cuda.is_available():
        # if torch.cuda.device_count() > 1:
            # model = torch.nn.DataParallel(model)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion


def train(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        output = model(images)
        loss = criterion(output, labels)

        # update metric
        losses.update(loss.item(), bsz)
        acc1, acc5 = accuracy(output, labels, topk=(1, 5))
        top1.update(acc1[0], bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))
            sys.stdout.flush()

    return losses.avg, top1.avg


def update(model, criterion, image, label, losses, top1):
    bsz = label.shape[0]
    output = model(image)
    loss = criterion(output,label)
    losses.update(loss.item(), bsz)
    acc1, acc5 = accuracy(output, label, topk=(1, 5))
    top1.update(acc1[0], bsz)

def validate(val_loader, model, criterion, opt):
    """validation"""
    model.eval()
    # All
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    # TA
    losses_ta = AverageMeter()
    top1_ta = AverageMeter()
    # RA
    losses_ra = AverageMeter()
    top1_ra = AverageMeter()
    # RA - PGD
    losses_ra_pgd = AverageMeter()
    top1_ra_pgd = AverageMeter()
    # RA - OTSA
    losses_ra_otsa = AverageMeter()
    top1_ra_otsa = AverageMeter()

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            i_1, i_2, i_3 = torch.split(images, [1, 1, 1], dim=1) # i_1: clean; i_2: PGD; i_3: OTSA
            i_1 = i_1.cuda(non_blocking=True)
            i_2 = i_2.cuda(non_blocking=True)
            i_3 = i_3.cuda(non_blocking=True)
            images = torch.cat([i_1, i_2, i_3], dim=0)
            if torch.cuda.is_available():
                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
            labels = labels.cuda()
            # Compute TA
            update(model,criterion,i_1,labels,losses_ta,top1_ta)
            # Compute RA
            img_ra = torch.cat([i_2,i_3])
            labels_ra = labels.repeat(2)
            update(model, criterion, img_ra, labels_ra, losses_ra, top1_ra)
            # Compute RA - PGD
            update(model,criterion,i_2,labels,losses_ra_pgd,top1_ra_pgd)
            # Compute RA - OTSA
            update(model,criterion,i_3,labels,losses_ra_otsa,top1_ra_otsa)
            #Compute all
            labels_cat = labels.repeat(3)
            update(model,criterion,images,labels_cat, losses, top1)
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time,
                       loss=losses, top1=top1))

    print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
    print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1_ta))
    print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1_ra))
    print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1_ra_pgd))
    print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1_ra_otsa))
    return losses.avg, top1.avg, top1_ta.avg, top1_ra.avg, top1_ra_pgd.avg, top1_ra_otsa.avg


def main():
    device = torch.device("cuda")
    best_acc = best_ta = best_ra = best_ra_pgd = best_ra_otsa = 0
    opt = parse_option()

    # build data loader
    train_loader, val_loader = set_loader(opt,device)

    # build model and criterion
    model, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    '''
        # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # evaluation
        loss, val_acc, ta, ra, ra_pgd, ra_otsa = validate(val_loader, model, criterion, opt)
        if val_acc > best_acc:
            best_acc = val_acc
        if ta > best_ta:
            best_ta = ta
        if ra > best_ra:
            best_ra = ra
        if ra_pgd > best_ra_pgd:
            best_ra_pgd = ra_pgd
        if ra_otsa > best_ra_otsa:
            best_ra_otsa = ra_otsa

        if val_acc > best_acc:
            best_acc = val_acc

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)
        print('best accuracy: ', best_acc, best_ta, best_ra, best_ra_pgd, best_ra_otsa)

    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)

    print('best accuracy: ', best_acc, best_ta, best_ra, best_ra_pgd, best_ra_otsa)
    '''
    # evaluation
    loss, val_acc, ta, ra, ra_pgd, ra_otsa = validate(val_loader, model, criterion, opt)
    if val_acc > best_acc:
        best_acc = val_acc
    if ta > best_ta:
        best_ta = ta
    if ra > best_ra:
        best_ra = ra
    if ra_pgd > best_ra_pgd:
        best_ra_pgd = ra_pgd
    if ra_otsa > best_ra_otsa:
        best_ra_otsa = ra_otsa

    if val_acc > best_acc:
        best_acc = val_acc

    print('best accuracy: ', best_acc, best_ta, best_ra, best_ra_pgd, best_ra_otsa)



if __name__ == '__main__':
    main()
