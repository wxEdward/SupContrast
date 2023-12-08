from torchvision import transforms
import torch
import numpy as np
from scipy.io import loadmat
# from main_supcon import set_model
from attacks.otsa import OTSA
from attacks.fgsm import RepresentationAdv
from util import load_train_images, load_test_images
import torch.backends.cudnn as cudnn
from networks.resnet_big import SupConResNet
from networks.aconvnet import AConvNet
from losses import SupConLoss
# from otsa import OTSA
import torch.optim as optim
from torchlars import LARS
from torch.utils.data import DataLoader
import argparse


class Augmentation():
    def __init__(self) -> None:
        #normalize = transforms.Normalize(mean=mea n, std=std)

        self. train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=88, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        # transforms.ToTensor(),
        # normalize,
        ])

    def __call__(self, x):
        x_1 = self.train_transform(x)
        x_2 = self.train_transform(x)
        return [x_1, x_2]
    

def set_augment_model(enc = 'aconv', mode = 'train'):
    modeli = None
    criterion = None
    if  enc == 'aconv':
        model = AConvNet()
    if  enc == 'resnet':
        model = SupConResNet(name='resnet50')

    if mode == 'train':
        criterion = SupConLoss(temperature=0.07)
    if mode == 'test':
        criterion = torch.nn.CrossEntropyLoss()
    # enable synchronized Batch Normalization
    #if opt.syncBN:
        #model = apex.parallel.convert_syncbn_model(model)
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion

from scipy.io import savemat
if __name__ == '__main__':
    parser = argparse.ArgumentParser('argument for augmentation')
    parser.add_argument('--mode', type=str, default='train',
                        help='train or test')
    parser.add_argument('--enc', type=str, default='aconv',
                        help='attacked model')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch_size')
    parser.add_argument('--attack', type=str, default='fgsm', help='attack type')
    parser.add_argument('--batch', type=int, default=32, help='batch size')

    arg = parser.parse_args()
    device = torch.device("cuda")
    augment = Augmentation()
    batch = arg.batch

    data_path = 'adv_dataset/' + arg.enc + '_' + arg.attack + '_' + arg.mode + '.npy'
    model, criterion = set_augment_model(arg.enc, arg.mode)

    print("Mode: ", arg.mode)
    print("Encoder: ", arg.enc)
    print("Attack: ", arg.attack)

    if arg.attack == 'fgsm':

        dataloader = None
        model_params = []
        model_params += model.parameters()
        attack_1 = RepresentationAdv(model, epsilon=0.0314, alpha=0.007)

        if arg.mode == 'train':
            X_train, y_train, musk_train = load_train_images(device)
            train_data = [[X_train[i], y_train[i]] for i in range(y_train.size()[0])]
            train_dataloader = DataLoader(train_data, batch_size=batch, shuffle=False)
            dataloader = train_dataloader

        if arg.mode == 'test':
            X_test, y_test, musk_test = load_test_images(device)
            test_data = [[X_test[i], y_test[i]] for i in range(y_test.size()[0])]
            test_dataloader = DataLoader(test_data, batch_size=batch, shuffle=False)
            dataloader = test_dataloader
            attack_1.loss_type = 'ce'

        base_optimizer = optim.SGD(model_params, lr=0.2, momentum=0.9, weight_decay=1e-6)
        optimizer = LARS(optimizer=base_optimizer, eps=1e-8, trust_coef=0.001)
        fgsm_data = []
        for idx, (images, labels) in enumerate(dataloader):
            images = augment(images)
            print("Batch ", idx)
            if torch.cuda.is_available():
                images[0] = images[0].cuda(non_blocking=True)
                images[1] = images[1].cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
            bsz = labels.shape[0]
            adv_1, _ = attack_1.get_loss(original_images=images[0], target=images[1], optimizer=optimizer,
                                         weight=256, random_start=True)
            # print(type(adv_1[0]), adv_1[0].size())
            # print(adv_1.cpu().numpy())
            fgsm_data.extend(adv_1.cpu().numpy())

        fgsm_data = np.array(fgsm_data)
        with open(data_path, 'wb') as f:
            np.save(f, fgsm_data, allow_pickle=False)
        # np.save('adv_dataset/fgsm_data_test.npy', fgsm_data, allow_plickle=False)
        print("Number of perturbed samples: ", len(fgsm_data))

    if arg.attack == 'otsa':

        dataloader = None

        if arg.mode == 'train':
            X_train, y_train, musk_train = load_train_images(device)
            train_data = [[X_train[i], y_train[i], musk_train[i]] for i in range(y_train.size()[0])]
            dataloader = DataLoader(train_data, batch_size=32, shuffle=False)

        if arg.mode == 'test':
            X_test, y_test, musk_test = load_test_images(device)
            test_data = [[X_test[i], y_test[i], musk_test[i]] for i in range(y_test.size()[0])]
            dataloader = DataLoader(test_data, batch_size=32, shuffle=False)

        attack_2 = OTSA(0.07)
        adv_2, adv_2_filtered = attack_2.generate(model, criterion, dataloader, batch=batch)

        with open(data_path, 'wb') as f:
            np.save(f, adv_2_filtered, allow_pickle=False)
        print("Number of samples: ", len(adv_2_filtered))

    '''

    fgsm_data = []

    for idx, (images, labels) in enumerate(test_dataloader):
        images = augment(images)
        print("Batch ", idx)
        if torch.cuda.is_available():
            images[0] = images[0].cuda(non_blocking=True)
            images[1] = images[1].cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]
        adv_1, _ = attack_1.get_loss(original_images=images[0], target = images[1], optimizer=optimizer,
                                          weight=256, random_start=True)
        #print(type(adv_1[0]), adv_1[0].size())
        #print(adv_1.cpu().numpy())
        fgsm_data.extend(adv_1.cpu().numpy())
    
    fgsm_data = np.array(fgsm_data)
    with open('adv_dataset/fgsm_data_test.npy', 'wb') as f:
        np.save(f, fgsm_data,allow_pickle=False)
    #np.save('adv_dataset/fgsm_data_test.npy', fgsm_data, allow_plickle=False)
    print(len(fgsm_data))
    
    '''

    '''
    
    print("Start for training dataset")
    train_data = [[X_train[i], y_train[i], musk_train[i]] for i in range(y_train.size()[0])]
    train_dataloader = DataLoader(train_data, batch_size=32, shuffle=False)
    
    test_data = [[X_test[i], y_test[i], musk_test[i]] for i in range(y_test.size()[0])]
    test_dataloader = DataLoader(test_data, batch_size=32, shuffle=False)
    
    
    attack_2 = OTSA(0.07)
    adv_2, adv_2_filtered = attack_2.generate(model, criterion, train_dataloader, batch=32)
    with open('adv_dataset/otsa_data_train.npy','wb') as f:
        np.save(f, adv_2, allow_pickle=False)
    with open('adv_dataset/otsa_data_train_filtered.npy','wb') as f:
        np.save(f, adv_2_filtered, allow_pickle=False)
    '''
    '''
    print("Start for test dataset")

    attack_2 = OTSA(0.07)
    adv_2, adv_2_filtered = attack_2.generate(model, criterion, test_dataloader, batch=32)
    with open('adv_dataset/otsa_data_test.npy','wb') as f:
        np.save(f, adv_2, allow_pickle=False)
    with open('adv_dataset/otsa_data_test_filtered.npy','wb') as f:
        np.save(f, adv_2_filtered, allow_pickle=False)
    '''
    # print(len(adv_2), len(adv_2_filtered))


    # print(len(adv_1))
