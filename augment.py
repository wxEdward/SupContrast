from torchvision import transforms
import torch
import numpy as np
from scipy.io import loadmat
# from main_supcon import set_model
from attacks.otsa import OTSA
from attacks.fgsm import RepresentationAdv, FastGradientSignUntargeted
from util import load_train_images, load_test_images
import torch.backends.cudnn as cudnn
from networks.resnet_big import SupConResNet, LinearClassifier, SupCEResNet
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
    model = None
    criterion = None
    classifier = None
    if  enc == 'aconv':
        model = AConvNet()
    if  enc == 'resnet':
        model = SupConResNet(name='resnet50')

    if mode == 'train':
        criterion = SupConLoss(temperature=0.07)
    if mode == 'tune' or 'test':
        criterion = torch.nn.CrossEntropyLoss()
        classifier = LinearClassifier(enc)
    # enable synchronized Batch Normalization
    #if opt.syncBN:
        #model = apex.parallel.convert_syncbn_model(model)
    model = SupCEResNet()
    model.load_state_dict(torch.load('/save/SupCon/models/final/SupCE_resnet_lr_0.2_decay_0.0001_bsz_64_trial_0/ckpt_epoch_30.pth'))

    model = model.encoder
    classifier = model.fc

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        if classifier is not None:
            classifier = classifier.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, classifier, criterion

from scipy.io import savemat
if __name__ == '__main__':
    parser = argparse.ArgumentParser('argument for augmentation')
    parser.add_argument('--mode', type=str, default='train',
                        help='train or tune or test')
    parser.add_argument('--enc', type=str, default='aconv',
                        help='attacked model')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch_size')
    parser.add_argument('--attack', type=str, default='pgd', help='attack type')

    arg = parser.parse_args()
    device = torch.device("cuda")
    augment = Augmentation()
    batch = arg.batch_size

    data_path = 'adv_dataset_ce/' + arg.enc + '_' + arg.attack + '_' + arg.mode + '.npy'
    model, classifier, criterion = set_augment_model(arg.enc, arg.mode)

    print("Mode: ", arg.mode)
    print("Encoder: ", arg.enc)
    print("Attack: ", arg.attack)

    if arg.attack == 'pgd':

        dataloader = None
        model_params = []
        model_params += model.parameters()
        base_optimizer = optim.SGD(model_params, lr=0.2, momentum=0.9, weight_decay=1e-6)
        optimizer = LARS(optimizer=base_optimizer, eps=1e-8, trust_coef=0.001)
        fgsm_data = []

        if arg.mode == 'train':
            X_train, y_train, musk_train = load_train_images(device)
            train_data = [[X_train[i], y_train[i]] for i in range(y_train.size()[0])]
            train_dataloader = DataLoader(train_data, batch_size=batch, shuffle=False)
            dataloader = train_dataloader
            attack_1 = RepresentationAdv(model, epsilon=0.0314, alpha=0.007)

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

        if arg.mode == 'tune':
            X_train, y_train, musk_train = load_train_images(device)
            train_data = [[X_train[i], y_train[i]] for i in range(y_train.size()[0])]
            train_dataloader = DataLoader(train_data, batch_size=batch, shuffle=False)
            dataloader = train_dataloader
            attack_1 = FastGradientSignUntargeted(model, classifier)
            for idx, (images, labels) in enumerate(dataloader):
                images = augment(images)
                print("Batch ", idx)
                if torch.cuda.is_available():
                    images[0] = images[0].cuda(non_blocking=True)
                    images[1] = images[1].cuda(non_blocking=True)
                    labels = labels.cuda(non_blocking=True)
                bsz = labels.shape[0]
                advinputs = attack_1.perturb(original_images=images[0], labels=labels, random_start=True)
                # print(type(adv_1[0]), adv_1[0].size())
                # print(adv_1.cpu().numpy())
                fgsm_data.extend(advinputs.cpu().numpy())

            fgsm_data = np.array(fgsm_data)
            with open(data_path, 'wb') as f:
                np.save(f, fgsm_data, allow_pickle=False)
            # np.save('adv_dataset/fgsm_data_test.npy', fgsm_data, allow_plickle=False)
            print("Number of perturbed samples: ", len(fgsm_data))

        if arg.mode == 'test':
            X_test, y_test, musk_test = load_test_images(device)
            test_data = [[X_test[i], y_test[i]] for i in range(y_test.size()[0])]
            test_dataloader = DataLoader(test_data, batch_size=batch, shuffle=False)
            dataloader = test_dataloader
            attack_1 = FastGradientSignUntargeted(model, classifier)
            for idx, (images, labels) in enumerate(dataloader):
                images = augment(images)
                print("Batch ", idx)
                if torch.cuda.is_available():
                    images[0] = images[0].cuda(non_blocking=True)
                    images[1] = images[1].cuda(non_blocking=True)
                    labels = labels.cuda(non_blocking=True)
                bsz = labels.shape[0]
                advinputs = attack_1.perturb(original_images=images[0], labels=labels, random_start=True)
                # print(type(adv_1[0]), adv_1[0].size())
                # print(adv_1.cpu().numpy())
                fgsm_data.extend(advinputs.cpu().numpy())

            fgsm_data = np.array(fgsm_data)
            with open(data_path, 'wb') as f:
                np.save(f, fgsm_data, allow_pickle=False)
            # np.save('adv_dataset/fgsm_data_test.npy', fgsm_data, allow_plickle=False)
            print("Number of perturbed samples: ", len(fgsm_data))

    if arg.attack == 'otsa':

        dataloader = None

        if arg.mode == 'train' or 'tune':
            X_train, y_train, musk_train = load_train_images(device)
            train_data = [[X_train[i], y_train[i], musk_train[i]] for i in range(y_train.size()[0])]
            dataloader = DataLoader(train_data, batch_size=batch, shuffle=False)

        if arg.mode == 'test':
            X_test, y_test, musk_test = load_test_images(device)
            test_data = [[X_test[i], y_test[i], musk_test[i]] for i in range(y_test.size()[0])]
            dataloader = DataLoader(test_data, batch_size=batch, shuffle=False)

        attack_2 = OTSA(0.07)
        adv_2, adv_2_filtered = attack_2.generate(model, criterion, dataloader, batch=batch, linear = classifier)

        with open(data_path, 'wb') as f:
            np.save(f, adv_2_filtered, allow_pickle=False)
        print("Number of samples: ", len(adv_2_filtered))
