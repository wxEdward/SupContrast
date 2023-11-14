from torchvision import transforms
import torch
import numpy as np
from scipy.io import loadmat
# from main_supcon import set_model
from attacks.otsa import OTSA
from util import load_train_images, load_test_images
import torch.backends.cudnn as cudnn
from networks.resnet_big import SupConResNet
from losses import SupConLoss
# from otsa import OTSA


class Augmentation():
    def __init__(self) -> None:
        
        #normalize = transforms.Normalize(mean=mean, std=std)

        self. train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        # transforms.ToTensor(),
        # normalize,
        ])

    def __call__(self, x):
        return self.train_transform(x)
    

def set_augment_model():

    model = SupConResNet(name='resnet50')
    criterion = SupConLoss(temperature=0.07)

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
    
if __name__ == '__main__':
    device = torch.device("cuda")
    
    augment = Augmentation()

    X_train, y_train, musk_train = load_train_images(device)

    y_train = y_train.squeeze()

    X_train_image = X_train
    #X_test_image = X_test

    X_train_image = (X_train_image - X_train_image.min(axis=(1, 2)).reshape([-1, 1, 1])) / X_train_image.ptp(
        axis=(1, 2)).reshape([-1, 1, 1])
    print(np.min(X_train_image), np.max(X_train_image))
    X_train_image = X_train_image[:, np.newaxis, :, :]
    X_train_image = torch.from_numpy(X_train_image).to(device)
    X_train_image = X_train_image.type(torch.float32)

    #X_test_image = (X_test_image - X_test_image.min(axis=(1, 2)).reshape([-1, 1, 1])) / X_test_image.ptp(
        #axis=(1, 2)).reshape([-1, 1, 1])
    #print(np.min(X_test_image), np.max(X_test_image))
    #X_test_image = X_test_image[:, np.newaxis, :, :]
    #X_test_image = torch.from_numpy(X_test_image).to(device)
    #X_test_image = X_test_image.type(torch.float32)

    if y_train.min().item == 1 and y_train.max().item() == 10:
        train_label = torch.from_numpy(y_train).to(device) - 1
    else:
        train_label = torch.from_numpy(y_train).to(device)
    train_label = train_label.type(torch.int64)
    '''

    if y_test.min().item == 1 and y_test.max().item() == 10:
        test_label = torch.from_numpy(y_test).to(device) - 1
    else:
        test_label = torch.from_numpy(y_test).to(device)
    test_label = test_label.type(torch.int64)
    '''
    # y_attack_target = np.ones(y_test.shape)
    # y_attack_target = np.where(y_test != y_attack_target, y_attack_target, y_attack_target+1)
    # test_attack_target = torch.from_numpy(y_attack_target).to(device) - 1
    # test_attack_target = test_attack_target.type(torch.int64)

    print(X_train_image.size(), train_label.size())

    #augment = TwoCropTransform(train_transform, model, opt)
    X_train_augmented = augment(X_train_image)
    #X_test_augmented = augment(X_test_image,test_label, musk_test)

    train_data = [[X_train_augmented[i], train_label[i]] for i in range(train_label.size()[0])]
    #test_data = [[X_test_augmented [i], test_label[i]] for i in range(test_label.size()[0])]

    #train_dataloader = DataLoader(train_data, batch_size=64,
                                  #shuffle=True)

    #return train_dataloader, test_dataloader
    print(len(train_data))
    
    model, criterion = set_augment_model()

    attack_2 = OTSA(0.07)

    adv_2, adv_2_filtered = attack_2.generate(model,criterion, X_train_augmented, y_train, musk_train)
    print(len(adv_2), len(adv_2_filtered))




