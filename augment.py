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
from losses import SupConLoss
# from otsa import OTSA
import torch.optim as optim
from torchlars import LARS
from torch.utils.data import DataLoader

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
    
def set_augment_loader():
    pass

from scipy.io import savemat
if __name__ == '__main__':
    device = torch.device("cuda")
    
    augment = Augmentation()

    X_train, y_train, musk_train = load_train_images(device)
    X_test, y_test, musk_test = load_test_images(device)

    print("training size:", X_train.size(), y_train.size())
    print("testing size:", X_test.size(), y_test.size())

    #augment = TwoCropTransform(train_transform, model, opt)
    #X_train_augmented = augment(X_train)
    #X_test_augmented = augment(X_test)

    #print("size after augmentation", len(X_train_augmented), X_train_augmented[0].size())
    train_data = [[X_train[i], y_train[i]] for i in range(y_train.size()[0])]
    test_data = [[X_test[i], y_test[i]] for i in range(y_test.size()[0])]

    train_dataloader = DataLoader(train_data, batch_size=64, shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False)

    
    model, criterion = set_augment_model()
    model_params = []
    model_params += model.parameters()
    attack_1 = RepresentationAdv(model, epsilon=0.0314, alpha=0.007)
    base_optimizer = optim.SGD(model_params, lr=0.2, momentum=0.9, weight_decay=1e-6)
    optimizer = LARS(optimizer=base_optimizer, eps=1e-8, trust_coef=0.001)

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
    print("Start for test dataset")

    attack_2 = OTSA(0.07)
    adv_2, adv_2_filtered = attack_2.generate(model, criterion, test_dataloader, batch=32)
    with open('adv_dataset/otsa_data_test.npy','wb') as f:
        np.save(f, adv_2, allow_pickle=False)
    with open('adv_dataset/otsa_data_test_filtered.npy','wb') as f:
        np.save(f, adv_2_filtered, allow_pickle=False)
    '''
    print(len(adv_2), len(adv_2_filtered))


    # print(len(adv_1))
