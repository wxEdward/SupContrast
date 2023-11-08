import torch
import numpy as np
from torch import pi, exp, sinc, sin, cos, tan, arctan, sqrt
from scipy.signal.windows import taylor
from torch.fft import ifft2 as ifft2
from matplotlib import pyplot as plt
import torch.nn.functional as F
import torch_geometric

from matplotlib.image import imsave
from time import time

from util import load_checkpoint
from advutil import load_adv_images


#checkpoint_path = "./models/gnn-checkpoint-1000.pt"

target = 'resnet50'
torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(model, adv_path, gt_path):
    testloader = load_adv_images(device, adv_path, gt_path)
    num_test = 0
    num_correct = 0
    for i, data in enumerate(testloader):
        # data: batch of 64 * [image, label]
        images, labels = data
        outputs = model(images)
        predictions = outputs.max(1, keepdim=True)[1].squeeze()

        labels = labels.squeeze()
        print(predictions)
        print(labels)
        print(labels.size())

        correctness = predictions == labels
        num_correct += torch.sum(correctness).item() 
        num_test += labels.size()[0]

    print("Accuracy: {} / {} = {}".format(num_correct, num_test, num_correct/num_test))


def test_fromfile(config):
    checkpoint_path = "./models/resnet50-checkpoint-110.pt"
    print(checkpoint_path)
    model, _, _, _ = load_checkpoint(device, checkpoint_path)
    model.eval()
    
    f = open(config, 'r')
    filenames = f.read().split('\n')
    f.close()
   
    adv_filenames = filenames[::2]
    gt_filenames = filenames[1::2]

    for adv_filename, gt_filename in zip(adv_filenames, gt_filenames):
        adv_path = '/data/tian/'+adv_filename
        gt_path = '/data/tian/'+gt_filename
        print("Adv path:", adv_path)
        print('Label path:', gt_path)
        #try:
        main(model, adv_path, gt_path)
        #except Exception as e:
        #    print(e)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Path to the config file')
    FLAGS = parser.parse_args()
    test_fromfile(FLAGS.config)

