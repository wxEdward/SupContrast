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

target = 'aconvnet'
torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(surrogate, checkpoint_path, adv_path, gt_path):
    #device = "cpu"
    summary = []
    model, _, _, _ = load_checkpoint(device, checkpoint_path)
    testloader = load_adv_images(device, adv_path, gt_path)
    model.eval()
    num_correct = 0
    num_test = 0
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
        summary.append(correctness.squeeze().cpu().numpy())
        num_correct += torch.sum(correctness).item()
        num_test += labels.size()[0]

    print("Accuracy: {} / {} = {}".format(num_correct, num_test, num_correct/num_test))
    summary = np.hstack(summary)
    np.savetxt('../advtest/advcorrect_{}_{}.txt'.format(surrogate, target), summary)

#if __name__ == '__main__':
def test_all(path):
    checkpoint_path = "./models/aconvnet-checkpoint-150.pt"
    print(checkpoint_path)

    model_names = ["gnn", "vgg11", "alexnet", "squeezenet", "densenet121", "mobilenetv2", "resnet50", "shufflenetv2", "aconvnet"]
    clear_dataset = '/data/tian/MSTAR/dataset88/X_test_100.npy'
    clear_gt = '/data/tian/MSTAR/dataset88/y_test_100.npy'
    
    main('clear', checkpoint_path, clear_dataset, clear_gt)
    
    for surrogate in model_names:
        adv_path = '/data/tian/scatter_attack_blackbox_88/{}/adv_images_obj_{}.npy'.format(path, surrogate)
        gt_path = '/data/tian/scatter_attack_blackbox_88/{}/adv_gt_{}.npy'.format(path, surrogate)
        print(adv_path)
        print(gt_path)
        try:
            main(surrogate, checkpoint_path, adv_path, gt_path)
        except Exception as e:
            print(e)

def test_one(path, surrogate):
    checkpoint_path = "./models/aconvnet-checkpoint-150.pt"
    print(checkpoint_path)

    adv_path = '/data/tian/scatter_attack_blackbox_88/{}/adv_images_obj_{}.npy'.format(path, surrogate)
    gt_path = '/data/tian/scatter_attack_blackbox_88/{}/adv_gt_{}.npy'.format(path, surrogate)
    print(adv_path)
    print(gt_path)
    try:
        main(surrogate, checkpoint_path, adv_path, gt_path)
    except Exception as e:
        print(e)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--surrogate', help='Surrogate model')
    parser.add_argument('--path', help='Path to the adversarial samples')
    FLAGS = parser.parse_args()

    if FLAGS.path is None:
        path = 'adv_dataset_obj'
    else:
        path = FLAGS.path

    if FLAGS.surrogate is None:
        test_all(path)
    else:
        test_one(path, FLAGS.surrogate)

