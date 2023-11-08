import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

from util import new_model, load_model, load_checkpoint, load_images
from fgsm import FGSM

#model_path = './models/aconvnet-checkpoint-180.pt'

torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

trainloader, testloader = load_images(device)


def main(model_path, verbose=0):
    model, _, _, _ = load_checkpoint(device, model_path) 
    model.eval()

    fgsm = FGSM(device)

    criterion = torch.nn.CrossEntropyLoss()
    epsilon = 8/255

    num_correct = 0
    num_total = 0
    pbar = tqdm(testloader)

    images_fgsm = []
    correct_labels = []
    wrong_labels = []
    for i, (inputs, labels) in enumerate(pbar, 1):
        inputs = inputs.to(device)
        inputs = fgsm.generate(model, inputs, labels, criterion, epsilon)
        images_fgsm.append(inputs.detach().cpu())

        out = model(inputs)
        pred = out.argmax(axis=1)
        num_correct += (pred == labels).sum()
        num_total += pred.size(0)

        correct_labels.append(labels)
        wrong_labels.append(pred)

    correct_labels = torch.cat(correct_labels)
    wrong_labels = torch.cat(wrong_labels)
    fgsm_results = torch.cat([correct_labels.view(-1,1), wrong_labels.view(-1,1)], dim=1)
    fgsm_results = fgsm_results.detach().cpu().numpy()


    print("Accuracy: {} / {} = {}".format(num_correct, num_total, num_correct/num_total))



if __name__ == '__main__':
    model_path = './models/aconvnet-checkpoint-150.pt'
    print(model_path)
    main(model_path, verbose=1)


