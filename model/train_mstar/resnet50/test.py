import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from util import new_model, load_model, load_checkpoint, load_images

#model_path = './models/resnet50-checkpoint-180.pt'

torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

trainloader, testloader = load_images(device)


def main(model_path, verbose=0):
    model, _, _, _ = load_checkpoint(device, model_path) 
    model.eval()

    num_correct = 0
    num_test = 0
    if verbose:
        log = np.array([])
    for i, data in enumerate(testloader):
        # data: batch of 64 * [image, label]
        images, labels = data
        outputs = model(images)
        predictions = outputs.max(1, keepdim=True)[1].squeeze()
        num_correct += torch.sum(predictions == labels).item()
        num_test += labels.size()[0]
        
        if verbose:
            log_batch = (predictions==labels).cpu().detach().numpy()
            log = np.concatenate([log, log_batch])
    
    if verbose:
        np.savetxt(model_path[:-3]+'-results.txt', log)

    print("Accuracy: {} / {} = {}".format(num_correct, num_test, num_correct/num_test))

#if __name__ == '__main__':
#    for step in range(10, 201, 10):
#        model_path = './models/resnet50-checkpoint-{}.pt'.format(step)
#        print(model_path)
#        main(model_path)

model_path = './models/resnet50-checkpoint-110.pt'
print(model_path)
main(model_path, 1)


