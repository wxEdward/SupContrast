import torch
import torch.nn as nn
import torch.optim as optim
from util import new_model, load_model, load_checkpoint, load_images

source = 'new'
#source = 'checkpoint'
checkpoint_step = 2000
checkpoint_path = './models/resnet50-checkpoint-{}.pt'.format(checkpoint_step)
model_path = './models/resnet50.pt'

torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

trainloader, testloader = load_images(device)

images_val, labels_val = next(iter(testloader))

old_epoch = 0

if source == 'checkpoint':
    print('Load from checkpoint')
    model, optimizer, old_epoch, epoch_loss = load_checkpoint(device, checkpoint_path)
elif source == 'saved':
    print('Load from saved model')
    model = load_model(device, model_path)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
else:
    print('Create new model')
    model = new_model(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

criterion = nn.NLLLoss()
#criterion = nn.CrossEntropyLoss()
num_epochs = 200
lambd = 0.005

for epoch in range(old_epoch + 1, num_epochs+1):
    epoch_loss = 0.0
    for i, data in enumerate(trainloader):
        # data: batch of 64 * [image, label]
        images, labels = data
        optimizer.zero_grad()
        outputs = model(images)
    
        #all_linear1_params = torch.cat([x.view(-1) for x in model.parameters()])
        #l1_reg = lambd * torch.norm(all_linear1_params, 1)
        loss = criterion(outputs, labels) #+ lambd * l1_reg
        
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    outputs = model(images_val)
    loss = criterion(outputs, labels_val)
    print("epoch: {}, train loss: {}, val loss: {}".format(epoch, epoch_loss, loss))
    f = open('train.log', 'a')
    f.write("epoch: {}, train loss: {}, val loss: {}".format(epoch, epoch_loss, loss))
    f.write("\n")
    f.close()

    if epoch % 10 == 0:
        checkpoint_path = 'models/resnet50-checkpoint-{}.pt'.format(epoch)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch_loss': epoch_loss
            }, checkpoint_path)

torch.save(model, model_path)

print('Finished Training')

