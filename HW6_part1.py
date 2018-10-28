import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class discriminator(nn.Module):
    def __init__(self, AC = True):
        super(discriminator, self).__init__()
        self.conv = nn.Sequential(
            # conv1
            nn.Conv2d(3, 196, 3, stride=1, padding=1),
            nn.LayerNorm([32,32]),
            nn.LeakyReLU(),
            # Conv2
            nn.Conv2d(196, 196, 3, stride=2, padding=1),
            nn.LayerNorm([16,16]),
            nn.LeakyReLU(),
            # Conv3
            nn.Conv2d(196, 196, 3, stride=1, padding=1),
            nn.LayerNorm([16,16]),
            nn.LeakyReLU(),
            # Conv4
            nn.Conv2d(196, 196, 3, stride=2, padding=1),
            nn.LayerNorm([8,8]),
            nn.LeakyReLU(),
            # Conv5
            nn.Conv2d(196, 196, 3, stride=1, padding=1),
            nn.LayerNorm([8,8]),
            nn.LeakyReLU(),
            # Conv6
            nn.Conv2d(196, 196, 3, stride=1, padding=1),
            nn.LayerNorm([8,8]),
            nn.LeakyReLU(),
            # conv7
            nn.Conv2d(196, 196, 3, stride=1, padding=1),
            nn.LayerNorm([8,8]),
            nn.LeakyReLU(),
            # conv8
            nn.Conv2d(196, 196, 3, stride=2, padding=1),
            nn.LayerNorm([4,4]),
            nn.LeakyReLU(),
            nn.MaxPool2d(4,4),
        )

        self.fc1 = nn.Linear(196,1)
        self.fc10 = nn.Linear(196,10)

    def forward(self,x):
        x = self.conv(x)
        x = x.view(-1,196)
        return self.fc1(x), self.fc10(x)

learning_rate = 0.0001
batch_size = 128
num_epoch = 100

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(32, scale=(0.7, 1.0), ratio=(1.0,1.0)),
    transforms.ColorJitter(
            brightness=0.1*torch.randn(1),
            contrast=0.1*torch.randn(1),
            saturation=0.1*torch.randn(1),
            hue=0.1*torch.randn(1)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

trainset = torchvision.datasets.CIFAR10(root='./', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR10(root='./', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)


model = discriminator()
model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)


for epoch in range(num_epoch):

    #change learning_rate
    if(epoch==50):
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate/10.0
    if(epoch==75):
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate/100.0

#Train
    model.train()
    correct, total = 0., 0.
    running_loss = 0.0
    for batch_idx, (X_train_batch, Y_train_batch) in enumerate(trainloader):

        if(Y_train_batch.shape[0] < batch_size):
            continue

        X_train_batch = Variable(X_train_batch).cuda()
        Y_train_batch = Variable(Y_train_batch).cuda()
        _, output = model(X_train_batch)

        loss = criterion(output, Y_train_batch)
        optimizer.zero_grad()
        loss.backward()

        if(epoch > 0):
            for group in optimizer.param_groups:
                for p in group['params']:
                    state = optimizer.state[p]
                    try:
                        if(state['step'] >= 1024):
                            state['step'] = 1000
                    except:
                        pass

        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(output, 1)
        total += Y_train_batch.size(0)
        correct += (predicted == Y_train_batch).sum()
    print('epoch: %d' % (epoch+1))
    print('Training Accuracy: %f' % (float(correct) / total))
    print('Loss: %f' % (running_loss))

#Test
    model.eval()
    correct, total = 0., 0.
    for batch_idx, (X_train_batch, Y_train_batch) in enumerate(testloader):
        X_train_batch = Variable(X_train_batch).cuda()
        Y_train_batch = Variable(Y_train_batch).cuda()
        _, output = model(X_train_batch)
        _, predicted = torch.max(output, 1)
        total += Y_train_batch.size(0)
        correct += (predicted == Y_train_batch).sum()
    test_acc = (float(correct) / total)
    print('Test Accuracy: %f' % (test_acc))

# if acc reached 89%, stop this train.
    if test_acc >= 0.89:
        print('Model reached 89% test accuracy! Break!')
        break

#save the trained model
torch.save(model,'cifar10.model')
