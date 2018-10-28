import os
import torch
import torchvision
import torchvision.transforms as transforms
import math
import torch.optim as optim
#import torch.optim.lr_scheduler as lr_scheduler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import time


class discriminator(nn.Module):
    def __init__(self, AC = True):
        super(discriminator, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 196, 3, stride=1, padding=1),
            nn.LayerNorm([32,32]),
            nn.LeakyReLU(),
            nn.Conv2d(196, 196, 3, stride=2, padding=1),
            nn.LayerNorm([16,16]),
            nn.LeakyReLU(),
            nn.Conv2d(196, 196, 3, stride=1, padding=1),
            nn.LayerNorm([16,16]),
            nn.LeakyReLU(),
            nn.Conv2d(196, 196, 3, stride=2, padding=1),
            nn.LayerNorm([8,8]),
            nn.LeakyReLU(),
            nn.Conv2d(196, 196, 3, stride=1, padding=1),
            nn.LayerNorm([8,8]),
            nn.LeakyReLU(),
            nn.Conv2d(196, 196, 3, stride=1, padding=1),
            nn.LayerNorm([8,8]),
            nn.LeakyReLU(),
            nn.Conv2d(196, 196, 3, stride=1, padding=1),
            nn.LayerNorm([8,8]),
            nn.LeakyReLU(),
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

# Create Directory
if os.path.exists('./visualization'):
    print('visualization exist')
else:
    print('visualization dir does not exist, create one')
    os.mkdir('visualization')


def plot(samples):
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(10, 10)
    gs.update(wspace=0.02, hspace=0.02)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample)
    return fig


batch_size = 128

transform_test = transforms.Compose([
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

testset = torchvision.datasets.CIFAR10(root='./', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)
testloader = enumerate(testloader)

model = torch.load('cifar10.model')
model.cuda()
model.eval()

batch_idx, (X_batch, Y_batch) = next(testloader)
X_batch = Variable(X_batch,requires_grad=True).cuda()
Y_batch_alternate = (Y_batch + 1)%10
Y_batch_alternate = Variable(Y_batch_alternate).cuda()
Y_batch = Variable(Y_batch).cuda()

## save real images
samples = X_batch.data.cpu().numpy()
samples += 1.0
samples /= 2.0
samples = samples.transpose(0,2,3,1)

fig = plot(samples[0:100])
plt.savefig('visualization/real_images.png', bbox_inches='tight')
plt.close(fig)


# Get the output from the fc10 layer and report the classification accuracy.
_, output = model(X_batch)
prediction = output.data.max(1)[1] # first column has actual prob.
accuracy = ( float( prediction.eq(Y_batch.data).sum() ) /float(batch_size))*100.0
print('Model Accuracy %f%%' %accuracy)



## slightly jitter all input images
criterion = nn.CrossEntropyLoss(reduce=False)
loss = criterion(output, Y_batch_alternate)

gradients = torch.autograd.grad(outputs=loss, inputs=X_batch,
                          grad_outputs=torch.ones(loss.size()).cuda(),
                          create_graph=True, retain_graph=False, only_inputs=True)[0]

# save gradient jitter
gradient_image = gradients.data.cpu().numpy()
gradient_image = (gradient_image - np.min(gradient_image))/(np.max(gradient_image)-np.min(gradient_image))
gradient_image = gradient_image.transpose(0,2,3,1)
fig = plot(gradient_image[0:100])
plt.savefig('visualization/gradient_image.png', bbox_inches='tight')
plt.close(fig)

# jitter input image
gradients[gradients>0.0] = 1.0
gradients[gradients<0.0] = -1.0

gain = 8.0
X_batch_modified = X_batch - gain*0.007843137*gradients
X_batch_modified[X_batch_modified>1.0] = 1.0
X_batch_modified[X_batch_modified<-1.0] = -1.0

## evaluate new fake images
_, output = model(X_batch_modified)
prediction = output.data.max(1)[1] # first column has actual prob.
accuracy = ( float( prediction.eq(Y_batch.data).sum() ) /float(batch_size))*100.0
print('Jittered Accuracy %f%%' %accuracy)

## save fake images
samples = X_batch_modified.data.cpu().numpy()
samples += 1.0
samples /= 2.0
samples = samples.transpose(0,2,3,1)

fig = plot(samples[0:100])
plt.savefig('visualization/jittered_images.png', bbox_inches='tight')
plt.close(fig)


####################################################################################
####################################################################################
####################################################################################




# Discriminator with out generator
model = torch.load('cifar10.model')
model.cuda()
model.eval()

X = X_batch.mean(dim=0)
X = X.repeat(10,1,1,1)

Y = torch.arange(10).type(torch.int64)
Y = Variable(Y).cuda()

lr = 0.1
weight_decay = 0.001
for i in range(200):
    _, output = model(X)

    loss = -output[torch.arange(10).type(torch.int64),torch.arange(10).type(torch.int64)]
    gradients = torch.autograd.grad(outputs=loss, inputs=X,
                              grad_outputs=torch.ones(loss.size()).cuda(),
                              create_graph=True, retain_graph=False, only_inputs=True)[0]

    prediction = output.data.max(1)[1] # first column has actual prob.
    accuracy = ( float( prediction.eq(Y.data).sum() ) /float(10.0))*100.0
    print(i,accuracy,-loss)

    X = X - lr*gradients.data - weight_decay*X.data*torch.abs(X.data)
    X[X>1.0] = 1.0
    X[X<-1.0] = -1.0

## save new images
samples = X.data.cpu().numpy()
samples += 1.0
samples /= 2.0
samples = samples.transpose(0,2,3,1)

fig = plot(samples)
plt.savefig('visualization/max_class_no_G.png', bbox_inches='tight')
plt.close(fig)





# discriminator with Generator
class discriminator(nn.Module):
    def __init__(self, AC = True):
        super(discriminator, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 196, 3, stride=1, padding=1),
            nn.LayerNorm([32,32]),
            nn.LeakyReLU(),
            nn.Conv2d(196, 196, 3, stride=2, padding=1),
            nn.LayerNorm([16,16]),
            nn.LeakyReLU(),
            nn.Conv2d(196, 196, 3, stride=1, padding=1),
            nn.LayerNorm([16,16]),
            nn.LeakyReLU(),
            nn.Conv2d(196, 196, 3, stride=2, padding=1),
            nn.LayerNorm([8,8]),
            nn.LeakyReLU(),
            nn.Conv2d(196, 196, 3, stride=1, padding=1),
            nn.LayerNorm([8,8]),
            nn.LeakyReLU(),
            nn.Conv2d(196, 196, 3, stride=1, padding=1),
            nn.LayerNorm([8,8]),
            nn.LeakyReLU(),
            nn.Conv2d(196, 196, 3, stride=1, padding=1),
            nn.LayerNorm([8,8]),
            nn.LeakyReLU(),
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

try: 
	model = torch.load('discriminator.model')
except:
	model = torch.load('tempD.model')

model.cuda()
model.eval()
X = X_batch.mean(dim=0)
X = X.repeat(10,1,1,1)

Y = torch.arange(10).type(torch.int64)
Y = Variable(Y).cuda()
lr = 0.1
weight_decay = 0.001
for i in range(200):
    _, output = model(X)

    loss = -output[torch.arange(10).type(torch.int64),torch.arange(10).type(torch.int64)]
    gradients = torch.autograd.grad(outputs=loss, inputs=X,
                              grad_outputs=torch.ones(loss.size()).cuda(),
                              create_graph=True, retain_graph=False, only_inputs=True)[0]

    prediction = output.data.max(1)[1] # first column has actual prob.
    accuracy = ( float( prediction.eq(Y.data).sum() ) /float(10.0))*100.0
    print(i,accuracy,-loss)

    X = X - lr*gradients.data - weight_decay*X.data*torch.abs(X.data)
    X[X>1.0] = 1.0
    X[X<-1.0] = -1.0

## save new images
samples = X.data.cpu().numpy()
samples += 1.0
samples /= 2.0
samples = samples.transpose(0,2,3,1)

fig = plot(samples)
plt.savefig('visualization/max_class_with_G.png', bbox_inches='tight')
plt.close(fig)


####################################################################################
####################################################################################
####################################################################################




# Synthetic Features Maximizing Features at Various Layers
# Do 2 layers for distriminator model without Generator
# Modifying the model
# Since we trained on a sequential block, we have to load model and load the state of the parameters.


# Load Model 
class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 196, 3, stride=1, padding=1),
            nn.LayerNorm([32,32]),
            nn.LeakyReLU(),
            nn.Conv2d(196, 196, 3, stride=2, padding=1),
            nn.LayerNorm([16,16]),
            nn.LeakyReLU(),
            nn.Conv2d(196, 196, 3, stride=1, padding=1),
            nn.LayerNorm([16,16]),
            nn.LeakyReLU(),
            nn.Conv2d(196, 196, 3, stride=2, padding=1),
            nn.LayerNorm([8,8]),
            nn.LeakyReLU(),
            nn.Conv2d(196, 196, 3, stride=1, padding=1),
            nn.LayerNorm([8,8]),
            nn.LeakyReLU(),
            nn.Conv2d(196, 196, 3, stride=1, padding=1),
            nn.LayerNorm([8,8]),
            nn.LeakyReLU(),
            nn.Conv2d(196, 196, 3, stride=1, padding=1),
            nn.LayerNorm([8,8]),
            nn.LeakyReLU(),
            nn.Conv2d(196, 196, 3, stride=2, padding=1),
            nn.LayerNorm([4,4]),
            nn.LeakyReLU(),
            nn.MaxPool2d(4,4),
        )

        self.fc1 = nn.Linear(196,1)
        self.fc10 = nn.Linear(196,10)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1,196)
        return self.fc1(x), self.fc10(x)


class discriminator2(nn.Module):
    def __init__(self):
        super(discriminator2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 196, 3, stride=1, padding=1),
            nn.LayerNorm([32,32]),
            nn.LeakyReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(196, 196, 3, stride=2, padding=1),
            nn.LayerNorm([16,16]),
            nn.LeakyReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(196, 196, 3, stride=1, padding=1),
            nn.LayerNorm([16,16]),
            nn.LeakyReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(196, 196, 3, stride=2, padding=1),
            nn.LayerNorm([8,8]),
            nn.LeakyReLU(),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(196, 196, 3, stride=1, padding=1),
            nn.LayerNorm([8,8]),
            nn.LeakyReLU(),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(196, 196, 3, stride=1, padding=1),
            nn.LayerNorm([8,8]),
            nn.LeakyReLU(),
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(196, 196, 3, stride=1, padding=1),
            nn.LayerNorm([8,8]),
            nn.LeakyReLU(),
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(196, 196, 3, stride=2, padding=1),
            nn.LayerNorm([4,4]),
            nn.LeakyReLU(),
            nn.MaxPool2d(4,4),
        )
        
        # self.max_pool2d = 
        self.fc1 = nn.Linear(196,1)
        self.fc10 = nn.Linear(196,10)

    
    def forward(self, x, extract_features):
        x = self.conv1(x)
        if extract_features == 1:
            x = F.max_pool2d(x, 32, 32)
            x = x.view(-1, 196)
            return x
        x = self.conv2(x)
        if extract_features == 2:
            # print('HERE')
            x = F.max_pool2d(x, 16, 16)
            x = x.view(-1, 196)
            return x
        x = self.conv3(x)
        if extract_features == 3:
            x = F.max_pool2d(x, 16, 16)
            x = x.view(-1, 196)
            return x
        x = self.conv4(x)
        if extract_features == 3:
            x = F.max_pool2d(x, 8, 8)
            x = x.view(-1, 196)
            return x 
        x = self.conv5(x)
        if extract_features == 5:
            x = F.max_pool2d(x, 8, 8)
            x = x.view(-1, 196)
            return x 
        x = self.conv6(x)
        if extract_features == 6:
            x = F.max_pool2d(x, 8, 8)
            x = x.view(-1, 196)
            return x 
        x = self.conv7(x)
        if extract_features == 7:
            x = F.max_pool2d(x, 8, 8)
            x = x.view(-1, 196)
            return x 
        x = self.conv8(x)
        x = x.view(-1,196)
        return x


model = torch.load('cifar10.model', map_location='cpu')

model_dic = copy.deepcopy(model.state_dict())
# The names doesn't match, we change the name
keys = []
for i in range(1,9):
    w1 = 'conv'+str(i)+'.0.weight'
    b1 = 'conv'+str(i)+'.0.bias'
    w2 = 'conv'+str(i)+'.1.weight'
    b2 = 'conv'+str(i)+'.1.bias'
    keys.extend([w1,b1,w2,b2])


model_dic_keys = list(model_dic.keys())
for x in zip(keys, model_dic_keys):
    model_dic[x[0]] = model_dic.pop(x[1])


model2 = discriminator2()
model2.load_state_dict(model_dic)
# for param_tensor in model2.state_dict():
#     print(param_tensor, "\t", model2.state_dict()[param_tensor].size())
for i in range(len(model.state_dict())):
    assert ((model.state_dict()[list(model.state_dict().keys())[i]] == model2.state_dict()[list(model2.state_dict().keys())[i]]).byte().all())

model2
model2.cuda()
model2.eval()

####################################################################################
# Do for layer 2
batch_size = 196

transform_test = transforms.Compose([
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

testset = torchvision.datasets.CIFAR10(root='./', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)
testloader = enumerate(testloader)

batch_idx, (X_batch, Y_batch) = next(testloader)
X_batch = Variable(X_batch,requires_grad=True).cuda()
Y_batch_alternate = (Y_batch + 1)%10
Y_batch_alternate = Variable(Y_batch_alternate).cuda()
Y_batch = Variable(Y_batch).cuda()

X = X_batch.mean(dim=0)
X = X.repeat(batch_size,1,1,1)

Y = torch.arange(batch_size).type(torch.int64)
Y = Variable(Y).cuda()

lr = 0.1
weight_decay = 0.001
for i in range(200):
    # Extract_features == 2
    output = model2(X, 2)

    loss = -output[torch.arange(batch_size).type(torch.int64),torch.arange(batch_size).type(torch.int64)]
    gradients = torch.autograd.grad(outputs=loss, inputs=X,
                              grad_outputs=torch.ones(loss.size()).cuda(),
                              create_graph=True, retain_graph=False, only_inputs=True)[0]

    prediction = output.data.max(1)[1] # first column has actual prob.
    accuracy = ( float( prediction.eq(Y.data).sum() ) /float(batch_size))*100.0
    print(i,accuracy,-loss)

    X = X - lr*gradients.data - weight_decay*X.data*torch.abs(X.data)
    X[X>1.0] = 1.0
    X[X<-1.0] = -1.0

## save new images
samples = X.data.cpu().numpy()
samples += 1.0
samples /= 2.0
samples = samples.transpose(0,2,3,1)

fig = plot(samples[0:100])
plt.savefig('visualization/max_features_without_G_Layer2.png', bbox_inches='tight')
plt.close(fig)

####################################################################################


####################################################################################
# Do for layer 8
batch_size = 196

transform_test = transforms.Compose([
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

testset = torchvision.datasets.CIFAR10(root='./', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)
testloader = enumerate(testloader)

batch_idx, (X_batch, Y_batch) = next(testloader)
X_batch = Variable(X_batch,requires_grad=True).cuda()
Y_batch_alternate = (Y_batch + 1)%10
Y_batch_alternate = Variable(Y_batch_alternate).cuda()
Y_batch = Variable(Y_batch).cuda()

X = X_batch.mean(dim=0)
X = X.repeat(batch_size,1,1,1)

Y = torch.arange(batch_size).type(torch.int64)
Y = Variable(Y).cuda()

lr = 0.1
weight_decay = 0.001
for i in range(200):
    # Extract_features == 2
    output = model2(X, 8)

    loss = -output[torch.arange(batch_size).type(torch.int64),torch.arange(batch_size).type(torch.int64)]
    gradients = torch.autograd.grad(outputs=loss, inputs=X,
                              grad_outputs=torch.ones(loss.size()).cuda(),
                              create_graph=True, retain_graph=False, only_inputs=True)[0]

    prediction = output.data.max(1)[1] # first column has actual prob.
    accuracy = ( float( prediction.eq(Y.data).sum() ) /float(batch_size))*100.0
    print(i,accuracy,-loss)

    X = X - lr*gradients.data - weight_decay*X.data*torch.abs(X.data)
    X[X>1.0] = 1.0
    X[X<-1.0] = -1.0

## save new images
samples = X.data.cpu().numpy()
samples += 1.0
samples /= 2.0
samples = samples.transpose(0,2,3,1)

fig = plot(samples[0:100])
plt.savefig('visualization/max_features_without_G_Layer8.png', bbox_inches='tight')
plt.close(fig)

####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################


# Redo above for discriminator with Generator

# Load Model 
class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 196, 3, stride=1, padding=1),
            nn.LayerNorm([32,32]),
            nn.LeakyReLU(),
            nn.Conv2d(196, 196, 3, stride=2, padding=1),
            nn.LayerNorm([16,16]),
            nn.LeakyReLU(),
            nn.Conv2d(196, 196, 3, stride=1, padding=1),
            nn.LayerNorm([16,16]),
            nn.LeakyReLU(),
            nn.Conv2d(196, 196, 3, stride=2, padding=1),
            nn.LayerNorm([8,8]),
            nn.LeakyReLU(),
            nn.Conv2d(196, 196, 3, stride=1, padding=1),
            nn.LayerNorm([8,8]),
            nn.LeakyReLU(),
            nn.Conv2d(196, 196, 3, stride=1, padding=1),
            nn.LayerNorm([8,8]),
            nn.LeakyReLU(),
            nn.Conv2d(196, 196, 3, stride=1, padding=1),
            nn.LayerNorm([8,8]),
            nn.LeakyReLU(),
            nn.Conv2d(196, 196, 3, stride=2, padding=1),
            nn.LayerNorm([4,4]),
            nn.LeakyReLU(),
            nn.MaxPool2d(4,4),
        )

        self.fc1 = nn.Linear(196,1)
        self.fc10 = nn.Linear(196,10)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1,196)
        return self.fc1(x), self.fc10(x)


class discriminator2(nn.Module):
    def __init__(self):
        super(discriminator2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 196, 3, stride=1, padding=1),
            nn.LayerNorm([32,32]),
            nn.LeakyReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(196, 196, 3, stride=2, padding=1),
            nn.LayerNorm([16,16]),
            nn.LeakyReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(196, 196, 3, stride=1, padding=1),
            nn.LayerNorm([16,16]),
            nn.LeakyReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(196, 196, 3, stride=2, padding=1),
            nn.LayerNorm([8,8]),
            nn.LeakyReLU(),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(196, 196, 3, stride=1, padding=1),
            nn.LayerNorm([8,8]),
            nn.LeakyReLU(),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(196, 196, 3, stride=1, padding=1),
            nn.LayerNorm([8,8]),
            nn.LeakyReLU(),
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(196, 196, 3, stride=1, padding=1),
            nn.LayerNorm([8,8]),
            nn.LeakyReLU(),
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(196, 196, 3, stride=2, padding=1),
            nn.LayerNorm([4,4]),
            nn.LeakyReLU(),
            nn.MaxPool2d(4,4),
        )
        
        # self.max_pool2d = 
        self.fc1 = nn.Linear(196,1)
        self.fc10 = nn.Linear(196,10)

    
    def forward(self, x, extract_features):
        x = self.conv1(x)
        if extract_features == 1:
            x = F.max_pool2d(x, 32, 32)
            x = x.view(-1, 196)
            return x
        x = self.conv2(x)
        if extract_features == 2:
            # print('HERE')
            x = F.max_pool2d(x, 16, 16)
            x = x.view(-1, 196)
            return x
        x = self.conv3(x)
        if extract_features == 3:
            x = F.max_pool2d(x, 16, 16)
            x = x.view(-1, 196)
            return x
        x = self.conv4(x)
        if extract_features == 3:
            x = F.max_pool2d(x, 8, 8)
            x = x.view(-1, 196)
            return x 
        x = self.conv5(x)
        if extract_features == 5:
            x = F.max_pool2d(x, 8, 8)
            x = x.view(-1, 196)
            return x 
        x = self.conv6(x)
        if extract_features == 6:
            x = F.max_pool2d(x, 8, 8)
            x = x.view(-1, 196)
            return x 
        x = self.conv7(x)
        if extract_features == 7:
            x = F.max_pool2d(x, 8, 8)
            x = x.view(-1, 196)
            return x 
        x = self.conv8(x)
        x = x.view(-1,196)
        return x


model = torch.load('discriminator.model', map_location='cpu')

model_dic = copy.deepcopy(model.state_dict())
# The names doesn't match, we change the name
keys = []
for i in range(1,9):
    w1 = 'conv'+str(i)+'.0.weight'
    b1 = 'conv'+str(i)+'.0.bias'
    w2 = 'conv'+str(i)+'.1.weight'
    b2 = 'conv'+str(i)+'.1.bias'
    keys.extend([w1,b1,w2,b2])


model_dic_keys = list(model_dic.keys())
for x in zip(keys, model_dic_keys):
    model_dic[x[0]] = model_dic.pop(x[1])


model2 = discriminator2()
model2.load_state_dict(model_dic)
# for param_tensor in model2.state_dict():
#     print(param_tensor, "\t", model2.state_dict()[param_tensor].size())
for i in range(len(model.state_dict())):
    assert ((model.state_dict()[list(model.state_dict().keys())[i]] == model2.state_dict()[list(model2.state_dict().keys())[i]]).byte().all())

model2
model2.cuda()
model2.eval()

####################################################################################
# Do for layer 2
batch_size = 196

transform_test = transforms.Compose([
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

testset = torchvision.datasets.CIFAR10(root='./', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)
testloader = enumerate(testloader)

batch_idx, (X_batch, Y_batch) = next(testloader)
X_batch = Variable(X_batch,requires_grad=True).cuda()
Y_batch_alternate = (Y_batch + 1)%10
Y_batch_alternate = Variable(Y_batch_alternate).cuda()
Y_batch = Variable(Y_batch).cuda()

X = X_batch.mean(dim=0)
X = X.repeat(batch_size,1,1,1)

Y = torch.arange(batch_size).type(torch.int64)
Y = Variable(Y).cuda()

lr = 0.1
weight_decay = 0.001
for i in range(200):
    # Extract_features == 2
    output = model2(X, 2)

    loss = -output[torch.arange(batch_size).type(torch.int64),torch.arange(batch_size).type(torch.int64)]
    gradients = torch.autograd.grad(outputs=loss, inputs=X,
                              grad_outputs=torch.ones(loss.size()).cuda(),
                              create_graph=True, retain_graph=False, only_inputs=True)[0]

    prediction = output.data.max(1)[1] # first column has actual prob.
    accuracy = ( float( prediction.eq(Y.data).sum() ) /float(batch_size))*100.0
    print(i,accuracy,-loss)

    X = X - lr*gradients.data - weight_decay*X.data*torch.abs(X.data)
    X[X>1.0] = 1.0
    X[X<-1.0] = -1.0

## save new images
samples = X.data.cpu().numpy()
samples += 1.0
samples /= 2.0
samples = samples.transpose(0,2,3,1)

fig = plot(samples[0:100])
plt.savefig('visualization/max_features_without_G_Layer2.png', bbox_inches='tight')
plt.close(fig)

####################################################################################


####################################################################################
# Do for layer 8
batch_size = 196

transform_test = transforms.Compose([
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

testset = torchvision.datasets.CIFAR10(root='./', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)
testloader = enumerate(testloader)

batch_idx, (X_batch, Y_batch) = next(testloader)
X_batch = Variable(X_batch,requires_grad=True).cuda()
Y_batch_alternate = (Y_batch + 1)%10
Y_batch_alternate = Variable(Y_batch_alternate).cuda()
Y_batch = Variable(Y_batch).cuda()

X = X_batch.mean(dim=0)
X = X.repeat(batch_size,1,1,1)

Y = torch.arange(batch_size).type(torch.int64)
Y = Variable(Y).cuda()

lr = 0.1
weight_decay = 0.001
for i in range(200):
    # Extract_features == 2
    output = model2(X, 8)

    loss = -output[torch.arange(batch_size).type(torch.int64),torch.arange(batch_size).type(torch.int64)]
    gradients = torch.autograd.grad(outputs=loss, inputs=X,
                              grad_outputs=torch.ones(loss.size()).cuda(),
                              create_graph=True, retain_graph=False, only_inputs=True)[0]

    prediction = output.data.max(1)[1] # first column has actual prob.
    accuracy = ( float( prediction.eq(Y.data).sum() ) /float(batch_size))*100.0
    print(i,accuracy,-loss)

    X = X - lr*gradients.data - weight_decay*X.data*torch.abs(X.data)
    X[X>1.0] = 1.0
    X[X<-1.0] = -1.0

## save new images
samples = X.data.cpu().numpy()
samples += 1.0
samples /= 2.0
samples = samples.transpose(0,2,3,1)

fig = plot(samples[0:100])
plt.savefig('visualization/max_features_without_G_Layer8.png', bbox_inches='tight')
plt.close(fig)

####################################################################################
