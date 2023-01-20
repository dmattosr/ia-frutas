# Importing libraries

# Basado en:
# https://github.com/IqraBaluch/Detection-of-Rotten-Fruits-DRF-Using-Image-Processing-Python/blob/master/Training.ipynb

from utils import (
    RUTA_DESTINO_TRAIN,
    RUTA_DESTINO_TEST,
)

import matplotlib.pyplot as plt  # for ploting
import numpy as np  # for arrays
import torch  # for definig neural network
from torch import nn  # for definign NN
from torch import optim  # for defining optimizer
from torchsummary import summary  # for model summary
import torch.nn.functional as F  # for
from torchvision import datasets, transforms, models  # for downloading models
import torchvision.models as models  # for downlaoding models
from PIL import Image  # for manipulating image
from matplotlib.ticker import FormatStrFormatter  # for Ploting
import os


train_dir = RUTA_DESTINO_TRAIN
test_dir = RUTA_DESTINO_TEST


# Tansform with data augmentation and normalization for training
# Just normalization for validation
# Training transform includes random rotation and flip to build a more robust model

#
# definig augmentation for train data RESIZING ROTATING FLIP CONVERTING TO TENSOR AND NORMALIZATION
#
transforms_params = [
    transforms.Resize((244, 244)),
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]
train_transforms = transforms.Compose(transforms_params)
# transform for valid data RESIZE CONVERTING TO TENSOR NORMALIZATIOM
valid_transforms = transforms.Compose(transforms_params)
# transform for test data RESIZE CONVERTING TO TENSOR NORMALIZATIOM
test_transforms = transforms.Compose(transforms_params)

#
#
#

batch_size = 64
# no of images feed to the network at one time

# Loading Dataset
dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
testdataset = datasets.ImageFolder(test_dir, transform=test_transforms)
# splitting our dataset into Train and Validation Dataset
valid_size = int(0.1 * len(dataset))
train_size = len(dataset) - valid_size
dataset_sizes = {'train': train_size, 'valid': valid_size}

train_dataset, valid_dataset = torch.utils.data.random_split(
    dataset, [train_size, valid_size])

# Loading datasets into dataloader
trainloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)
validloader = torch.utils.data.DataLoader(
    valid_dataset, batch_size=batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(
    testdataset, batch_size=batch_size, shuffle=True)

print("Number of Samples in Train: ", len(train_dataset))
print("Number of Samples in Valid: ", len(valid_dataset))
print("Number of Samples in Test: ", len(testdataset))
print("Total: ", len(testdataset) + len(valid_dataset) + len(train_dataset))

print("Number of Classes: ", len(testdataset.classes))

print(dataset.classes)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

#
#
#


#loading pre define ResNet model
model = models.resnet34(pretrained=True)
#****************CHANGING THE OUTPUT LAYER A/C to our requirement*********************
# /home/daniel/.cache/torch/hub/checkpoints/resnet34-b627a593.pth
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 5)
model = model.to(device)


summary(model, input_size=(3, 244, 244))



criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
lrscheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, threshold = 0.9)


#
#
#



def validation(model, validloader, criterion):
    valid_loss = 0
    accuracy = 0

    # change model to work with cuda
    model.to(device)

    # Iterate over data from validloader
    for ii, (images, labels) in enumerate(validloader):

        # Change images and labels to work with cuda
        images, labels = images.to(device), labels.to(device)

        # Forward pass image though model for prediction
        output = model.forward(images)
        # Calculate loss
        valid_loss += criterion(output, labels).item()
        # Calculate probability
        ps = torch.exp(output)

        # Calculate accuracy
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return valid_loss, accuracy


#
#
#


epochs = 50
steps = 0
print_every = 10

#for ploting Graphs
valid_loss_A = []
valid_accuracy_A= []
train_loss_A= []

#change to gpu mode
model.to(device)
model.train()

for e in range(epochs):
    print("Starting Epoch",e+1)
    running_loss = 0

    # Iterating over data to carry out training step
    for ii, (inputs, labels) in enumerate(trainloader):
        steps += 1

        inputs, labels = inputs.to(device), labels.to(device)

        # zeroing parameter gradients
        optimizer.zero_grad()

        # Forward and backward passes
        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        # Carrying out validation step
        if steps % print_every == 0:
            # setting model to evaluation mode during validation
            model.eval()
            # Gradients are turned off as no longer in training
            with torch.no_grad():
                valid_loss, accuracy = validation(model, validloader, criterion)

            ValidLoss = round(valid_loss/len(validloader),3)
            ValidAccuracy = round(float(accuracy/len(validloader)),3)
            TrainingLoss = round(running_loss/print_every,3)
            print("No. epochs:",(e+1),"\tTraining Loss:",TrainingLoss,"\tValid Loss",ValidLoss,"\tValid Accuracy",ValidAccuracy)

            valid_loss_A.append(ValidLoss)
            valid_accuracy_A.append(ValidAccuracy)
            train_loss_A.append(TrainingLoss)

            if (e+1)  == epochs :
                # Saving: feature weights, new model.fc, index-to-class mapping, optimiser state, and No. of epochs
                checkpoint = {
                    'state_dict': model.state_dict(),
                    'model': model.fc,
                    'class_to_idx': dataset.class_to_idx,
                    'opt_state': optimizer.state_dict,
                    'num_epochs': epochs,
                }
                #name = str(e)
                path = './modelFinal.pth'
                torch.save(checkpoint, path)

            # Turning training back on
            model.train()
            lrscheduler.step(accuracy * 100)


epochs = range(len(valid_accuracy_A))

plt.plot(epochs, valid_loss_A, 'r', label='Valid Loss')
plt.plot(epochs, train_loss_A, 'b', label='Train Loss')
plt.title('Valid Loss and Train Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend()
plt.figure()
plt.plot(epochs, valid_loss_A, 'r', label='Valid Loss')
plt.plot(epochs, valid_accuracy_A, 'b', label='Valid Accuracy')
plt.title('Valid Loss and Valid Accuracy')
plt.ylabel('Validation')
plt.xlabel('Epochs')
plt.legend()
plt.show()