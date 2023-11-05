import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import torch.nn.functional as F


import torchvision
from torchvision.models import resnet18, ResNet18_Weights, vgg16, VGG16_Weights
from torchvision import datasets
from torchvision import transforms

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


import os
from torchsummary import summary

# Returns a list of transformations when called
class GetTransforms():
    '''Returns a list of transformations when type as requested amongst train/test
       Transforms('train') = list of transforms to apply on training data
       Transforms('test') = list of transforms to apply on testing data'''

    def __init__(self):
        pass

    def trainparams(self):
        train_transformations = [ #resises the image so it can be perfect for our model.
            transforms.RandomHorizontalFlip(), # FLips the image w.r.t horizontal axis
            transforms.RandomRotation((-7,7)),     #Rotates the image to a specified angel
            transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)), #Performs actions like zooms, change shear angles.
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Set the color params
            transforms.ToTensor(), # comvert the image to tensor so that it can work with torch
            transforms.Normalize((0.491, 0.482, 0.446), (0.247, 0.243, 0.261)) #Normalize all the images
            ]

        return train_transformations

    def testparams(self):
        test_transforms = [
            transforms.ToTensor(),
            transforms.Normalize((0.491, 0.482, 0.446), (0.247, 0.243, 0.261)) #Normalize all the images
        ]
        return test_transforms



#Download CIFAR-10 Dataset
class GetCIFAR10_TrainData():
    def __init__(self, dir_name:str, train_transforms=None, test_transforms=None):
        self.dirname = dir_name
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms

    def download_train_data(self):
        return datasets.CIFAR10(self.dirname, train=True, download=True, transform=self.train_transforms)

    def download_test_data(self):
        return datasets.CIFAR10(self.dirname, train=False, download=True, transform=self.test_transforms)


def CIFAR10DataLoader(batch_size=512):

    transformations = GetTransforms()
    train_transforms = transforms.Compose(transformations.trainparams())
    test_transforms = transforms.Compose(transformations.testparams()) 

    data = GetCIFAR10_TrainData('./Datasets/CIFAR-10', train_transforms, test_transforms)
    trainset = data.download_train_data()
    testset = data.download_test_data()

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,shuffle=False, num_workers=2)

    return trainloader, testloader
