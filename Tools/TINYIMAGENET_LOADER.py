import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
import pandas as pd

import torchvision
from torchvision.models import resnet18, ResNet18_Weights, vgg16, VGG16_Weights
from torchvision import datasets
from torchvision import transforms

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

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
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) #Normalize all the images
            ]

        return train_transformations

    def testparams(self):
        test_transforms = [
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) #Normalize all the images
        ]
        return test_transforms
    

class CustomDataset(datasets.VisionDataset):
    def __init__(self, images, labels ,transform):
        self.images = images
        self.labels = labels
        self.transform = transform
    

    def __getitem__(self, index):
        img = self.images[index]
        img = Image.fromarray(img)
        img = self.transform(img)

        lab = self.labels[index]

        return img, lab
    
    def __len__(self):
        return len(self.images)


def TINYIMAGENETDataLoader(batch_size=256, NPZ=True):
    transformations = GetTransforms()
    train_transforms = transforms.Compose(transformations.trainparams())
    test_transforms = transforms.Compose(transformations.testparams()) 

    if NPZ==False:

        DataPath="./Datasets/TINY-IMAGENET/tiny-imagenet-200"
        trainset = datasets.ImageFolder(DataPath+'/train', transform=train_transforms)
        testset = datasets.ImageFolder(DataPath+'/test_pro', transform=test_transforms)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True, num_workers=2)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,shuffle=False, num_workers=2)

        return trainloader, testloader
    
    #Load from NPZ, good for colab
    if NPZ==True:

        DataPath="./Datasets/TINY-IMAGENET/tiny-imagenet-200_NPZ"

        data_train = np.load(DataPath+'/train.npz')
        images_train = data_train['images']
        labels_train = data_train['labels']

        data_test = np.load(DataPath+'/test_pro.npz')
        images_test = data_test['images']
        labels_test = data_test['labels']

        trainset = CustomDataset(images_train, labels_train, transform=train_transforms)
        testset = CustomDataset(images_test, labels_test, transform=test_transforms)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True, num_workers=2)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,shuffle=False, num_workers=2)

        return trainloader, testloader