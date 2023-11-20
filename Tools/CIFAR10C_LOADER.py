import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import torch.nn.functional as F
from torch.utils.data import Subset


import torchvision
from torchvision.models import resnet18, ResNet18_Weights, vgg16, VGG16_Weights
from torchvision import datasets
from torchvision import transforms

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import PIL
from PIL import Image

import os
from torchsummary import summary



# from google.colab import drive
# drive.mount('/content/drive')
# #Code folder path
# %cd /content/drive/My Drive/ECE591_DL_CL_PROJECT/

CIFAR10C_corruptions = ["brightness", "contrast", "defocus_blur", "elastic_transform", 
               "fog", "frost", "gaussian_blur", "gaussian_noise","glass_blur",
               "impulse_noise", "jpeg_compression", "motion_blur",
               "pixelate", "saturate", "shot_noise", "snow", "spatter", "speckle_noise","zoom_blur"]

#https://github.com/tanimutomo/cifar10-c-eval/blob/master
class CIFAR10C(datasets.VisionDataset):
    def __init__(self, root :str, name :str,transform):

        corruptions = ["brightness", "contrast", "defocus_blur", "elastic_transform", 
               "fog", "frost", "gaussian_blur", "gaussian_noise","glass_blur",
               "impulse_noise", "jpeg_compression", "motion_blur",
               "pixelate", "saturate", "shot_noise", "snow", "spatter", "speckle_noise","zoom_blur"]
        
        assert name in corruptions
        super(CIFAR10C, self).__init__(root, transform=transform)

        data_path = os.path.join(root, name + '.npy')
        target_path = os.path.join(root, 'labels.npy')
        
        self.data = np.load(data_path)
        self.targets = np.load(target_path)
        
    def __getitem__(self, index):
        img, targets = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        img = self.transform(img)

            
        return img, targets
    
    def __len__(self):
        return len(self.data)
    

def CIFAR10C_DataLoader(root="./Datasets/CIFAR-10/CIFAR-10-C/", batch_size=64):
    Dataloaders={}
    transformList = [transforms.ToTensor(),transforms.Normalize((0.491, 0.482, 0.446), (0.247, 0.243, 0.261))]
    transform = transforms.Compose(transformList)

    CIFAR10C_corruptions = ["brightness", "contrast", "defocus_blur", "elastic_transform", 
               "fog", "frost", "gaussian_blur", "gaussian_noise","glass_blur",
               "impulse_noise", "jpeg_compression", "motion_blur",
               "pixelate", "saturate", "shot_noise", "snow", "spatter", "speckle_noise","zoom_blur"]
    
    for cname in CIFAR10C_corruptions:
        dataset = CIFAR10C(root,cname,transform=transform)
        CIFARC_Loader=torch.utils.data.DataLoader(dataset, batch_size=batch_size,shuffle=False, num_workers=2)
        Dataloaders[cname]=CIFARC_Loader

    return Dataloaders
