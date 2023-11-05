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



def TinyImagenetC_DataLoader(root="./Datasets/TINY-IMAGENET/TinyImageNet-C/Tiny-ImageNet-C",batch_size=64):

    TinyImagenetC_corruptions = ["brightness", "contrast", "defocus_blur", "elastic_transform", 
               "fog", "frost", "gaussian_blur", "gaussian_noise","glass_blur",
               "impulse_noise", "jpeg_compression", "motion_blur",
               "pixelate", "saturate", "shot_noise", "snow", "spatter", "speckle_noise","zoom_blur"]

    SeverityLevels=["1","2","3","4","5"]

    transformList = [transforms.ToTensor(),transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
    transform = transforms.Compose(transformList)

    Dataloaders={}
    for cname in TinyImagenetC_corruptions:
        Level_Dataloaders={}
        for clevel in SeverityLevels:
            data_path=os.path.join(root, cname,clevel)
            Dataset = datasets.ImageFolder(data_path, transform=transform)
            Dataloader = torch.utils.data.DataLoader(Dataset, batch_size=batch_size,shuffle=True, num_workers=2)
            Level_Dataloaders[clevel]=Dataloader #Levels for individual corruptions

        Dataloaders[cname]=Level_Dataloaders #All Corruptions and their levels

    return Dataloaders