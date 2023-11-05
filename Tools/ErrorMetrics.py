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
import sys
from torchsummary import summary


def Evaluate_Model_TOP1(model, device, dataloader, acc=[], misclassified = []):
    model.to(device)
    model.eval()
    loss = 0
    correct = 0
    pbar = tqdm(dataloader)

    with torch.no_grad():

        for index, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)

            for d,i,j in zip(data, pred, target):
                if i != j:
                    misclassified.append([d.cpu(),i[0].cpu(),j.cpu()])

            loss += F.nll_loss(output, target, reduction='sum').item()
            correct += pred.eq(target.view_as(pred)).sum().item()

    loss /= len(dataloader.dataset)

    # print('\Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
    #     loss, correct, len(dataloader.dataset),
    #     100. * correct / len(dataloader.dataset)))

    acc=100. * correct / len(dataloader.dataset)
    return loss, acc, misclassified

