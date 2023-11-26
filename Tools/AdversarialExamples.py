# FGSM attack code
# https://pytorch.org/tutorials/beginner/fgsm_tutorial.html

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


def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image

def GetAdversarialExamples(model, data, target, device, epsilon=0.01, mean=(0.491, 0.482, 0.446), std=(0.247, 0.243, 0.261)):
    # Set requires_grad attribute of tensor. Important for Attack
    model=model.to(device)
    data.requires_grad = True

    # Forward pass the data through the model
    output = model(data)
    init_pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability

    # Calculate the loss
    loss = F.nll_loss(output, target)

    # Zero all existing gradients
    model.zero_grad()

    # Calculate gradients of model in backward pass
    loss.backward()

    # Collect ``datagrad``
    data_grad = data.grad.data

    # Restore the data to its original scale
    data_denorm = transforms.Normalize((-mean[0]/std[0], -mean[1]/std[1], -mean[2]/std[2]), (1/std[0], 1/std[1], 1/std[2]))(data)

    # Call FGSM Attack
    perturbed_data = fgsm_attack(data_denorm, epsilon, data_grad)

    # Reapply normalization
    perturbed_data_normalized = transforms.Normalize(mean, std)(perturbed_data)

    return perturbed_data_normalized
