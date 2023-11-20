import torch
import torch.optim as optim
import torch.nn.utils.prune as prune
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import torch.nn.functional as F
import torch.nn as nn
from torchsummary import summary

import os
import copy

import torchvision
from torchvision.models import resnet18, ResNet18_Weights, vgg16, VGG16_Weights
from torchvision import datasets
from torchvision import transforms

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import sys
from torchsummary import summary

#Error Metrics
from ErrorMetrics import Evaluate_Model_TOP1

#Measure sparsity
def measure_module_sparsity(module, weight=True, bias=False, use_mask=False):

    num_zeros = 0
    num_elements = 0

    if use_mask == True:
        for buffer_name, buffer in module.named_buffers():
            if "weight_mask" in buffer_name and weight == True:
                num_zeros += torch.sum(buffer == 0).item()
                num_elements += buffer.nelement()
            if "bias_mask" in buffer_name and bias == True:
                num_zeros += torch.sum(buffer == 0).item()
                num_elements += buffer.nelement()
    else:
        for param_name, param in module.named_parameters():
            if "weight" in param_name and weight == True:
                num_zeros += torch.sum(param == 0).item()
                num_elements += param.nelement()
            if "bias" in param_name and bias == True:
                num_zeros += torch.sum(param == 0).item()
                num_elements += param.nelement()

    sparsity = num_zeros / num_elements

    return num_zeros, num_elements, sparsity


def measure_global_sparsity(model,
                            weight=True,
                            bias=False,
                            conv2d_use_mask=False,
                            linear_use_mask=False):

    num_zeros = 0
    num_elements = 0

    for module_name, module in model.named_modules():

        if isinstance(module, torch.nn.Conv2d):

            module_num_zeros, module_num_elements, _ = measure_module_sparsity(
                module, weight=weight, bias=bias, use_mask=conv2d_use_mask)
            num_zeros += module_num_zeros
            num_elements += module_num_elements

        elif isinstance(module, torch.nn.Linear):

            module_num_zeros, module_num_elements, _ = measure_module_sparsity(
                module, weight=weight, bias=bias, use_mask=linear_use_mask)
            num_zeros += module_num_zeros
            num_elements += module_num_elements

    sparsity = num_zeros / num_elements

    return num_zeros, num_elements, sparsity


def model_training(model, device, train_dataloader, optimizer, train_acc, train_losses):

    model.to(device)
    model.train()
    pbar = tqdm(train_dataloader)
    correct = 0
    processed = 0
    running_loss = 0.0

    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        y_pred = model(data)
        loss = F.nll_loss(y_pred, target) #negative log likelihood loss


        train_losses.append(loss.item())
        loss.backward()
        optimizer.step() #Performs a single optimization step (parameter update).

        pred = y_pred.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)
        
        # print statistics
        running_loss += loss.item()
        pbar.set_description(desc=f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
        train_acc.append(100*correct/processed)

def model_testing(model, device, test_dataloader, test_acc, test_losses, misclassified = [], TrainSet=False):
    
    model.to(device)
    model.eval()
    test_loss = 0
    correct = 0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    # label = 0
    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    with torch.no_grad():

        for index, (data, target) in enumerate(test_dataloader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)

            for d,i,j in zip(data, pred, target):
                if i != j:
                    misclassified.append([d.cpu(),i[0].cpu(),j.cpu()])

            test_loss += F.nll_loss(output, target, reduction='sum').item()
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_dataloader.dataset)
    test_losses.append(test_loss)

    if TrainSet==False:
      print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
          test_loss, correct, len(test_dataloader.dataset),
          100. * correct / len(test_dataloader.dataset)))
    
    if TrainSet==True:
      print('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
          test_loss, correct, len(test_dataloader.dataset),
          100. * correct / len(test_dataloader.dataset)))

    test_acc.append(100. * correct / len(test_dataloader.dataset))
    return misclassified

#TestLoader: Non Corrupted Test Data
#TestCLoader: Corrupted Test Data
#PruningRate: Rate that we will prune the model e.g. 10% per iteration
#NumEpochsEachPruneRetrain: Number of epochs that we will retrain the model after pruning (do it untill converge)

SaveCorruptedAtP=[0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.05,0.01]#Corrupted data will be saved at these sparsity levels
def iterative_pruning(model, device, TrainLoader, TestLoader, TestCLoader, PruningRate, SaveTrainStepsName,ReadMe, NumEpochsEachPruneRetrain=1,MaximumPruneLevel=0.01, SaveCorruptedAtP=SaveCorruptedAtP): 
  #Total number of pruning iterations so that (1-PruningRate)^TotalNumPruningIterations = MaximumPruneLevel
  TotalNumPruningIterations= int(np.round(np.log(MaximumPruneLevel)/np.log(1-PruningRate))) 


  Corruptions=list(TestCLoader.keys())
  Acc_Corruptions={}
  for cname in Corruptions:
    Acc_Corruptions[cname]=[]
  
  test_acc = []
  test_C_cc = []
  
  Sparcity_List=[]
  Avg_Corrup_Acc_List=[]

  RemainingWeights_List=[]

  RemainingWeightsP = 1
  RemainingWeights_List.append(RemainingWeightsP)
  for i in range(TotalNumPruningIterations):
    RemainingWeightsP=RemainingWeightsP*(1-PruningRate)
    RemainingWeights_List.append(RemainingWeightsP)
  RemainingWeights_List=np.array(RemainingWeights_List)

  IdxSaveCorrupted=[]
  #Find closes index to
  for i in SaveCorruptedAtP:
    IdxSaveCorrupted.append(np.argmin(np.abs(RemainingWeights_List[1:]-i)))

  #remove duplicates
  IdxSaveCorrupted=list(set(IdxSaveCorrupted))

  #Get statistics before pruning---------
  Sparcity_List.append(-1)

  print("--Get statistics before pruning--")
  train_acc = []
  _ = model_testing(model, device, TrainLoader, train_acc, [],TrainSet=True)
  _ = model_testing(model, device, TestLoader, test_acc, [])


  AvgCorrupAcc=0
  print("--Testing Corruptions--")
  for cname in tqdm(Corruptions):

    loss, acc, misclassified=Evaluate_Model_TOP1(model, device, TestCLoader[cname],TQDM=False)
    AvgCorrupAcc+=acc
    Acc_Corruptions[cname].append(acc)

  AvgCorrupAcc=AvgCorrupAcc/len(Corruptions)
  Avg_Corrup_Acc_List.append(AvgCorrupAcc)
  print(f"\nAvg Corruption Accuracy: {AvgCorrupAcc:.2f}%")

  #Get statistics before pruning---------end      
  
  for i in range(TotalNumPruningIterations):
    print("---------- Pruning Iteration: ", i+1, "/", TotalNumPruningIterations)

    # All Conv2D Params are Pruned
    pruning_params = []
    for module_name, module in model.named_modules():
      if isinstance(module, torch.nn.Conv2d):
        pruning_params.append((module, "weight"))

    prune.global_unstructured(pruning_params, pruning_method=prune.L1Unstructured, amount = PruningRate)


   
    train_losses = []
    train_acc = []

    test_losses = []
 
    # Retrain the model
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.05, patience=2, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=True)
    

    #Non Corrupted Data
    for i in range(NumEpochsEachPruneRetrain):
        model_training(model, device, TrainLoader, optimizer, train_acc, train_losses)
        scheduler.step(train_losses[-1])

    _ = model_testing(model, device, TestLoader, test_acc, test_losses)

    #Corrupted Data
    if i in IdxSaveCorrupted:
      AvgCorrupAcc=0
      print("--Testing Corruptions--")
      for cname in tqdm(Corruptions):

        loss, acc, misclassified=Evaluate_Model_TOP1(model, device, TestCLoader[cname],TQDM=False)
        AvgCorrupAcc+=acc
        Acc_Corruptions[cname].append(acc)

      AvgCorrupAcc=AvgCorrupAcc/len(Corruptions)
      Avg_Corrup_Acc_List.append(AvgCorrupAcc)
      print(f"\nAvg Corruption Accuracy: {AvgCorrupAcc:.2f}%")

    else:
      print("--Skipping Acc Comp For Corrupted Data--")
      for cname in tqdm(Corruptions):
        Acc_Corruptions[cname].append(-1)
      Avg_Corrup_Acc_List.append(-1)


    #Get global sparsity
    num_zeros, num_elements, sparsity_i = measure_global_sparsity(model,weight=True,bias=False,conv2d_use_mask=True,linear_use_mask=False)
    print("Global Sparsity: {:.2f}% \n".format(sparsity_i*100))
    Sparcity_List.append(sparsity_i)


  np.savez("./Networks/"+SaveTrainStepsName+".npz",ReadMe=ReadMe, TestAcc=test_acc, Acc_Corruptions=Acc_Corruptions, Avg_Corrup_Acc_List=Avg_Corrup_Acc_List, RemainingWeights_List=RemainingWeights_List, Sparcity_List=Sparcity_List)
  
  return test_acc, Avg_Corrup_Acc_List, Acc_Corruptions, RemainingWeights_List, Sparcity_List


