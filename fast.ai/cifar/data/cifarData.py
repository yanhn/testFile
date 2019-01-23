# encoding: utf-8

'''
@author: double4tar
@contact: double4tar@gmail.com
@file: cifarData.py
@time: 19-1-16 下午5:45
@desc: modify & copy crom https://github.com/amdegroot/ssd.pytorch
'''

import torchvision as tv
from torchvision import datasets
from torchvision import transforms
import torch

def getDataLoader(config):
    # transform = transforms.Compose([
    #     transforms.Scale(config.image_size),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    cifa10 = datasets.CIFAR10(root=config.cifar10_path, download=True, transform=transform)

    cifa10_loader = torch.utils.data.DataLoader(dataset=cifa10,
                                              batch_size=config.batch_size,
                                              shuffle=True,
                                              num_workers=config.num_workers)

    return cifa10_loader

def getTestDataLoader(config):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    cifa10 = datasets.CIFAR10(root=config.cifar10_path, download=True, transform=transform, train=False)

    cifa10_loader = torch.utils.data.DataLoader(dataset=cifa10,
                                              batch_size=1,
                                              shuffle=True,
                                              num_workers=config.num_workers)
    return cifa10_loader