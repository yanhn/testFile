# encoding: utf-8

'''
@author: double4tar
@contact: double4tar@gmail.com
@file: train.py.py
@time: 19-1-16 下午5:45
@desc: modify & copy crom https://github.com/amdegroot/ssd.pytorch
'''

import argparse
from data.cifarData import *
import cv2
import numpy as np
import torch.nn as nn
from torch import optim
from model import baseConv

def parsArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument('--image_size', type=int, default=32)
    parser.add_argument('--dataSet', type=str, default="Cifar10")
    parser.add_argument('--cifar10_path', type=str, default="./data/raw")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)

    config = parser.parse_args()
    return config

def showDataAndLabel(dataIter):
    while (True):
        try:
            images, targets = next(dataIter)
        except StopIteration:
            # batch_iterator = iter(dataLoader)
            # images, targets = next(batch_iterator)
            break

        # show img
        for i in range(len(images)):
            img = images[i]
            imWidth = img.shape[1]
            imHeight = img.shape[0]
            img_color = img.numpy()
            img_color = np.transpose(img_color, [1,2,0])
            img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)

            cv2.putText(img_color, "{}".format(targets[i]), (100, 100), 1, 4, (0, 255, 0), 2)
            cv2.imshow("input", img_color)
            cv2.waitKey()

# data/model/loss/lr/saving/validation
def trainCls(config):
    # data
    dataLoader = getDataLoader(config)
    testDataLoader = getTestDataLoader(config)
    cifar10_iter = iter(dataLoader)
    cifar10_test_iter = iter(testDataLoader)
    # showDataAndLabel(cifar10_iter)

    # model
    myCls = baseConv.baseClassfier(3, 10)
    if torch.cuda.is_available():
        myCls.cuda()
    myCls.train()

    # criterion
    criterion = nn.CrossEntropyLoss()

    # optimizer
    optimizer = optim.SGD(myCls.parameters(), lr=0.01, momentum=0.9)

    for e in range(3):
        for i in range(len(cifar10_iter)):      # total 50k, batchSize 32, len = 1563
            try:
                images, targets = next(cifar10_iter)
            except StopIteration:
                cifar10_iter = iter(dataLoader)
                # images, targets = next(cifar10_iter)
                break

            if torch.cuda.is_available():
                images = images.cuda()
                targets = targets.cuda()

            out = myCls(images)
            loss = criterion(out, targets)
            myCls.zero_grad()
            if (i % 100 == 0):
                print("epoch: {}, iter: {}, loss: {}".format(e, i, loss.data.item()))
            loss.backward()
            optimizer.step()

    return

if __name__ == "__main__":
    # loss = nn.CrossEntropyLoss()
    # # input, NxC=2x3
    # input = torch.randn(2, 3, requires_grad=True)
    # # target, N
    # target = torch.empty(2, dtype=torch.long).random_(3)
    # output = loss(input, target)
    # output.backward()


    config = parsArgs()
    trainCls(config)