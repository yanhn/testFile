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
from model import resnet

import os
import logging
import time

def parsArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument('--image_size', type=int, default=32)
    parser.add_argument('--dataSet', type=str, default="Cifar10")
    parser.add_argument('--network', type=str, default="selfDefine")
    # parser.add_argument('--network', type=str, default="res18")
    parser.add_argument('--cifar10_path', type=str, default="./data/raw")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--logPath', type=str, default="log/")          # relative dir to train.py
    parser.add_argument('--savePrefix', type=str, default="save/")
    parser.add_argument('--saveInterval', type=int, default=100)
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
    # cifar10_iter = iter(dataLoader)
    # cifar10_test_iter = iter(testDataLoader)
    # showDataAndLabel(cifar10_iter)

    # model
    if config.network == "selfDefine":
        myCls = baseConv.baseClassfier(3, 10)
    elif config.network == "res18":
        myCls = resnet.ResNet18()
    if torch.cuda.is_available():
        myCls.cuda()

    # criterion
    criterion = nn.CrossEntropyLoss()

    # optimizer
    optimizer = optim.SGD(myCls.parameters(), lr=0.01, momentum=0.9)

    for e in range(400):
        cifar10_iter = iter(dataLoader)
        myCls.train()
        trainLoss = 0
        totalSample = 0
        correctSample = 0
        for i in range(len(cifar10_iter)):      # total 50k, batchSize 32, len = 1563
            try:
                images, targets = next(cifar10_iter)
            except StopIteration:
                # cifar10_iter = iter(dataLoader)
                # images, targets = next(cifar10_iter)
                break

            if torch.cuda.is_available():
                images = images.cuda()
                targets = targets.cuda()

            out = myCls(images)
            loss = criterion(out, targets)
            myCls.zero_grad()

            trainLoss += loss.item()
            _, predicted = out.max(1)
            totalSample += targets.size(0)
            correctSample += predicted.eq(targets).sum().item()


            if ((i+1) % 200 == 0):
                logging.info("epoch: {}, iter: {}, loss: {}, acc: {}".format(e, i, trainLoss / (i + 1), correctSample / totalSample))
                # print("epoch: {}, iter: {}, loss: {}".format(e, i, loss.data.item()))
            loss.backward()
            optimizer.step()

        # do validation
        cifar10_test_iter = iter(testDataLoader)
        myCls.eval()
        valLoss = 0
        valNum = 0
        totalSample = 0
        correctSample = 0
        logging.info("Do validation")
        for i in range(len(cifar10_test_iter)):
            try:
                images, targets = next(cifar10_test_iter)
            except StopIteration:
                break
            valNum += images.shape[0]
            if torch.cuda.is_available():
                images = images.cuda()
                targets = targets.cuda()
            out = myCls(images)

            valLoss += criterion(out, targets).item()
            _, predicted = out.max(1)
            totalSample += targets.size(0)
            correctSample += predicted.eq(targets).sum().item()
        logging.info("epoch: {}, loss: {}, acc: {} of total {} pics.".format(e, valLoss / (i + 1), correctSample / totalSample, totalSample))

        # save checkpoint
        if (e % config.saveInterval == 0):
            saveName = config.savePrefix + getDateString() + "_" + config.network + "_" + str(e) + ".pth"
            torch.save(myCls.state_dict(), saveName)
    return

def getTimeString():
    timeString = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    return timeString

def getDateString():
    dateString = time.strftime("%Y-%m-%d", time.localtime())
    return dateString

def console_output(config):
    logging.basicConfig(filename=config.logPath + getTimeString() + ".log")
    bannerLogger = logging.getLogger()
    bannerLogger.setLevel(logging.INFO)

    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)

if __name__ == "__main__":
    config = parsArgs()
    console_output(config)
    logging.info(config)

    # train loop
    trainCls(config)