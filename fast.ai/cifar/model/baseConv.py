# encoding: utf-8

'''
@author: double4tar
@contact: double4tar@gmail.com
@file: baseConv.py
@time: 19-1-18 下午12:19
@desc: modify & copy crom https://github.com/amdegroot/ssd.pytorch
'''

import torch
from model import opName
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def conv_bn_relu(inChannel, outChannel, kernelSize, prefix="", stride=2, padding=1, bn=True):
    layers = []
    if bn:
        layers.append(opName(prefix + "/conv", nn.Conv2d(inChannel, outChannel, kernelSize, stride, padding, bias=False)))
        layers.append(opName(prefix + "/bn", nn.BatchNorm2d(outChannel)))
    else:
        layers.append(opName(prefix + "/conv", nn.Conv2d(inChannel, outChannel, kernelSize, stride, padding, bias=False)))
    layers.append(opName(prefix + "/relu", nn.ReLU(inplace=True)))
    return nn.Sequential(*layers)

def conv_bn(inChannel, outChannel, kernelSize, prefix="", stride=2, padding=1, bn=True):
    layers = []
    if bn:
        layers.append(opName(prefix + "/conv", nn.Conv2d(inChannel, outChannel, kernelSize, stride, padding, bias=False)))
        layers.append(opName(prefix + "/bn", nn.BatchNorm2d(outChannel)))
    else:
        layers.append(opName(prefix + "/conv", nn.Conv2d(inChannel, outChannel, kernelSize, stride, padding, bias=False)))
    return nn.Sequential(*layers)

class baseClassfier(nn.Module):
    def __init__(self, inChannel, numClass):
        super(baseClassfier, self).__init__()
        self.inChannel = inChannel
        self.numClass = numClass
        self.layer1 = conv_bn_relu(inChannel, 64, 3, "layer1", 2, 1, True)  # stride 2, featSize = 16
        self.layer2 = conv_bn_relu(64, 128, 3, "layer2", 2, 1, True)        # stride 2, featSize = 8
        self.layer3 = conv_bn_relu(128, 256, 3, "layer3", 2, 1, True)       # stride 2, featSize = 4

        # self.layera = conv_bn_relu(256, 256, 3, "layera", 1, 1, True)       # stride 2, featSize = 4
        # self.layerb = conv_bn_relu(256, 256, 3, "layerb", 1, 1, True)       # stride 2, featSize = 4
        # self.layerc = conv_bn_relu(256, 256, 3, "layerc", 1, 1, True)       # stride 2, featSize = 4

        self.layer4 = conv_bn(256, 256, 4, "layer4", 1, 0, False)            # stride 4, featSize = 1
        self.fc = nn.Linear(256, numClass, bias=False)
        self.initWeights()

    def forward(self, x):
        batchSize = x.shape[0]
        feat = self.layer1(x)
        feat = self.layer2(feat)
        feat = self.layer3(feat)

        # feat = self.layera(feat)
        # feat = self.layerb(feat)
        # feat = self.layerc(feat)

        feat = self.layer4(feat)
        feat = self.fc(feat.view(batchSize, -1))
        out = F.softmax(feat)
        return out

    def initWeights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                # nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

if __name__ == "__main__":
    # a = torch.Tensor()
    # opName("tensor", a)

    net = baseClassfier(3, 10)
    # data = np.zeros([1, 3, 32, 32], dtype='float')
    data = np.random.random([1, 3, 32, 32])
    input = torch.Tensor(data)
    out = net(input)
    print("Stop")