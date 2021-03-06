# encoding: utf-8

'''
@author: double4tar
@contact: double4tar@gmail.com
@file: resnet.py
@time: 18-12-27 下午8:24
@desc: modify & copy from https://github.com/Cadene/pretrained-models.pytorch
'''

from __future__ import print_function, division, absolute_import
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo

import nets

__all__ = ['FBResNet',
           #'fbresnet18', 'fbresnet34', 'fbresnet50', 'fbresnet101',
           'fbresnet152']

pretrained_settings = {
    'fbresnet152': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/fbresnet152-2e20f6b4.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    }
}

def conv3x3(in_planes, out_planes, stride=1, moduleName=""):
    "3x3 convolution with padding"
    return nets.setLayName(nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=True), moduleName)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, prefix=""):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, moduleName = prefix + "conv1")
        self.bn1 = nets.setLayName(nn.BatchNorm2d(planes), prefix + "bn1")
        self.relu = nets.setLayName(nn.ReLU(inplace=True), prefix + "relu")
        self.conv2 = conv3x3(planes, planes, moduleName = prefix + "conv2")
        self.bn2 = nets.setLayName(nn.BatchNorm2d(planes), prefix + "bn2")
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=True)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=True)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class FBResNet(nn.Module):

    def __init__(self, block, layers, inputNum=3, num_classes=1000):
        self.inplanes = 64
        # Special attributs
        self.input_space = None
        self.input_size = (299, 299, 3)
        self.mean = None
        self.std = None
        super(FBResNet, self).__init__()
        # Modules
        self.conv1 = nets.setLayName(nn.Conv2d(inputNum, 64, kernel_size=7, stride=2, padding=3,
                                bias=True), 'conv1')
        self.bn1 = nets.setLayName(nn.BatchNorm2d(64), 'bn1')
        self.relu = nets.setLayName(nn.ReLU(inplace=True), 'relu')
        self.maxpool = nets.setLayName(nn.MaxPool2d(kernel_size=3, stride=2, padding=1), 'maxpool')

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, prefixName='resBlock1/')
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, prefixName='resBlock2/')
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, prefixName='resBlock3/')
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, prefixName='resBlock4/')
        self.avgpool = nets.setLayName(nn.AvgPool2d(7), 'avgpool')
        self.last_linear = nets.setLayName(nn.Linear(512 * block.expansion, num_classes), 'linear')

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, prefixName='block'):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nets.setLayName(nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=True), prefixName + 'dw/conv'),
                nets.setLayName(nn.BatchNorm2d(planes * block.expansion), prefixName + 'dw/bn')
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, prefix = prefixName + "step0/"))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, prefix = prefixName + "step" + str(i) + "/"))

        return nn.Sequential(*layers)

    def features(self, input):
        x = self.conv1(input)
        self.conv1_input = x.clone()
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        xStride8 = self.layer2(x)
        xStride16 = self.layer3(xStride8)
        xStride32 = self.layer4(xStride16)
        return [xStride8, xStride16, xStride32]

    def logits(self, features):
        x = self.avgpool(features)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        # origin featExstract and logic regression
        #
        # x = self.logits(x)

        x = self.features(input)
        return x


def fbresnet18(inputNum=3, num_classes=1000):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = FBResNet(BasicBlock, [2, 2, 2, 2], inputNum=inputNum, num_classes=num_classes)
    return model


def fbresnet34(num_classes=1000):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = FBResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)
    return model


def fbresnet50(num_classes=1000):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = FBResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)
    return model


def fbresnet101(num_classes=1000):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = FBResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)
    return model


def fbresnet152(num_classes=1000, pretrained='imagenet'):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = FBResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['fbresnet152'][pretrained]
        assert num_classes == settings['num_classes'], \
            "num_classes should be {}, but is {}".format(settings['num_classes'], num_classes)
        model.load_state_dict(model_zoo.load_url(settings['url']))
        model.input_space = settings['input_space']
        model.input_size = settings['input_size']
        model.input_range = settings['input_range']
        model.mean = settings['mean']
        model.std = settings['std']
    return model
