# encoding: utf-8

'''
@author: double4tar
@contact: double4tar@gmail.com
@file: __init__.py
@time: 18-12-25 下午4:38
@desc: modify & copy crom https://github.com/amdegroot/ssd.pytorch
'''

from .voc0712 import VOCDetection, VOCAnnotationTransform, VOC_CLASSES, VOC_ROOT, VOC_COLOR_ID_MAP
from .config import *
import torch
import cv2
import numpy as np
from data.augmentation import FFTTrans, SSDAugmentationGray

def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(imgs, 0), targets

def getNumpy_collate(batch):
    '''
    Used to convert tensor data into numpy and show it directly, no need to Tensorize it and deTensorize it.
    :param batch:
    :return:
    '''
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0].numpy())
        targets.append(sample[1])
    return imgs, targets

def base_transform(image, size, mean):
    x = cv2.resize(image, (size, size)).astype(np.float32)
    x -= mean
    x = x.astype(np.float32)
    return x

class BaseTransform:
    def __init__(self, size, mean, useFFT=False):
        self.size = size
        self.mean = np.array(mean, dtype=np.float32)
        self._fftTransform = FFTTrans()
        self._useFFT = useFFT

    def __call__(self, image, boxes=None, labels=None):
        im, box, label = base_transform(image, self.size, self.mean), boxes, labels
        if (self._useFFT):
            im, box, label = self._fftTransform(im, box, label)
        return im, box, label
