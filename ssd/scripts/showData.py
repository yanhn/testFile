# encoding: utf-8

'''
@author: double4tar
@contact: double4tar@gmail.com
@file: showData.py
@time: 18-12-25 下午4:55
@desc:
'''

# import data
from data.voc0712 import VOCDetection, VOCAnnotationTransform, VOC_COLOR_ID_MAP
from data import getNumpy_collate
from data.augmentation import SSDAugmentationGray, SSDAugmentationTest
import torch.utils.data as torchData
import cv2

def showData():
    dataDir = r'/mnt/hdisk1/trainSet/pytorch/antifollow'

    trainData = VOCDetection(root = dataDir,
                             image_sets=[('2012', 'trainval')],
                             transform=SSDAugmentationTest(),
                             target_transform=VOCAnnotationTransform(None, True),
                             dataset_name='VOC0712')
    epoch_size = len(trainData)
    data_loader = torchData.DataLoader(trainData, 2,
                                  num_workers=1,
                                  shuffle=False, collate_fn= getNumpy_collate,
                                  pin_memory=True)
    batch_iterator = iter(data_loader)
    while(True):
        try:
            images, targets = next(batch_iterator)
        except StopIteration:
            batch_iterator = iter(data_loader)
            images, targets = next(batch_iterator)
            break

        #show img
        for i in range(len(images)):
            img = images[i]
            imWidth = img.shape[1]
            imHeight = img.shape[0]
            imgColor = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            for rect in targets[i]:
                cv2.rectangle(imgColor, (int(rect[0]*imWidth), int(rect[1]*imHeight)), (int(rect[2]*imWidth), int(rect[3]*imHeight)), VOC_COLOR_ID_MAP[int(rect[4])])

            cv2.imshow("input", imgColor)
            cv2.waitKey()

if __name__ == "__main__":
    showData()
    print("done")