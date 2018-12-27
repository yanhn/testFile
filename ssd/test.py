# encoding: utf-8

'''
@author: double4tar
@contact: double4tar@gmail.com
@file: test.py
@time: 18-12-27 上午10:18
@desc: modify & copy crom https://github.com/amdegroot/ssd.pytorch
'''

from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from data import BaseTransform, VOC_CLASSES, VOC_COLOR_ID_MAP
from nets.ssd import build_ssd
import cv2
import numpy as np
import eval.BBoxXmlTool as bxt
from utils import *

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--trained_model', default='weights/ssd300_3cls_cont110000_fftLessPad.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--testImgDir', default='/mnt/hdisk1/testSet/antiFollow/AntiFollowTest/JPEGImages', type=str,
                    help='Dir to load test img from')
parser.add_argument('--action', default='show', type=str, choices=['show', 'draw', 'xml'],
                    help='use demo to to what?')
parser.add_argument('--saveXMLPath', default='log/', type=str,
                    help='output xml path')
parser.add_argument('--visualThreshold', default=0.6, type=float,
                    help='Final confidence threshold')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use cuda to train model')
parser.add_argument('--useFFT', default=True, type=bool,
                    help='Use fft preprocess or not')
parser.add_argument('--inputChannel', default=3, type=int,
                    help='Number of input channel')

args = parser.parse_args()

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


def testNetDrawDir(testDir, net, transform, thresh):
    for root, dirs, files in os.walk(testDir):
        for file in files:
            picDir = os.path.join(root, file)
            img = cv2.imread(picDir, 0)
            bboxs = DetectImgWithNet(net, img, transform, thresh)

            if (len(img.shape)<3):
                # grayPic
                frame = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                frame = img.copy()

            # show result
            for idx, name in enumerate(VOC_CLASSES):
                for bbox in bboxs[idx]:
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), VOC_COLOR_ID_MAP[idx], 2)
                    cv2.putText(frame, "{:f}".format(bbox[4]), (bbox[0], bbox[1]), 1, 1, VOC_COLOR_ID_MAP[idx])
            cv2.imshow("result", frame)
            cv2.waitKey()

def DetectImgWithNet(net, frame, transform, thresh):
    '''
    Use net to detect img, with transform, filter result by confidence threshold
    :param net:
    :param frame:
    :param transform:
    :param thresh:
    :return: bounding box list in format [xmin, ymin, xmax, ymax, score], with dim0 represents class index
    '''
    tStart = cv2.getTickCount()
    # frame = img.copy()
    try:
        height, width, channels = frame.shape
        x = torch.from_numpy(transform(frame)[0]).permute(2, 0, 1)
    except:
        height, width = frame.shape
        x = torch.from_numpy(transform(frame)[0])
        x = Variable(x.unsqueeze(0))

    # # # img_id, annotation = testset.pull_anno(i)
    if (len(x.shape) < 4):
        x = Variable(x.unsqueeze(0))
    if args.cuda:
        x = x.cuda()

    y = net(x)  # forward pass
    detections = y.data
    retBBoxs = []
    for j in range(1, detections.size(1)):
        # if (j not in [2]):
        #     continue

        dets = detections[0, j, :]
        scores = dets[:, 0].cpu().numpy()
        idx = np.where(scores > thresh)[0]

        if (len(idx) <= 0):
            retBBoxs.append(np.array(()))
            continue

        boxes = dets[idx, 0:]
        boxes[:, 1] *= width
        boxes[:, 3] *= width
        boxes[:, 2] *= height
        boxes[:, 4] *= height

        boxes[:,:] = boxes[:, [1,2,3,4,0]]  # change order to [xmin, ymin, xmax, ymax, score]
        retBBoxs.append(boxes.cpu().numpy())

    tEnd = cv2.getTickCount()
    print("Time cost: {} ms.".format((tEnd - tStart) / cv2.getTickFrequency() * 1000))
    return retBBoxs

def testSaveXmlDir(testDir, net, transform, thresh, display=False):
    modelCode, _ = os.path.splitext(args.trained_model.split('/')[-1])
    saveXMLDir = args.saveXMLPath + getTimeString() + "_" + modelCode + "_xml"
    if not os.path.isdir(saveXMLDir): os.makedirs(saveXMLDir)
    if (display):
        saveDisplayDir = args.saveXMLPath + getTimeString() + "_" + modelCode + "_display"
        if not os.path.isdir(saveDisplayDir): os.makedirs(saveDisplayDir)

    for root, dirs, files in os.walk(testDir):
        totalPicNum = len(files)
        picCounter = 0
        for file in files:
            picDir = os.path.join(root, file)
            img = cv2.imread(picDir, 0)
            bboxs = DetectImgWithNet(net, img, transform, thresh)

            tmpBXT = bxt.IMGBBox()
            tmpBXT.setIMG(img)
            tmpBXT.img_name = os.path.basename(picDir)
            fname, _ = os.path.splitext(tmpBXT.img_name)
            tmpBXT.xml_name = fname + '.xml'

            for cls_idx in range(len(VOC_CLASSES)):

                # can have class filter here
                #

                cls_name = VOC_CLASSES[cls_idx]
                tmpBXT.addDet(bboxs[cls_idx], cls_name)
            if len(tmpBXT.bboxes) > 0:
                picCounter += 1
                tmpBXT.saveXML(save_dir=saveXMLDir)
                if (display):
                    tmpBXT.showIMG(save_dir=saveDisplayDir)
            logging.info("Detected {} of {}.".format(picCounter, totalPicNum))

def test_voc():
    # load net
    num_classes = len(VOC_CLASSES) + 1 # +1 background
    net = build_ssd('test', 300, num_classes, args.inputChannel) # initialize SSD
    net.load_state_dict(torch.load(args.trained_model))
    net.eval()
    print('Finished loading model!')

    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True

    if (args.action == 'show'):
        testNetDrawDir(args.testImgDir, net, BaseTransform(net.size, (120), useFFT=args.useFFT), args.visualThreshold)
    elif(args.action == 'xml'):
        testSaveXmlDir(args.testImgDir, net, BaseTransform(net.size, (120), useFFT=args.useFFT), args.visualThreshold)
    elif(args.action == 'draw'):
        testSaveXmlDir(args.testImgDir, net, BaseTransform(net.size, (120), useFFT=args.useFFT), args.visualThreshold, display=True)

if __name__ == '__main__':
    console_output()
    test_voc()
    print("Done")