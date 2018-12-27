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
from data import VOC_ROOT, VOC_CLASSES as labelmap
from data import VOCAnnotationTransform, VOCDetection, BaseTransform, VOC_CLASSES, VOC_COLOR_ID_MAP
from nets.ssd import build_ssd
import cv2
import numpy as np

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--trained_model', default='weights/ssd300_3cls_cont110000_fft.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--testImgDir', default='/mnt/hdisk1/testSet/antiFollow/AntiFollowTest/JPEGImages', type=str,
                    help='Dir to load test img from')
parser.add_argument('--save_folder', default='eval/', type=str,
                    help='Dir to save results')
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

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

def test_net_draw_dir(testDir, net, transform, thresh):
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

def testSaveXmlDir(testDir, net, transform, thresh):
    all_boxes = [[] for _ in VOC_CLASSES]
    pass


def test_voc():
    # load net
    num_classes = len(VOC_CLASSES) + 1 # +1 background
    net = build_ssd('test', 300, num_classes, args.inputChannel) # initialize SSD
    net.load_state_dict(torch.load(args.trained_model))
    net.eval()
    print('Finished loading model!')
    # load data
    # testset = VOCDetection(args.voc_root, [('2007', 'test')], None, VOCAnnotationTransform())
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True

    test_net_draw_dir(args.testImgDir, net, BaseTransform(net.size, (120), useFFT=args.useFFT), args.visualThreshold)


if __name__ == '__main__':
    test_voc()
    print("Done")