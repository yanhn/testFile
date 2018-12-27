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
from data import VOCAnnotationTransform, VOCDetection, BaseTransform, VOC_CLASSES
from nets.ssd import build_ssd
import cv2
import numpy as np

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--trained_model', default='weights/ssd300_3cls_cont110000_ori.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--testImgDir', default='/mnt/hdisk1/testSet/antiFollow/AntiFollowTest/JPEGImages', type=str,
                    help='Dir to load test img from')
parser.add_argument('--save_folder', default='eval/', type=str,
                    help='Dir to save results')
parser.add_argument('--visualThreshold', default=0.6, type=float,
                    help='Final confidence threshold')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use cuda to train model')
parser.add_argument('--useFFT', default=False, type=bool,
                    help='Use fft preprocess or not')
parser.add_argument('--inputChannel', default=1, type=int,
                    help='Number of input channel')
args = parser.parse_args()

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def test_net(save_folder, net, cuda, testset, transform, thresh):
    # dump predictions and assoc. ground truth to text file for now
    filename = save_folder+'test1.txt'
    num_images = len(testset)
    for i in range(num_images):
        print('Testing image {:d}/{:d}....'.format(i+1, num_images))
        img = testset.pull_image(i)
        img_id, annotation = testset.pull_anno(i)
        x = torch.from_numpy(transform(img)[0]).permute(2, 0, 1)
        x = Variable(x.unsqueeze(0))

        with open(filename, mode='a') as f:
            f.write('\nGROUND TRUTH FOR: '+img_id+'\n')
            for box in annotation:
                f.write('label: '+' || '.join(str(b) for b in box)+'\n')
        if cuda:
            x = x.cuda()

        y = net(x)      # forward pass
        detections = y.data
        # scale each detection back up to the image
        scale = torch.Tensor([img.shape[1], img.shape[0],
                             img.shape[1], img.shape[0]])
        pred_num = 0
        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= 0.6:
                if pred_num == 0:
                    with open(filename, mode='a') as f:
                        f.write('PREDICTIONS: '+'\n')
                score = detections[0, i, j, 0]
                label_name = labelmap[i-1]
                pt = (detections[0, i, j, 1:]*scale).cpu().numpy()
                coords = (pt[0], pt[1], pt[2], pt[3])
                pred_num += 1
                with open(filename, mode='a') as f:
                    f.write(str(pred_num)+' label: '+label_name+' score: ' +
                            str(score) + ' '+' || '.join(str(c) for c in coords) + '\n')
                j += 1

def test_net_draw_dir(testDir, net, transform, thresh):
    for root, dirs, files in os.walk(testDir):
        for file in files:
            picDir = os.path.join(root, file)
            img = cv2.imread(picDir, 0)
            DetectImgWithNet(net, img, transform, thresh)

def DetectImgWithNet(net, img, transform, thresh):
    tStart = cv2.getTickCount()
    frame = img.copy()
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
    for j in range(1, detections.size(1)):
        if (j not in [2]):
            continue

        dets = detections[0, j, :]
        scores = dets[:, 0].cpu().numpy()
        idx = np.where(scores > thresh)[0]

        if (len(idx) <= 0):
            continue

        boxes = dets[idx, 0:]
        boxes[:, 1] *= width
        boxes[:, 3] *= width
        boxes[:, 2] *= height
        boxes[:, 4] *= height

        for k in range(0, boxes.size(0)):
            rect = boxes.cpu().numpy()[k, :]
            cv2.rectangle(frame, (rect[1], rect[2]), (rect[3], rect[4]), (255, 255, 255), 2)
            cv2.putText(frame, "{:f}".format(rect[0]), (rect[1], rect[2]), 1, 1, (0, 0, 255))

    tEnd = cv2.getTickCount()
    print("Time cost: {} ms.".format((tEnd - tStart) / cv2.getTickFrequency() * 1000))
    cv2.imshow("result", frame)
    cv2.waitKey()


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