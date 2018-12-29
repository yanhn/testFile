# encoding: utf-8

'''
@author: double4tar
@contact: double4tar@gmail.com
@file: train.py
@time: 18-12-25 下午3:42
@desc:
'''

import argparse
from nets.selfSSD import resSSD
import numpy as np
import torch
from utils import *
from data import *
import torch.backends.cudnn as cudnn
from nets.modules import MultiBoxLoss
import torch.optim as optim
import torch.utils.data as data
from torch.autograd import Variable
import pdb

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def parseParameters():
    parser = argparse.ArgumentParser(
        description='Single Shot MultiBox Detector Training With Pytorch')
    parser.add_argument('--cuda', default=True, type=str2bool,
                        help='Use CUDA to train model')
    parser.add_argument('--dataset_root', default=r"/home/hawk/dataset/trainSet/antiFollow",
                        help='Dataset root directory path')
    parser.add_argument('--basenet', default='vgg16_reducedfc.pth',
                        help='Pretrained base model')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size for training')
    parser.add_argument('--num_workers', default=8, type=int,
                        help='Number of workers used in dataloading')
    parser.add_argument('--resume', default='weights/scratchRes18.pth', type=str,
                        help='Checkpoint state_dict file to resume training from')
    parser.add_argument('--start_iter', default=0, type=int,
                        help='Resume training at this iter')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='Momentum value for optim')
    parser.add_argument('--weight_decay', default=5e-4, type=float,
                        help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.1, type=float,
                        help='Gamma update for SGD')
    parser.add_argument('--save_folder', default='weights/',
                        help='Directory for saving checkpoint models')
    args = parser.parse_args()
    return args

def trainSSD():
    # cfg = voc
    # ssdNet = resSSD(phase='train', numClasses=cfg['num_classes'], size=cfg['min_dim'], inChannelNum=3)
    # # torch.save(ssdNet.state_dict(), "weights/scratchRes18.pth")
    # ssdNet = ssdNet.cuda()
    # x = np.ndarray((32,3,300,300),dtype='float')
    # x = torch.Tensor(x).cuda()
    # y = ssdNet(x)
    # print(y[0].shape)

    # data set
    cfg = voc
    dataset = VOCDetection(root=args.dataset_root,
                           transform=SSDAugmentationGray(cfg['min_dim'],
                                                         MEANS))

    # net build
    ssdNet = resSSD(phase='train', numClasses=cfg['num_classes'], size=cfg['min_dim'], inChannelNum=3)
    net = ssdNet

    if args.cuda:
        # cuda net
        net = torch.nn.DataParallel(ssdNet)
        cudnn.benchmark = True

    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        ssdNet.load_weights(args.resume)
    else:
        vgg_weights = torch.load(args.save_folder + args.basenet)
        # baseNetWeights = collections.OrderedDict()
        # for key in vgg_weights:
        #     if(key[0:3] == 'vgg'):
        #         newkey = key[4:]
        #         baseNetWeights[newkey] = vgg_weights[key]
        #     else:
        #         break
        #     print(key)
        print('Loading base network...')
        # ssd_net.vgg.load_state_dict(baseNetWeights)
        model_dict = ssdNet.vgg.state_dict()
        pretrained_dict = torch.load(args.save_folder + args.basenet)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k != "0.weight"}
        model_dict.update(pretrained_dict)
        ssdNet.vgg.load_state_dict(model_dict)


    # criterion
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                             False, args.cuda)

    net.train()

    if args.cuda:
        net = net.cuda()

    loc_loss = 0
    conf_loss = 0
    epoch = 0
    step_index = 0

    epoch_size = len(dataset) // args.batch_size
    print('Training SSD on:', dataset.name)
    print('Using the specified args:')
    print(args)

    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)

    # create batch iterator
    batch_iterator = iter(data_loader)
    for iteration in range(args.start_iter, cfg['max_iter']):
        if iteration in cfg['lr_steps']:
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)

        try:
            images, targets = next(batch_iterator)
        except StopIteration:
            # end of this epoch
            batch_iterator = iter(data_loader)
            images, targets = next(batch_iterator)

        if args.cuda:
            images = Variable(images.cuda())
            targets = [Variable(ann.cuda(), volatile=True) for ann in targets]
        else:
            images = Variable(images)
            targets = [Variable(ann, volatile=True) for ann in targets]
    #     # forward
    #     t0 = time.time()
    #     # # uncomment if use gray img
    #     # images = torch.Tensor.unsqueeze(images, dim = 1)
        out = net(images)
        # backprop
        optimizer.zero_grad()
        loss_l, loss_c = criterion(out, targets)
        loss = loss_l + loss_c
        loss.backward()
        optimizer.step()
        # t1 = time.time()
        # pdb.set_trace()
        loc_loss += loss_l.data.item()
        conf_loss += loss_c.data.item()

        if iteration % 25 == 0:
            # print('timer: %.4f sec.' % (t1 - t0))
            # print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.data[0]), end=' ')
            timeString = time.strftime("%Y-%m-%d %H:%M:%S ", time.localtime())
            logging.info('@ ' + timeString + 'iter ' + repr(iteration) + ' || locLoss: %.4f ||  clsLoss: %.4f || Loss: %.4f ||'
                         % (loss_l.data.item(), loss_c.data.item(), loss.data.item()))

        if iteration != 0 and iteration % 10000 == 0:
            print('Saving state, iter:', iteration)
            torch.save(ssdNet.state_dict(), 'weights/res18/ssd300_res18' +
                       repr(iteration) + '.pth')
    torch.save(ssdNet.state_dict(),
               args.save_folder + '' + args.dataset + '.pth')

def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':
    args = parseParameters()
    console_output()
    trainSSD()
    print("Done!")