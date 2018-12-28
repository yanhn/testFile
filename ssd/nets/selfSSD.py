# encoding: utf-8

'''
@author: double4tar
@contact: double4tar@gmail.com
@file: selfSSD.py
@time: 18-12-28 上午10:40
@desc:
'''

import torch
import torch.nn as nn
from nets.resnet import fbresnet18
from nets import *
from torch.autograd import Variable
from data.config import voc

RES_SSD_CONFIG = {
    "300" : {
        "res_Stride8" : 128,
        "res_Stride16" : 256,
        "res_Stride32" : 512,
        "extra_Stride64" : 256,
        "extra_Stride128" : 256,
        "extra_Stride256" : 256,
    },
    "512" : {}
}
MULTIBOX_CONFIG = {
    '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [],
}

class extraFeats(nn.Module):

    def __init__(self, cfg):
        super(extraFeats, self).__init__()
        self._extras = []
        for k,v in cfg.items():
            tmpSequential = []
            if(k.startswith('extra')):
                tmpSequential.append(setLayName(nn.Conv2d(lastLayerOutputChanel, v,
                           kernel_size=3, stride=2, padding=1), k + '/conv'))
                tmpSequential.append(setLayName(nn.BatchNorm2d(v), k + '/bn'))
                tmpSequential.append(setLayName(nn.ReLU(inplace=True), k + '/relu'))
                tmpSequential = nn.Sequential(*tmpSequential)
                self._extras.append(tmpSequential)
            lastLayerOutputChanel = v

        self._initialize_weights()

    def forward(self, input):
        ret = []
        for stage in self._extras:
            input = stage(input)
            ret.append(input)
        return ret

    def _initialize_weights(self):
        for part in self._extras:
            for m in part.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_normal_(m.weight)
                    # nn.init.kaiming_normal_(m.weight, mode='fan_out')
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

class resSSD(nn.Module):
    def __init__(self, phase, size=300, numClasses=21, inChannelNum=3):
        super(resSSD, self).__init__()
        self._phase = phase
        self._numClasses = numClasses
        self._inChannelNum = inChannelNum
        self._size = str(size)
        assert self._size in MULTIBOX_CONFIG.keys() and self._size in RES_SSD_CONFIG.keys(), "this input size not implemented. {}".format(size)

        self._features = fbresnet18(inputNum=inChannelNum)
        self._extras = extraFeats(RES_SSD_CONFIG['300'])
        assert len(RES_SSD_CONFIG['300']) == len(MULTIBOX_CONFIG['300'])
        self._loc_layers = []
        self._conf_layers = []
        for id, item in enumerate(RES_SSD_CONFIG['300'].items()):
            self._loc_layers += [setLayName(nn.Conv2d(item[1], MULTIBOX_CONFIG['300'][id] * 4, kernel_size=3, padding=1), 'loc' + str(id))]
            self._conf_layers += [setLayName(nn.Conv2d(item[1], MULTIBOX_CONFIG['300'][id] * self._numClasses, kernel_size=3, padding=1), 'loc' + str(id))]

        self.cfg = voc
        self.priorbox = PriorBox(self.cfg)      # need to check
        self.priors = Variable(self.priorbox.forward(), volatile=True)

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(self._numClasses, 0, 200, 0.01, 0.45)

        self._initialize_weights()

    def forward(self, input):
        resFeats = self._features(input)
        extras = self._extras(resFeats[-1])
        # add two parts of feats
        allFeats = resFeats + extras
        assert len(allFeats) == len(self._loc_layers) == len(self._conf_layers)

        loc = list()
        conf = list()
        # apply multibox head to source layers
        for (x, l, c) in zip(allFeats, self._loc_layers, self._conf_layers):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        if self._phase == "test":
            output = self.detect(
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(conf.size(0), -1,
                                       self._numClasses)),                # conf preds
                self.priors.type(type(input.data))                  # default boxes
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self._numClasses),
                self.priors
            )
        return output

    def _initialize_weights(self):
        for m in self._loc_layers:
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                # nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    m.bias.data.zero_()
        for m in self._conf_layers:
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                # nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    m.bias.data.zero_()