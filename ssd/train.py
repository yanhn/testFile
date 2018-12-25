# encoding: utf-8

'''
@author: double4tar
@contact: double4tar@gmail.com
@file: train.py
@time: 18-12-25 下午3:42
@desc:
'''

import argparse

def trainSSD():
    pass

def parseParameters():
    parser = argparse.ArgumentParser(
        description='Single Shot MultiBox Detector Training With Pytorch')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parseParameters()
    trainSSD()
    print("Done!")