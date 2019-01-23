# encoding: utf-8

'''
@author: double4tar
@contact: double4tar@gmail.com
@file: __init__.py
@time: 19-1-18 下午12:18
@desc: modify & copy crom https://github.com/amdegroot/ssd.pytorch
'''

def opName(name, layer):
    layer.opName = name
    return layer