# encoding: utf-8

'''
@author: double4tar
@contact: double4tar@gmail.com
@file: target.py
@time: 19-1-16 下午5:48
@desc: modify & copy crom https://github.com/amdegroot/ssd.pytorch
'''

def printTarget():
    print("TQDM")
    print("Validation test")                # done
    print("Visualisation layers")
    print("Self-defined network")           # done but bad result
    print("Self-defined augmentation")

def printProblems():
    print("data augment, mean & std")       # not really
    print("data augment, random crop")      # not really
    print("process flow, first zero_gradient")
    print("lr strategy")                    # not really 0.08 is too high, 0.01 is good
    print("self-define network")            # seems to be the problem res18 is better than self-defined network
    print("batchSize")                      # seems relative



