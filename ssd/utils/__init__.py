# encoding: utf-8

'''
@author: double4tar
@contact: double4tar@gmail.com
@file: __init__.py
@time: 18-12-26 下午3:23
@desc:
'''
import os
import logging
import time

def getTimeString():
    timeString = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    return timeString

logging.basicConfig(filename=os.getcwd() + "/../log/" + getTimeString() + ".log")
bannerLogger = logging.getLogger()
bannerLogger.setLevel(logging.INFO)

def console_output():
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    # formatter = logging.Formatter('LINE %(lineno)-4d : %(levelname)-8s %(message)s')
    # tell the handler to use this format
    # console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)