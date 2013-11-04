#!/usr/bin/python

__author__    = "Eric Chiang"
__copyright__ = "Copyright 2013, Eric Chiang"
__email__     = "eric.chiang.m@gmail.com"

__license__   = "GPL"
__version__   = "3.0"

from threading import Thread
from datetime import datetime
import math
import numpy

def RMSE(pred,actual):
    return numpy.sqrt(meanSquaredError(pred,actual))

"""
Calculate mean squared error for a set of values
"""
def MSE(pred,actual):
    return numpy.mean((pred - actual)**2)

def printInfo(message):
    t = datetime.now().isoformat().replace('T',' ').split('.')[0]
    print '[INFO %s] %s' % (t,message,)
