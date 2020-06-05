#!/usr/bin/env python3

import pycuda.driver as drv

import os
from ctypes import *

dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'measureMetricPW.so')

#my_functions = CDLL("/usr/lib/x86_64-linux-gnu/libstdc++.so.6", mode=RTLD_GLOBAL)
my_functions = CDLL("libcudart.so", mode=RTLD_GLOBAL)
my_functions = CDLL(filename, mode=RTLD_GLOBAL)
my_functions.measureMetricStop.restype = py_object


def measureBandwidthStart():
    drv.Context.synchronize()
    my_functions.measureBandwidthStart()
    drv.Context.synchronize()

def measureMetricStop():    
     drv.Context.synchronize()
     return my_functions.measureMetricStop()
