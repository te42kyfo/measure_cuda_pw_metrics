#!/usr/bin/env python3

import pycuda.driver as drv

import os
from ctypes import *

#my_functions = CDLL("/usr/lib/x86_64-linux-gnu/libstdc++.so.6", mode=RTLD_GLOBAL)
my_functions = CDLL("libcudart.so", mode=RTLD_GLOBAL)
my_functions = CDLL("./measureMetricPW.so", mode=RTLD_GLOBAL)

def measureBandwidthStart():
    drv.Context.synchronize()
    my_functions.measureBandwidthStart()
    drv.Context.synchronize()

def measureMetricStop():    
     drv.Context.synchronize()
     my_functions.measureMetricStop()
     drv.Context.synchronize()
