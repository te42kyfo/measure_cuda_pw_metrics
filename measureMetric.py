#!/usr/bin/env python3

import pycuda.driver as drv
import pycuda.autoinit
from pycuda.compiler import SourceModule
from subprocess import run, PIPE
import numpy

import os
from ctypes import *

#my_functions = CDLL("/usr/lib/x86_64-linux-gnu/libstdc++.so.6", mode=RTLD_GLOBAL)
my_functions = CDLL("libcudart.so", mode=RTLD_GLOBAL)
my_functions = CDLL("./measureMetricPW.so", mode=RTLD_GLOBAL)



N = 20000000
A_gpu = drv.mem_alloc(N*8)
B_gpu = drv.mem_alloc(N*8)
C_gpu = drv.mem_alloc(N*8)
D_gpu = drv.mem_alloc(N*8)
E_gpu = drv.mem_alloc(N*8)

text = "__global__ void VecSub(const double* A, const double* B, double* C, int N)  { int i = blockDim.x * blockIdx.x + threadIdx.x; if (i < N) C[i] = A[i] - B[i];}"

mod = SourceModule(text, arch="sm_70", options=["-lineinfo" "-O3" "-w" "-std=c++11"])
function  = mod.get_function("VecSub")
function.prepare(('P', 'P', 'P',  numpy.int32 ))
blockCount = (N-1) // 256 + 1
function.prepared_call((blockCount, 1, 1), (256, 1, 1), A_gpu, B_gpu, C_gpu, numpy.int32(N))
function.prepared_call((blockCount, 1, 1), (256, 1, 1), A_gpu, B_gpu, E_gpu, numpy.int32(N))

drv.Context.synchronize()
my_functions.measureBandwidthStart()
drv.Context.synchronize()
function.prepared_call((blockCount, 1, 1), (256, 1, 1), A_gpu, B_gpu, D_gpu, numpy.int32(N))
drv.Context.synchronize()
my_functions.measureMetricStop()
print("jodel")
my_functions.measureBandwidthStart()
function.prepared_call((blockCount, 1, 1), (256, 1, 1), A_gpu, B_gpu, D_gpu, numpy.int32(N))
my_functions.measureMetricStop()
print("icehockey")
