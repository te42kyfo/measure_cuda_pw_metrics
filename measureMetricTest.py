#!/usr/bin/env python3

import pycuda.driver as drv
import pycuda.autoinit
from pycuda.compiler import SourceModule
from subprocess import run, PIPE
import numpy

from measureMetric import *

N = 200000000
A_gpu = drv.mem_alloc(N*8)
B_gpu = drv.mem_alloc(N*8)
C_gpu = drv.mem_alloc(N*8)
D_gpu = drv.mem_alloc(N*8)
E_gpu = drv.mem_alloc(N*8)

text = "__global__ void VecSub(const double* A, const double* B, double* C, int N)  { int i = blockDim.x * blockIdx.x + threadIdx.x; if (i < N) C[i] = A[i] - B[i];}"

mod = SourceModule(text, arch="sm_80", options=["-lineinfo" "-O3" "-w" "-std=c++11"])
function  = mod.get_function("VecSub")
function.prepare(('P', 'P', 'P',  numpy.int32 ))
blockCount = (N-1) // 256 + 1
function.prepared_call((blockCount, 1, 1), (256, 1, 1), A_gpu, B_gpu, C_gpu, numpy.int32(N))
function.prepared_call((blockCount, 1, 1), (256, 1, 1), A_gpu, B_gpu, E_gpu, numpy.int32(N))

measureBandwidthStart()
function.prepared_call((blockCount, 1, 1), (256, 1, 1), A_gpu, B_gpu, D_gpu, numpy.int32(N))
values = measureMetricStop()


print( "DRAM LOAD:  {:2.2f} GB, {:5.2f} B/thread".format( values[0] * 1e-9, values[0] / blockCount / 256 ))
print( "DRAM STORE: {:2.2f} GB, {:5.2f} B/thread".format( values[1] * 1e-9, values[1] / blockCount / 256 ))
print( "L2 LOAD:    {:2.2f} GB, {:5.2f} B/thread".format( values[2]*32 * 1e-9, values[2]*32 / blockCount / 256 ))
print( "L2 STORE:   {:2.2f} GB, {:5.2f} B/thread".format( values[3]*32 * 1e-9, values[3]*32 / blockCount / 256 ))
