#include <iostream>
#include "measureMetricPW.hpp"


 // Device code
  __global__ void VecAdd(const int* A, const int* B, int* C, int N)
 {
     int i = blockDim.x * blockIdx.x + threadIdx.x;
     if (i < N)
         C[i] = A[i] + B[i];
 }


  __global__ void VecMul(const int* A, const int* B, int* C, int N)
 {
     int i = blockDim.x * blockIdx.x + threadIdx.x;
     if (i < N)
         C[i] = A[i] * B[i];
 }

 // Device code
  __global__ void VecSub(const int* A, const int* B, int* C, int N)
 {
     int i = blockDim.x * blockIdx.x + threadIdx.x;
     if (i < N)
         C[i] = A[i] - B[i];
 }



static double VectorAddSubtract()
{
  int N = 500000000;
  size_t size = N * sizeof(int);
  int threadsPerBlock = 0;
  int blocksPerGrid = 0;
  int *d_A, *d_B, *d_C, *d_D, *d_E;

  // Allocate vectors in device memory
  cudaMalloc((void**)&d_A, size);
  cudaMalloc((void**)&d_B, size);
  cudaMalloc((void**)&d_C, size);
  cudaMalloc((void**)&d_D, size);
  cudaMalloc((void**)&d_E, size);


  // Invoke kernel
  threadsPerBlock = 256;
  blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
  printf("Launching kernel: blocks %d, thread/block %d\n",
         blocksPerGrid, threadsPerBlock);


  VecMul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_E, N);
  VecSub<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_D, N);
  VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaFree(d_D);
  cudaFree(d_E);
  return 0.0;
}


int main(int argc, char* argv[]) {
    measureMetricStart({ "dram__bytes_write.sum.per_second", "dram__bytes_read.sum.per_second"});
    VectorAddSubtract();
    measureMetricStop();

    measureMetric(VectorAddSubtract, { "dram__bytes_write.sum.per_second", "dram__bytes_read.sum.per_second"});

    return 0;
}
