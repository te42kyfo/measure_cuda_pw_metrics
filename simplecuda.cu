#include <cupti_target.h>
#include <cupti_profiler_target.h>
#include <nvperf_host.h>
#include <cuda.h>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <functional>
#include "Metric.hpp"
#include "Eval.hpp"
//#include <FileOp.h>

#define NVPW_API_CALL(apiFuncCall)                                             \
do {                                                                           \
    NVPA_Status _status = apiFuncCall;                                         \
    if (_status != NVPA_STATUS_SUCCESS) {                                      \
        fprintf(stderr, "%s:%d: error: function %s failed with error %d.\n",   \
                __FILE__, __LINE__, #apiFuncCall, _status);                    \
        exit(-1);                                                              \
    }                                                                          \
} while (0)

#define CUPTI_API_CALL(apiFuncCall)                                            \
do {                                                                           \
    CUptiResult _status = apiFuncCall;                                         \
    if (_status != CUPTI_SUCCESS) {                                            \
        fprintf(stderr, "%s:%d: error: function %s failed with error %d.\n",   \
                __FILE__, __LINE__, #apiFuncCall, _status);                    \
        exit(-1);                                                              \
    }                                                                          \
} while (0)

#define DRIVER_API_CALL(apiFuncCall)                                           \
do {                                                                           \
    CUresult _status = apiFuncCall;                                            \
    if (_status != CUDA_SUCCESS) {                                             \
        fprintf(stderr, "%s:%d: error: function %s failed with error %d.\n",   \
                __FILE__, __LINE__, #apiFuncCall, _status);                    \
        exit(-1);                                                              \
    }                                                                          \
} while (0)

#define RUNTIME_API_CALL(apiFuncCall)                                          \
do {                                                                           \
    cudaError_t _status = apiFuncCall;                                         \
    if (_status != cudaSuccess) {                                              \
        fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",   \
                __FILE__, __LINE__, #apiFuncCall, cudaGetErrorString(_status));\
        exit(-1);                                                              \
    }                                                                          \
} while (0)

static int numRanges = 4;
#define METRIC_NAME "smsp__warps_launched.avg+"

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

bool CreateCounterDataImage(
    std::vector<uint8_t>& counterDataImage,
    std::vector<uint8_t>& counterDataScratchBuffer,
    std::vector<uint8_t>& counterDataImagePrefix)
{

    CUpti_Profiler_CounterDataImageOptions counterDataImageOptions;
    counterDataImageOptions.pCounterDataPrefix = &counterDataImagePrefix[0];
    counterDataImageOptions.counterDataPrefixSize = counterDataImagePrefix.size();
    counterDataImageOptions.maxNumRanges = numRanges;
    counterDataImageOptions.maxNumRangeTreeNodes = numRanges;
    counterDataImageOptions.maxRangeNameLength = 64;

    CUpti_Profiler_CounterDataImage_CalculateSize_Params calculateSizeParams = {CUpti_Profiler_CounterDataImage_CalculateSize_Params_STRUCT_SIZE};

    calculateSizeParams.pOptions = &counterDataImageOptions;
    calculateSizeParams.sizeofCounterDataImageOptions = CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE;

    CUPTI_API_CALL(cuptiProfilerCounterDataImageCalculateSize(&calculateSizeParams));

    CUpti_Profiler_CounterDataImage_Initialize_Params initializeParams = {CUpti_Profiler_CounterDataImage_Initialize_Params_STRUCT_SIZE};
    initializeParams.sizeofCounterDataImageOptions = CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE;
    initializeParams.pOptions = &counterDataImageOptions;
    initializeParams.counterDataImageSize = calculateSizeParams.counterDataImageSize;

    counterDataImage.resize(calculateSizeParams.counterDataImageSize);
    initializeParams.pCounterDataImage = &counterDataImage[0];
    CUPTI_API_CALL(cuptiProfilerCounterDataImageInitialize(&initializeParams));

    CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params scratchBufferSizeParams = {CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params_STRUCT_SIZE};
    scratchBufferSizeParams.counterDataImageSize = calculateSizeParams.counterDataImageSize;
    scratchBufferSizeParams.pCounterDataImage = initializeParams.pCounterDataImage;
    CUPTI_API_CALL(cuptiProfilerCounterDataImageCalculateScratchBufferSize(&scratchBufferSizeParams));

    counterDataScratchBuffer.resize(scratchBufferSizeParams.counterDataScratchBufferSize);

    CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params initScratchBufferParams = {CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params_STRUCT_SIZE};
    initScratchBufferParams.counterDataImageSize = calculateSizeParams.counterDataImageSize;

    initScratchBufferParams.pCounterDataImage = initializeParams.pCounterDataImage;
    initScratchBufferParams.counterDataScratchBufferSize = scratchBufferSizeParams.counterDataScratchBufferSize;
    initScratchBufferParams.pCounterDataScratchBuffer = &counterDataScratchBuffer[0];

    CUPTI_API_CALL(cuptiProfilerCounterDataImageInitializeScratchBuffer(&initScratchBufferParams));

    return true;
}

bool runTest(std::function<double()> runPass, CUdevice cuDevice,
             std::vector<uint8_t>& configImage,
             std::vector<uint8_t>& counterDataScratchBuffer,
             std::vector<uint8_t>& counterDataImage,
             CUpti_ProfilerReplayMode profilerReplayMode,
             CUpti_ProfilerRange profilerRange)
{

    CUcontext cuContext;
    DRIVER_API_CALL(cuCtxCreate(&cuContext, 0, cuDevice));

    CUpti_Profiler_BeginSession_Params beginSessionParams = {CUpti_Profiler_BeginSession_Params_STRUCT_SIZE};
    CUpti_Profiler_SetConfig_Params setConfigParams = {CUpti_Profiler_SetConfig_Params_STRUCT_SIZE};
    CUpti_Profiler_EnableProfiling_Params enableProfilingParams = {CUpti_Profiler_EnableProfiling_Params_STRUCT_SIZE};
    CUpti_Profiler_DisableProfiling_Params disableProfilingParams = {CUpti_Profiler_DisableProfiling_Params_STRUCT_SIZE};

    beginSessionParams.ctx = NULL;
    beginSessionParams.counterDataImageSize = counterDataImage.size();
    beginSessionParams.pCounterDataImage = &counterDataImage[0];
    beginSessionParams.counterDataScratchBufferSize = counterDataScratchBuffer.size();
    beginSessionParams.pCounterDataScratchBuffer = &counterDataScratchBuffer[0];
    beginSessionParams.range = profilerRange;
    beginSessionParams.replayMode = profilerReplayMode;
    beginSessionParams.maxRangesPerPass = numRanges;
    beginSessionParams.maxLaunchesPerPass = numRanges;

    CUPTI_API_CALL(cuptiProfilerBeginSession(&beginSessionParams));

    setConfigParams.pConfig = &configImage[0];
    setConfigParams.configSize = configImage.size();

    if(profilerReplayMode == CUPTI_KernelReplay)    /* Profile in KernelReplayMode */
    {
        setConfigParams.passIndex = 0;
        CUPTI_API_CALL(cuptiProfilerSetConfig(&setConfigParams));
        CUPTI_API_CALL(cuptiProfilerEnableProfiling(&enableProfilingParams));
        runPass();
        CUPTI_API_CALL(cuptiProfilerDisableProfiling(&disableProfilingParams));
    }
    else if(profilerReplayMode == CUPTI_UserReplay) /* Profiler in UserReplayMode */
    {
        setConfigParams.passIndex = 0;
        CUPTI_API_CALL(cuptiProfilerSetConfig(&setConfigParams));
        /* User takes the resposiblity of replaying the kernel launches */
        CUpti_Profiler_BeginPass_Params beginPassParams = {CUpti_Profiler_BeginPass_Params_STRUCT_SIZE};
        CUpti_Profiler_EndPass_Params endPassParams = {CUpti_Profiler_EndPass_Params_STRUCT_SIZE};
        do
        {
            CUPTI_API_CALL(cuptiProfilerBeginPass(&beginPassParams));
            {
                CUPTI_API_CALL(cuptiProfilerEnableProfiling(&enableProfilingParams));
                runPass();
                CUPTI_API_CALL(cuptiProfilerDisableProfiling(&disableProfilingParams));
            }
            CUPTI_API_CALL(cuptiProfilerEndPass(&endPassParams));
        }while(!endPassParams.allPassesSubmitted);
        CUpti_Profiler_FlushCounterData_Params flushCounterDataParams = {CUpti_Profiler_FlushCounterData_Params_STRUCT_SIZE};
        CUPTI_API_CALL(cuptiProfilerFlushCounterData(&flushCounterDataParams));
    }
    CUpti_Profiler_UnsetConfig_Params unsetConfigParams = {CUpti_Profiler_UnsetConfig_Params_STRUCT_SIZE};
    CUPTI_API_CALL(cuptiProfilerUnsetConfig(&unsetConfigParams));
    CUpti_Profiler_EndSession_Params endSessionParams = {CUpti_Profiler_EndSession_Params_STRUCT_SIZE};
    CUPTI_API_CALL(cuptiProfilerEndSession(&endSessionParams));

    DRIVER_API_CALL(cuCtxDestroy(cuContext));

    return true;
}

double measureMetric(std::function<double()> runPass, std::vector<std::string> metricNames) {
    cudaFree(0);

    int deviceNum = 0;
    CUdevice cuDevice;
    int computeCapabilityMajor = 0, computeCapabilityMinor = 0;
    DRIVER_API_CALL(cuDeviceGet(&cuDevice, deviceNum));
    DRIVER_API_CALL(cuDeviceGetAttribute(&computeCapabilityMajor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice));
    DRIVER_API_CALL(cuDeviceGetAttribute(&computeCapabilityMinor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice));
    if(computeCapabilityMajor < 7) {
      printf("Sample unsupported on Device with compute capability < 7.0\n");
      return -2.0;
    }

    std::vector<uint8_t> counterDataImagePrefix;
    std::vector<uint8_t> configImage;
    std::vector<uint8_t> counterDataImage;
    std::vector<uint8_t> counterDataScratchBuffer;

    CUpti_Profiler_Initialize_Params profilerInitializeParams = {CUpti_Profiler_Initialize_Params_STRUCT_SIZE};
    CUPTI_API_CALL(cuptiProfilerInitialize(&profilerInitializeParams));
    /* Get chip name for the cuda  device */
    CUpti_Device_GetChipName_Params getChipNameParams = { CUpti_Device_GetChipName_Params_STRUCT_SIZE };
    getChipNameParams.deviceIndex = deviceNum;
    CUPTI_API_CALL(cuptiDeviceGetChipName(&getChipNameParams));
    std::string chipName(getChipNameParams.pChipName);

    /* Generate configuration for metrics, this can also be done offline*/
    NVPW_InitializeHost_Params initializeHostParams = { NVPW_InitializeHost_Params_STRUCT_SIZE };
    NVPW_API_CALL(NVPW_InitializeHost(&initializeHostParams));
    if (metricNames.size()) {
        if(!NV::Metric::Config::GetConfigImage(chipName, metricNames, configImage)) {
            std::cout << "Failed to create configImage" << std::endl;
            return -1.0;
        }
        if(!NV::Metric::Config::GetCounterDataPrefixImage(chipName, metricNames, counterDataImagePrefix)) {
            std::cout << "Failed to create counterDataImagePrefix" << std::endl;
            return -1.0;
        }
    } else {
        std::cout << "No metrics provided to profile" << std::endl;
        return -1.0;
    }




    CUpti_ProfilerReplayMode profilerReplayMode = CUPTI_KernelReplay;
    CUpti_ProfilerRange profilerRange = CUPTI_AutoRange;

    if(!CreateCounterDataImage(counterDataImage, counterDataScratchBuffer, counterDataImagePrefix)) {
        std::cout << "Failed to create counterDataImage" << std::endl;
    }
    runTest(runPass,  cuDevice, configImage, counterDataScratchBuffer, counterDataImage, profilerReplayMode, profilerRange);


    CUpti_Profiler_DeInitialize_Params profilerDeInitializeParams = {CUpti_Profiler_DeInitialize_Params_STRUCT_SIZE};
    CUPTI_API_CALL(cuptiProfilerDeInitialize(&profilerDeInitializeParams));

    NV::Metric::Eval::PrintMetricValues(chipName, counterDataImage, metricNames);

    return 0.0;
}

int main(int argc, char* argv[]) {
    measureMetric(VectorAddSubtract, { "dram__bytes_write.sum.per_second", "dram__bytes_read.sum.per_second"});



    return 0;
}
