#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

// CUDA运行时库
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <iostream>

// CUDA内核函数的声明
// __global__ void imageConversion(float *inputImage, float *outputImage, int height, int width, int channels);
// __global__ void rearrangeChannels(const float *inputData, float *outputData, int height, int width);
// __global__ void rearrangeBGRToBGR1(const float *inputData, float *outputData, int height, int width);
// __global__ void rearrangeBGRToRBG255(const float *inputData, float *outputData, int height, int width);

extern "C" void runImageConversion(float * temp,float * gpuInData, int width,int height);
extern "C" void runRearrangeChannels(float * temp,float *gputOutData , int outputWidth,int outputHeight);

extern "C" void convertAndNormalizeWithCuda(unsigned char * d_input,float *d_output , int width,int height);
#endif // CUDA_KERNELS_H