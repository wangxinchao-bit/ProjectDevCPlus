#pragma once
#include <string>
#include <NvInfer.h>
#include "Logger.h" 

class TensorRTInference {
public:
    TensorRTInference(const std::string &enginePath);
    ~TensorRTInference();
    void setInputDataShape(int height, int width, int channels = 3);
    // void accelorateTempAddress(int outputHeight, int outputWidth, int channels = 3);
    void srInference(float *gpuInData,float * gpuOutData);
    nvinfer1::IExecutionContext *context;
    // float *tmpImage;

private:
    void loadEngine();
    nvinfer1::IRuntime *runtime;
    nvinfer1::ICudaEngine *engine;
    Logger gLogger;
    std::string enginePath;
};
