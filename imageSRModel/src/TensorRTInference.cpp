

#include <fstream>
#include <vector>
#include <iostream>
#include <cuda_runtime_api.h> 
#include "TensorRTInference.h"

// TensorRTInference::TensorRTInference(const std::string &enginePath) 
//     : enginePath(enginePath), context(nullptr), tmpImage(nullptr), runtime(nullptr), engine(nullptr) {
//     runtime = nvinfer1::createInferRuntime(gLogger);
//     loadEngine();
// }
TensorRTInference::TensorRTInference(const std::string &enginePath) 
    : enginePath(enginePath), context(nullptr),  runtime(nullptr), engine(nullptr) {
    runtime = nvinfer1::createInferRuntime(gLogger);
    loadEngine();
}
TensorRTInference::~TensorRTInference() {
    if (context) {
        context->destroy();
    }
    if (engine) {
        engine->destroy();
    }
    if (runtime) {
        runtime->destroy();
    }
    // if (tmpImage) {
    //     cudaFree(tmpImage);
    // }
}

void TensorRTInference::setInputDataShape(int height, int width, int channels) {
    long unsigned int nIO = engine->getNbIOTensors();
    std::vector<std::string> vTensorName(nIO);
    for (int i = 0; i < nIO; ++i) {
        vTensorName[i] = std::string(engine->getIOTensorName(i));
    }
    context->setInputShape(vTensorName[0].c_str(), nvinfer1::Dims32{4, {1, channels, height, width}});
}

// void TensorRTInference::accelorateTempAddress(int outputHeight, int outputWidth, int channels) {
//     cudaMalloc(&tmpImage, channels * outputHeight * outputWidth * sizeof(float));
// }

extern "C" void TensorRTInference::loadEngine() {
    std::ifstream engineFile(enginePath, std::ios::binary);
    if (!engineFile) {
        std::cerr << "Failed to open engine file: " << enginePath << std::endl;
        return;
    }
    engineFile.seekg(0, engineFile.end);
    int engineSize = engineFile.tellg();
    engineFile.seekg(0, engineFile.beg);
    std::vector<char> engineData(engineSize);
    engineFile.read(engineData.data(), engineSize);
    engine = runtime->deserializeCudaEngine(engineData.data(), engineSize, nullptr);
    if (!engine) {
        std::cerr << "Failed to deserialize CUDA engine" << std::endl;
        return;
    }
    context = engine->createExecutionContext();
    if (!context) {
        std::cerr << "Failed to create execution context" << std::endl;
        return;
    }
}

extern "C" void TensorRTInference:: srInference(float *gpuInData,float * gpuOutData){
    //  设置输入输出地址
    context->setTensorAddress("input", gpuInData);
    context->setTensorAddress("output", gpuOutData);
    //  推理操作：修改输出地址中存放的数据
    context->enqueueV3(0);
}