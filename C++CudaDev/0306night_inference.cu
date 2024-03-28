// 伪代码示例
#include <fstream>
#include <iostream>
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvOnnxParser.h>
#include <opencv2/opencv.hpp>
#include <chrono>


using namespace nvinfer1;
using namespace std;

int calSpendTime()
{
    auto start = std::chrono::steady_clock::now();

    auto end = std::chrono::steady_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Spend Time:" << duration.count() << std::endl;
}

class Logger : public ILogger
{
    void log(Severity severity, const char *msg) noexcept override
    {
        // suppress info-level messages
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
} gLogger;

class TensorRTInference
{
public:
    TensorRTInference(const std::string &enginePath) : enginePath(enginePath)
    {
        runtime = nvinfer1::createInferRuntime(gLogger);
        loadEngine();
    }

    ~TensorRTInference()
    {
        context->destroy();
        engine->destroy();
        runtime->destroy();
        cudaFree(tmpImage);
    }

    void setInputDataShape(int height, int width, int channels);

    void accelorateTempAddress(int outputHeight, int outputWeight, int chanenls);

public:
    nvinfer1::IExecutionContext *context;
    float *tmpImage;

private:
    nvinfer1::IRuntime *runtime;
    nvinfer1::ICudaEngine *engine;

    Logger gLogger;
    std::string enginePath;

    void loadEngine();
};

void TensorRTInference::loadEngine()
{
    std::ifstream engineFile(enginePath, std::ios::binary);
    engineFile.seekg(0, engineFile.end);
    int engineSize = engineFile.tellg();
    engineFile.seekg(0, engineFile.beg);
    char *engineData = new char[engineSize];
    engineFile.read(engineData, engineSize);
    engineFile.close();

    engine = runtime->deserializeCudaEngine(engineData, engineSize);
    context = engine->createExecutionContext();

    if (!engine)
    {
        std::cerr << "Failed to deserialize CUDA engine" << std::endl;
    }
    if (!context)
    {
        std::cerr << "Failed to create execution context" << std::endl;
    }
}

void TensorRTInference::setInputDataShape(int height, int width, int channels = 3)
{
    long unsigned int nIO = engine->getNbIOTensors();
    std::vector<std::string> vTensorName(nIO);
    for (int i = 0; i < nIO; ++i)
    {
        vTensorName[i] = std::string(engine->getIOTensorName(i));
    }
    context->setInputShape(vTensorName[0].c_str(), Dims32{4, {1, channels, height, width}});
}

void TensorRTInference::accelorateTempAddress(int outputHeight, int outputWidth, int channels = 3)
{
    cudaMalloc(&tmpImage, channels * outputHeight * outputWidth * sizeof(float));
}

// 输入信息的处理编程BBBBBGGGGRRRR->
__global__ void imageConversion(float *inputImage, float *outputImage, int height, int width, int channels)
{
    /**
     *
     *  这个是将BGR图像中的mat.data  BGRBGRBGR排列转换为 BBBBBBGGGGGRRRR排列的方式
     *  其中 mat.data的范围是（0-1)
     *
     *
     * */
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;

    int index = (idy * width + idx) * channels;

    if (idx < width && idy < height)
    {
        outputImage[idy * width + idx] = inputImage[index];                          // Blue
        outputImage[height * width + idy * width + idx] = inputImage[index + 1];     // Green
        outputImage[2 * height * width + idy * width + idx] = inputImage[index + 2]; // Red
    }
}

//  返回结果的不同保存方式
__global__ void rearrangeChannels(const float *inputData, float *outputData, int height, int width)
{
    /**
     *
     *  将 BBBBGGGGRRRR 转换为 BGRBGRBGRBGRBGRBGR
     *  Input: 输入的数据范围 
     * */
    // 获取线程的全局索引
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;

    int outputIndex = (idy * width + idx) * 3;
    if (idx < width && idy < height)
    {
        int inputIndex = idy * width + idx;
        // 将 BBBBGGGGRRRR 转换为 BGRBGRBGRBGRBGRBGR
        outputData[outputIndex] = inputData[inputIndex] * 255;                          // Blue
        outputData[outputIndex + 1] = inputData[inputIndex + height * width] * 255;     // Green
        outputData[outputIndex + 2] = inputData[inputIndex + 2 * height * width] * 255; // Red
        // // 将 BBBBGGGGRRRR 转换为 RGBRGBRGBRGBRGBRGRB
        // outputData[outputIndex  +2 ] = static_cast<int> (inputData[inputIndex]  * 255 );                       //Blue
        // outputData[outputIndex + 1] =  static_cast<int>(inputData[inputIndex + height * width] *255 );    // Green
        // outputData[outputIndex ] = static_cast<int>(inputData[inputIndex + 2* height * width] *255 );  // Red
    }
}

__global__ void rearrangeBGRToBGR1(const float *inputData, float *outputData, int height, int width)
{
    /**
     *
     *  将 BBBBGGGGRRRR 转换为 BGRBGRBGRBGRBGRBGR
     *  Input: 输入的数据范围 
     * */
    // 获取线程的全局索引
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;
    
    int outputIndex = (idy * width + idx) * 3;
    if (idx < width && idy < height)
    {
        int inputIndex = idy * width + idx;
        // 将 BBBBGGGGRRRR 转换为 BGRBGRBGRBGRBGRBGR
        outputData[outputIndex] = inputData[inputIndex];                          // Blue
        outputData[outputIndex + 1] = inputData[inputIndex + height * width] ;     // Green
        outputData[outputIndex + 2] = inputData[inputIndex + 2 * height * width]; // Red
        // // 将 BBBBGGGGRRRR 转换为 RGBRGBRGBRGBRGBRGRB
        // outputData[outputIndex  +2 ] = static_cast<int> (inputData[inputIndex]  * 255 );                       //Blue
        // outputData[outputIndex + 1] =  static_cast<int>(inputData[inputIndex + height * width] *255 );    // Green
        // outputData[outputIndex ] = static_cast<int>(inputData[inputIndex + 2* height * width] *255 );  // Red
    }
}

__global__ void rearrangeBGRToRBG255(const float *inputData, float *outputData, int height, int width)
{
    /**
     *
     *  将 BBBBGGGGRRRR 转换为 BGRBGRBGRBGRBGRBGR
     *  Input: 输入的数据范围 
     * */
    // 获取线程的全局索引
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;
    
    int outputIndex = (idy * width + idx) * 3;
    if (idx < width && idy < height)
    {
        int inputIndex = idy * width + idx;
        // 将 BBBBGGGGRRRR 转换为 RGBRGBRGBRGBRGBRGRB
        outputData[outputIndex ] =     inputData[inputIndex + 2* height * width] *255 ; // R ed
        outputData[outputIndex + 1] =  inputData[inputIndex + height * width] *255;     // Green
        outputData[outputIndex  +2 ] = inputData[inputIndex]  * 255;                    //Blue
        
    }
}

__global__ void rearrangeBGRToRBG1(const float *inputData, float *outputData, int height, int width)
{
    /**
     *
     *  将 BBBBGGGGRRRR 转换为 BGRBGRBGRBGRBGRBGR
     *  Input: 输入的数据范围 
     * */
    // 获取线程的全局索引
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;
    
    int outputIndex = (idy * width + idx) * 3;
    if (idx < width && idy < height)
    {
        int inputIndex = idy * width + idx;
        // 将 BBBBGGGGRRRR 转换为 RGBRGBRGBRGBRGBRGRB
        outputData[outputIndex ] =     inputData[inputIndex + 2* height * width] ; // R ed
        outputData[outputIndex + 1] =  inputData[inputIndex + height * width] ;    // Green
        outputData[outputIndex  +2 ] = inputData[inputIndex] ;                     //Blue
        
    }
}

int main(int argc, char *argv[])
{
    if (argc != 4)
    {
        std::cerr << "Usage: " << argv[0] << " <engine_path> <image_path>" << std::endl;
        return 1;
    }
    std::string enginePath = argv[1];
    std::string imagePath = argv[2];
    std::string resName = argv[3];

    cv::Mat inputImage = cv::imread(imagePath, cv::IMREAD_UNCHANGED);
    if (inputImage.empty())
    {
        std::cerr << "Could not open or find the input image" << std::endl;
        return false;
    }

    cv::cvtColor(inputImage, inputImage, cv::COLOR_RGBA2BGR);
    inputImage.convertTo(inputImage, CV_32FC3, 1.0 / 255.);

    const int batchSize = 1;
    int height = inputImage.rows;
    int width = inputImage.cols;
    int channels = inputImage.channels();

    const int outputHeight = height * 2;
    const int outputWidth = width * 2;

    const size_t inputSize = batchSize * channels * height * width * sizeof(float);
    const size_t outputSize = batchSize * channels * outputHeight * outputWidth * sizeof(float);

    float *gpuInData;
    float *gputOutData;
    cudaMalloc(&gpuInData, inputSize);
    cudaMalloc(&gputOutData, outputSize);

    TensorRTInference trtInference(enginePath);

    trtInference.setInputDataShape(height, width, channels);
    trtInference.accelorateTempAddress(outputHeight, outputWidth);

    trtInference.context->setTensorAddress("input", gpuInData);
    trtInference.context->setTensorAddress("output", gputOutData);

    cudaMemcpy(trtInference.tmpImage, inputImage.data, height * width * channels * sizeof(float), cudaMemcpyHostToDevice);

    // 测试
    auto start = std::chrono::steady_clock::now();
    dim3 blockDim(32, 32);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);
    imageConversion<<<gridDim, blockDim>>>(trtInference.tmpImage, gpuInData, height, width, channels);
    trtInference.context->enqueueV3(0);

    dim3 blockResDim(32, 32);
    dim3 gridDimRes((outputWidth + blockResDim.x - 1) / blockResDim.x, (outputHeight + blockResDim.y - 1) / blockResDim.y);

    rearrangeChannels<<<gridDimRes, blockResDim>>>(gputOutData, trtInference.tmpImage, outputHeight, outputWidth);

    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Spend Time:" << duration.count() << std::endl;    

    // 测试
    cv::Mat reconstructedImage(outputHeight, outputWidth, CV_32FC3);
    cudaMemcpy(reconstructedImage.data, trtInference.tmpImage, outputHeight * outputWidth * channels * sizeof(float), cudaMemcpyDeviceToHost);
    std::string reconstructedImageImagePath = resName;
    cv::cvtColor(reconstructedImage, reconstructedImage, cv::COLOR_BGR2RGB);
    cv::imwrite("reconstructedImageImagePath.png", reconstructedImage);

    return 0;
}
