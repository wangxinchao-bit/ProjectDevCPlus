// 伪代码示例
#include <fstream>
#include <iostream>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>

#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <iostream>
#include <opencv2/opencv.hpp> 
#include <NvInferRuntime.h>   
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
    }

    void setInputDataShape(int height, int width, int channels);

    void setInputOutputAddress(float *input, float *output);

public:
    nvinfer1::IExecutionContext *context;

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


__global__ void imageConversion(float *inputImage, float *outputImage, int height, int width, int channels)
{
    /**
     * 
     *  这个是将BGR图像中的mat.data  BGRBGRBGR排列转换为 BBBBBBGGGGGRRRR排列的方式
     *  其中 mat.data的范围是（0-1)
     * 
     * 
     * */
    // 获取线程的全局索引
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;

    // 计算一维数组中的索引
    int index = (idy * width + idx) * channels;

    // 检查索引是否在图像范围内
    if (idx < width && idy < height)
    {
        // 将BGR值存储到一维数组中
        outputImage[idy * width + idx] = inputImage[index];                          // Blue
        outputImage[height * width + idy * width + idx] = inputImage[index + 1];     // Green
        outputImage[2 * height * width + idy * width + idx] = inputImage[index + 2]; // Red
    }
}

// 输入是BGR flaot 
__global__ void rearrangeChannels(const float *inputData, float *outputData, int height, int width) {
    // 获取线程的全局索引
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;

    // 计算输出数组中的索引
    int outputIndex = (idy * width + idx) * 3;

    // 检查索引是否在图像范围内
    if (idx < width && idy < height) {
        // 计算输入数组中的索引
        int inputIndex = idy * width + idx;

        // 将 BBBBGGGGRRRR 转换为 RGBRGBRGBRGBRGBRGRB
        outputData[outputIndex] = inputData[inputIndex + 2* height * width] *255 ;     //Red
        outputData[outputIndex + 1] = inputData[inputIndex + height * width] *255;    // Green
        outputData[outputIndex + 2] =inputData[inputIndex] *255;                      // Blue
    }
}


int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <engine_path> <image_path>" << std::endl;
        return 1;
    }
    std::string enginePath = argv[1];
    std::string imagePath = argv[2];
    std:: string resName = argv[3];
    // 0.初始化engine 和context 操作从而便于后续的运行和推理操作
    TensorRTInference trtInference(enginePath);
    // 1.data Prepapre操作
    cv::Mat inputImage = cv::imread(imagePath, cv::IMREAD_UNCHANGED);
    if (inputImage.empty())
    {
        std::cerr << "Could not open or find the input image" << std::endl;
        return false;
    }
    
    //  首先BGR化，然后再进行归一化浮点数操作
    cv::cvtColor(inputImage, inputImage, cv::COLOR_RGBA2BGR);
    inputImage.convertTo(inputImage, CV_32FC3, 1.0 / 255.);

    //  获取图像信息以及设置context 信息操作和初始化
    const int batchSize = 1;
    int height = inputImage.rows;
    int width = inputImage.cols;
    int channels = inputImage.channels();

    int model_channels = 3;

    const int outputHeight = height * 2; 
    const int outputWidth = width * 2; 

    trtInference.setInputDataShape(height, width, model_channels);
    const size_t inputSize = batchSize * channels * height * width * sizeof(float);
    const size_t outputSize = batchSize * channels * outputHeight * outputWidth* sizeof(float);

    float *gpuInData;
    float *gputOutData;
    float *tmpImage;

    cudaMalloc(&tmpImage, channels * outputHeight * outputWidth * sizeof(float));

    cudaMalloc(&gpuInData, inputSize);
    cudaMalloc(&gputOutData, outputSize);

    trtInference.context->setTensorAddress("input", gpuInData);
    trtInference.context->setTensorAddress("output", gputOutData);

    cudaMemcpy(tmpImage, inputImage.data, height * width * channels * sizeof(float), cudaMemcpyHostToDevice);
 

    dim3 blockDim(32, 32);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    

    imageConversion<<<gridDim, blockDim>>>(tmpImage, gpuInData, height, width, channels);
 
   /* 
    2. 推理操作
    */
    auto start = std::chrono::steady_clock::now();
    trtInference.context->enqueueV3(0);
    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Spend Time:" << duration.count() << std::endl;
    /*
        3. 推理结果的后处理操作
    */
    dim3 blockResDim(32, 32);
    dim3 gridDimRes((outputWidth + blockResDim.x - 1) / blockResDim.x, (outputHeight + blockResDim.y - 1) / blockResDim.y);

    rearrangeChannels <<<gridDimRes, blockResDim>>>(gputOutData, tmpImage, outputHeight, outputWidth);
    cv::Mat reconstructedImage(outputHeight, outputWidth, CV_32FC3);

    cudaMemcpy(reconstructedImage.data, tmpImage, outputHeight * outputWidth * channels * sizeof(float), cudaMemcpyDeviceToHost);
   
    
    //  最后输出的图像是BGR图像
    std::string reconstructedImageImagePath = resName;
    cv::imwrite(reconstructedImageImagePath, reconstructedImage);

    //  for (int i = 0; i < 5; ++i) {
    //     for (int j = 0; j < 5; ++j) {
    //         cv::Vec3f pixel = reconstructedImage.at<cv::Vec3f>(i, j);
    //         uchar blue = pixel[0];
    //         uchar green = pixel[1];
    //         uchar red = pixel[2];
    //         // 输出像素值
    //         std::cout << "Pixel at (" << i << ", " << j << "): "
    //                   << "B=" << static_cast<float>(blue) << ", "
    //                   << "G=" << static_cast<float>(green) << ", "
    //                   << "R=" << static_cast<float>(red) << std::endl;
    //     }
    // }

    // cv::Mat intImage;
    // reconstructedImage.convertTo(intImage, CV_8UC3);
    // std::string resPath = "int8res.png";
    // cv::imwrite(resPath, reconstructedImage);

    return 0;
}
