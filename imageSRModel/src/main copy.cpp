
#include <iostream>
#include <string>
#include <NvInfer.h>
#include <fstream>
#include <iostream>
#include <NvInferRuntime.h>
#include <NvOnnxParser.h>
#include <opencv2/opencv.hpp>
#include "CudaKernels.cuh"
# include "TensorRTInference.h"

# include<chrono>

int mainsr(int argc, char *argv[])
{
    if (argc != 4)
    {
        std::cerr << "Usage: " << argv[0] << " <engine_path> <image_path>" << std::endl;
        return 1;
    }
    std::string enginePath = argv[1];
    std::string imagePath = argv[2];
    std::string resName = argv[3];

    //  模型初始化操作
    const int height = 1080;
    const int width = 1920;
    const int channels = 3 ;
    const int scale = 2 ; 
    int inputSize = height * width * channels * sizeof(float);

    int outputWidth = width << 1;
    int outputHeight = height << 1;
    int outputSize = inputSize << 2;  

    TensorRTInference trtInference(enginePath);
    trtInference.setInputDataShape(height, width, channels);   

    //----测试数据准备操作

    float *gpuInData;
    float *gpuOutData;
    cudaMalloc(&gpuInData, inputSize);
    cudaMalloc(&gpuOutData, outputSize);

    float * temp;
    cudaMalloc(&temp, outputSize);


    /*下面的数据前处理和推理结果的数据后处理需要根据流程的操作从而实现优化
        gpuInData:  BBBBBGGGGGRRRR  float 0-1范围
        gpuOutData: BBBBBGGGGGRRRRR float 0-1

        下面的两个数据处理自行优化:
        runImageConversion:核函数是对BGRBGR float(0-1)数据进行转换为 BBBBBGGGGGRRR 
        runRearrangeChannels:核函数是对BBBGGGGRRR float(0-1)重新转换为 BGRBGRBGR  float(0-255)
    */
    cv::Mat inputImage = cv::imread(imagePath, cv::IMREAD_UNCHANGED);
    if (inputImage.empty())
    {
        std::cerr << "Could not open or find the input image" << std::endl;
        return false;
    }
    cv::cvtColor(inputImage, inputImage, cv::COLOR_RGBA2BGR);
    inputImage.convertTo(inputImage, CV_32FC3, 1.0 / 255.);


    cudaMemcpy(temp, inputImage.data, height * width * channels * sizeof(float), cudaMemcpyHostToDevice);

    runImageConversion(temp, gpuInData, width, height);

    /* 模型推理操作*/
    trtInference.srInference(gpuInData,gpuOutData);

    // 对模型输出的结果进行数据格式转换
    runRearrangeChannels(temp, gpuOutData, outputWidth, outputHeight);


    //----测试 结果保存操作
    cv::Mat reconstructedImage(outputHeight, outputWidth, CV_32FC3);
    cudaMemcpy(reconstructedImage.data, temp, outputHeight * outputWidth * channels * sizeof(float), cudaMemcpyDeviceToHost);
    std::string reconstructedImageImagePath = resName;
    cv::cvtColor(reconstructedImage, reconstructedImage, cv::COLOR_BGR2RGB);
    cv::imwrite("testResult.png", reconstructedImage);

    return 0;
}



#include <opencv2/opencv.hpp>
#include <vector>

int main() {
    cv::Mat image = cv::imread("/home/wxcwxc/wxcpython/dataSets/Set5/HR/baby.png", cv::IMREAD_COLOR);
    if (image.empty())
    {
        std::cerr << "Could not open or find the input image" << std::endl;
        return false;
    }
    // cv::cvtColor(image, image, cv::COLOR_RGBA2BGR);
    image.convertTo(image, CV_32FC3, 1.0 / 255.);


    float * d_input;
    float* d_output;
    int width = image.cols;
    int height = image.rows;
    int channels = 3; 

    cudaMalloc(&d_input, width * height * channels *  sizeof(float));
    cudaMalloc(&d_output, width * height * channels * sizeof(float));

    cudaMemcpy(d_input, image.data, width * height * channels * sizeof(float), cudaMemcpyHostToDevice);


    cv::Mat output(height, width, CV_32FC3);
    cudaMemcpy(output.data, d_input, width * height * channels * sizeof(float), cudaMemcpyDeviceToHost);
    std::string reconstructedImageImagePath = "res.png";
    output.convertTo(output, CV_8UC3, 255.0);
    cv::imwrite(reconstructedImageImagePath, output);

    // auto start =  std::chrono::high_resolution_clock::now();
    // convertAndNormalizeWithCuda(d_input, d_output, width, height);
    // auto end =  std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end-start);
    // std::cout << "执行时间 实现数据尺度的转换时间：" << duration.count() << "毫秒" << std::endl;
    

    // cv::Mat output(height, width, CV_32FC3);
    // cudaMemcpy(output.data, d_output, width * height * channels * sizeof(float), cudaMemcpyDeviceToHost);
    // std::string reconstructedImageImagePath = "res.png";
    // cv::imwrite(reconstructedImageImagePath, output);

    // // cv::Mat convertedImage;
    // // output.convertTo(convertedImage, CV_8UC3, 255.0); // 反归一化并转换为8位无符号整数格式
    // // cv::imwrite("unchar.png", convertedImage);

    // //  // 遍历图像
    // // for (int row = output.cols; row >  output.cols -5; row--) {
    // //     for (int col = output.rows; col > output.rows -5; col--) {
    // //         // 获取(row, col)位置的像素
    // //         Vec3f pixel = output.at<Vec3f>(row, col);
    // //         // pixel[0]是蓝色分量，pixel[1]是绿色分量，pixel[2]是红色分量
    // //         std::cout << "Pixel at (" << row << ", " << col << ") - "
    // //                   << "B: " << (int)pixel[0] << ", "
    // //                   << "G: " << (int)pixel[1] << ", "
    // //                   << "R: " << (int)pixel[2] << std::endl;
    // //     }
    // // }


    return 0;
}
