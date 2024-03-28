#include <fstream> 
#include <iostream> 
 
#include <NvInfer.h> 
#include <NvOnnxParser.h> 
#include "logger.h"
 
using namespace nvinfer1; 
using namespace nvonnxparser; 
using namespace sample; 
 
int main(int argc, char** argv) 
{ 
        if (argc != 3) {
            std::cerr << "Usage: " << argv[0] << " <input_onnx_file> <output_engine_file>" << std::endl;
            return 1;
        }

        const char* inputOnnxFile = argv[1];
        const char* outputEngineFile = argv[2];
       	// 1.创建构建器的实例
        Logger logger; 

		nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);

		// 2.创建网络定义
		uint32_t flag = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
		nvinfer1::INetworkDefinition* network = builder->createNetworkV2(flag);

		// 3.创建一个 ONNX 解析器来填充网络
		nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger);

		// 4.读取模型文件并处理任何错误
		parser->parseFromFile(inputOnnxFile, static_cast<int32_t>(nvinfer1::ILogger::Severity::kWARNING));
		for (int32_t i = 0; i < parser->getNbErrors(); ++i)
		{
			std::cout << parser->getError(i)->desc() << std::endl;
		}

		// 5.创建一个构建配置，指定 TensorRT 应该如何优化模型
		nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();

        auto profile = builder->createOptimizationProfile();
        profile->setDimensions("input", nvinfer1::OptProfileSelector::kMIN, Dims4(1, 3, 224, 224));
        profile->setDimensions("input", nvinfer1::OptProfileSelector::kOPT, Dims4(1, 3, 1280, 640));
        profile->setDimensions("input", nvinfer1::OptProfileSelector::kMAX, Dims4(1, 3, 1920, 1080));
        config->addOptimizationProfile(profile);
        // 分配显存作为工作区间，一般建议为显存一半的大小
        config->setMaxWorkspaceSize(1 << 30);  // 1 Mi

		// 7.指定配置后，构建引擎
		nvinfer1::IHostMemory* serializedModel = builder->buildSerializedNetwork(*network, *config);

        if (!serializedModel) {
            std::cerr << "Failed to build TensorRT engine." << std::endl;
            return 1;
        }

		// 8.保存TensorRT模型
		std::ofstream p(outputEngineFile, std::ios::binary);
		p.write(reinterpret_cast<const char*>(serializedModel->data()), serializedModel->size());

		// 9.序列化引擎包含权重的必要副本，因此不再需要解析器、网络定义、构建器配置和构建器，可以安全地删除
		delete parser;
		delete network;
		delete config;
		delete builder;

		// 10.将引擎保存到磁盘，并且可以删除它被序列化到的缓冲区
		delete serializedModel;
        
        return 0; 

} 