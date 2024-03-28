import tensorrt as trt

def export_tensorrt_engine(onnx_file_path, engine_path):
    '''
    通过加载onnx文件，构建engine
    :param onnx_file_path: onnx文件路径
    :return: engine
    '''

    logger = trt.Logger(trt.Logger.WARNING) 

    builder = trt.Builder(logger) 
    # builder.fp16_mode = True  # 设置为使用float16精度
    # builder.set_flag(trt.BuilderFlag.FP16)

    # 预创建网络 
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) 
    # 加载onnx解析器 
    parser = trt.OnnxParser(network, logger) 
    success = parser.parse_from_file(onnx_file_path) 
    for idx in range(parser.num_errors):
        print(parser.get_error(idx))
    if not success: 
        pass  # Error handling code here 
    
    # builder配置 
    config = builder.create_builder_config()
    profile = builder.create_optimization_profile()     
    profile.set_shape("input", (1, 3, 1080, 1920),(1, 3, 1080, 1920),(1, 3, 1080, 1920))

    config.add_optimization_profile(profile)

    config.set_flag(trt.BuilderFlag.FP16)

    # 分配显存作为工作区间，一般建议为显存一半的大小 
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30) # 1 GiB
    serialized_engine = builder.build_serialized_network(network, config) 
    # 序列化生成engine文件 
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
        print("generate file success!")


if __name__ =="__main__":
    onnx_path1 = 'onnx_res/retnet_0314.onnx'
    engine_path = 'retnet_0314.engine'
    export_tensorrt_engine(onnx_path1, engine_path)
