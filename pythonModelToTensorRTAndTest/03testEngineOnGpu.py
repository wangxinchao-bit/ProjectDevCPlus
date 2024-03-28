# Load TRT engine

import tensorrt as trt
import numpy as np
from cuda import cudart
import time
import cv2
import numpy as np


engine_path ="/home/wxcwxc/wxcpython/rt4ksr_aoto/retnet_0314.engine"
logger = trt.Logger(trt.Logger.ERROR)
trt.init_libnvinfer_plugins(logger, namespace="")
with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())
context = engine.create_execution_context()

imgpath = '/home/wxcwxc/wxcpython/rt4ksr_aoto/0001x2.png'
img = cv2.imread(imgpath, cv2.IMREAD_COLOR).astype(np.float32) / 255.
img = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_LINEAR)  # Resize if necessary

img = np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))
image_array = np.expand_dims(img, 0)  
input = {
    "input":image_array
}

start = time.time()
input_buffers = {}
for i in range(engine.num_io_tensors):
    name = engine.get_tensor_name(i)
    if engine.get_tensor_mode(name) != trt.TensorIOMode.INPUT:
        continue

    array = input[name]
    dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(name)))
    ## 保持输入张量类型与模型要求的数据类型一致
    array = array.astype(dtype)
    ## numpy 数组的内存布局有可能不是连续的，这里需要转换为连续的内存布局，以便使用指针拷贝
    array = np.ascontiguousarray(array)

    ## cudaMalloc 分配 GPU 内存，返回内存指针和错误码
    err, ptr = cudart.cudaMalloc(array.nbytes)
    if err > 0:
        raise Exception("cudaMalloc failed, error code: {}".format(err))
    ## 暂时保存内存指针，后续还需要释放    
    input_buffers[name] = ptr
    ## cudaMemcpy 将数据从 CPU 拷贝到 GPU，其中 array.ctypes.data 是 numpy 数组的内存指针
    cudart.cudaMemcpy(ptr, array.ctypes.data, array.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
    
    ## set_input_shape 设置输入张量的实际形状，对于 dynamic shape 这一步是必要的，因为动态维度在 ONNX 转换过程中被设置成了 -1，这里不设置将会报错
    context.set_input_shape(name, array.shape)
    context.set_tensor_address(name, ptr)

    class OutputAllocator(trt.IOutputAllocator):
        def __init__(self):
            trt.IOutputAllocator.__init__(self)
            self.buffers = {}
            self.shapes = {}

        def reallocate_output(self, tensor_name, memory, size, alignment):
            ptr = cudart.cudaMalloc(size)[1]
            self.buffers[tensor_name] = ptr
            return ptr

        def notify_shape(self, tensor_name, shape):
            self.shapes[tensor_name] = tuple(shape)


output_allocator = OutputAllocator()
for i in range(engine.num_io_tensors):
    name = engine.get_tensor_name(i)
    if engine.get_tensor_mode(name) != trt.TensorIOMode.OUTPUT:
        continue
    context.set_output_allocator(name, output_allocator)
    start = time.time()
    context.execute_async_v3(0)
    end = time.time()
    print("花费的时间： ",end-start)


output = {}
for name in output_allocator.buffers.keys():
    ptr = output_allocator.buffers[name]
    shape = output_allocator.shapes[name]
    dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(name)))
    nbytes = np.prod(shape) * dtype.itemsize

    output_buffer = np.empty(shape, dtype = dtype)
    cudart.cudaMemcpy(output_buffer.ctypes.data, ptr, nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
    output[name] = output_buffer

for name in input_buffers.keys():
    ptr = input_buffers[name]
    cudart.cudaFree(ptr)

for name in output_allocator.buffers.keys():
    ptr = output_allocator.buffers[name]
    cudart.cudaFree(ptr)

output = output["output"][0]

output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
output = (output.clip(0, 1) * 255.0).round().astype(np.uint8)
cv2.imwrite( f'resRGB---00.png', output)
