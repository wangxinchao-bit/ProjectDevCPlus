#include <cuda_runtime.h>
#include <iostream>

// CUDA核函数定义
__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements)
    {
        C[i] = A[i] + B[i];
    }
}

int main()
{
    int numElements = 50000;
    size_t size = numElements * sizeof(float);
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;

    // 分配主机内存
    h_A = (float *)malloc(size);
    h_B = (float *)malloc(size);
    h_C = (float *)malloc(size);

    // 初始化主机数据
    for(int i = 0; i < numElements; ++i)
    {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // 分配设备内存
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // cudaMalloc((void **)&d_A, size);
    // cudaMalloc((void **)&d_B, size);
    // cudaMalloc((void **)&d_C, size);

    // 创建两个流
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    // 复制数据到设备，并在两个不同的流中执行向量加法
    cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(d_B, h_B, size, cudaMemcpyHostToDevice, stream1);
    vectorAdd<<<(numElements + 255) / 256, 256, 0, stream1>>>(d_A, d_B, d_C, numElements);
    
    // 在第二个流中再次执行相同的操作
    // 为了示例简单，这里再次使用同样的输入数据
    cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, stream2);
    cudaMemcpyAsync(d_B, h_B, size, cudaMemcpyHostToDevice, stream2);
    vectorAdd<<<(numElements + 255) / 256, 256, 0, stream2>>>(d_A, d_B, d_C, numElements);

    // 等待流完成
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    // 复制结果回主机
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // 检查错误并打印几个结果
    for (int i = 0; i < 5; ++i)
    {
        std::cout << "C[" << i << "] = " << h_C[i] << "\n";
    }
    
    // 清理
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);

    return 0;
}
