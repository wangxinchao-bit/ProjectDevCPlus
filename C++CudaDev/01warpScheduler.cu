
#include <cuda_runtime.h>
#include <iostream>


__global__ void warpReduceSum(int *input, int *result)
{
    int tid = threadIdx.x;
    int sum = input[tid];

    // 对warp内的数据进行归约求和
    // 使用__shfl_down_sync交换warp内的线程数据
    // 每一步都将跨度加倍，直到覆盖整个warp
    for(int offset = warpSize / 2; offset > 0; offset /= 2)
    {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // 将计算结果写回到全局内存，只需要一个线程执行
    if(tid == 0)
    {
        *result = sum;
    }
}

int main()
{
    int numElements = 32; // warp的大小
    size_t size = numElements * sizeof(int);
    int h_data[32];
    int h_result = 0;
    int *d_data, *d_result;

    // 初始化输入数据
    for(int i = 0; i < numElements; ++i)
    {
        h_data[i] = 1; // 以简单的数据进行初始化，方便验证结果
    }

    // 分配设备内存
    cudaMalloc((void **)&d_data, size);
    cudaMalloc((void **)&d_result, sizeof(int));
    // 复制输入数据到设备
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
    // 调用kernel
    warpReduceSum<<<1, numElements>>>(d_data, d_result);
    // 复制结果回主机
    cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    // 打印结果
    std::cout << "Sum: " << h_result << std::endl;

    // 清理
    cudaFree(d_data);
    cudaFree(d_result);

    return 0;
}