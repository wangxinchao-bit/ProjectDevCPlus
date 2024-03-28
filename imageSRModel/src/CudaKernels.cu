#include "CudaKernels.cuh"


// 输入信息的处理转换为BBBBBGGGGRRRR
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
     *  将 BBBBGGGGRRRR 转换为 BGRBGRBGRBGRBGRBGR float-255
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
    }
}

__global__ void rearrangeBGRToBGR1(const float *inputData, float *outputData, int height, int width)
{
    /**
     *  // 将 BBBBGGGGRRRR 转换为 BGRBGRBGRBGRBGRBGR(float 0-1)
     * 
     * */
    // 获取线程的全局索引
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;
    
    int outputIndex = (idy * width + idx) * 3;
    if (idx < width && idy < height)
    {
        int inputIndex = idy * width + idx;
        // 将 BBBBGGGGRRRR 转换为 BGRBGRBGRBGRBGRBGR(float 0-1)
        outputData[outputIndex] = inputData[inputIndex];                           // Blue
        outputData[outputIndex + 1] = inputData[inputIndex + height * width] ;     // Green
        outputData[outputIndex + 2] = inputData[inputIndex + 2 * height * width];  // Red
    }
}


void runImageConversion(float * temp,float * gpuInData, int width,int height) {
    int channels =3;
    dim3 blockDim(32, 32);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);
    imageConversion<<<gridDim, blockDim>>>(temp, gpuInData, height, width, channels);
    cudaDeviceSynchronize();
}

void runRearrangeChannels(float * temp,float *gputOutData , int outputWidth,int outputHeight){
    dim3 blockResDim(32, 32);
    dim3 gridDimRes((outputWidth + blockResDim.x - 1) / blockResDim.x, (outputHeight + blockResDim.y - 1) / blockResDim.y);
    rearrangeChannels<<<gridDimRes, blockResDim>>>(gputOutData, temp, outputHeight, outputWidth);
    
	cudaDeviceSynchronize();
}


// __global__ void rearrangeBGRToRBG255(const float *inputData, float *outputData, int height, int width)
// {
//     /**
//      *
//      *  将 BBBBGGGGRRRR 转换为 BGRBGRBGRBGRBGRBGR
//      *  Input: 输入的数据范围 
//      * */
//     // 获取线程的全局索引
//     int idx = threadIdx.x + blockIdx.x * blockDim.x;
//     int idy = threadIdx.y + blockIdx.y * blockDim.y;
    
//     int outputIndex = (idy * width + idx) * 3;
//     if (idx < width && idy < height)
//     {
//         int inputIndex = idy * width + idx;
//         // 将 BBBBGGGGRRRR 转换为 RGBRGBRGBRGBRGBRGRB
//         outputData[outputIndex ] =     inputData[inputIndex + 2* height * width] *255 ; // R ed
//         outputData[outputIndex + 1] =  inputData[inputIndex + height * width] *255;     // Green
//         outputData[outputIndex  +2 ] = inputData[inputIndex]  * 255;                    //Blue
        
//     }
// }

// __global__ void rearrangeBGRToRBG(const float *inputData, float *outputData, int height, int width)
// {
//     /**
//      *
//      *  将 BBBBGGGGRRRR 转换为 BGRBGRBGRBGRBGRBGR
//      *  Input: 输入的数据范围 
//      * */
//     // 获取线程的全局索引
//     int idx = threadIdx.x + blockIdx.x * blockDim.x;
//     int idy = threadIdx.y + blockIdx.y * blockDim.y;
    
//     int outputIndex = (idy * width + idx) * 3;
//     if (idx < width && idy < height)
//     {
//         int inputIndex = idy * width + idx;
//         // 将 BBBBGGGGRRRR 转换为 RGBRGBRGBRGBRGBRGRB
//         outputData[outputIndex ] =     inputData[inputIndex + 2* height * width] *255 ; // R ed
//         outputData[outputIndex + 1] =  inputData[inputIndex + height * width] *255;     // Green
//         outputData[outputIndex  +2 ] = inputData[inputIndex]  * 255;                    //Blue
        
//     }
// }



__global__ void convertAndNormalizeKernel(unsigned char *input, float* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    
    int outputIndex = (y * width + x) *3  ;

    if(x < width && y < height) {
        int index = (y * width + x) *3  ;
        output[outputIndex] = input[index] /255.0f * 255.0;
        output[outputIndex + 1] = input[index + 1] /255.0f * 255.0;
        output[outputIndex + 2] = input[index + 2]  /255.0f * 255.0;
    }
}


void convertAndNormalizeWithCuda(unsigned char * d_input,float *d_output , int width,int height){
    dim3 blockResDim(32, 32);
    dim3 gridDimRes((width + blockResDim.x - 1) / blockResDim.x, (height + blockResDim.y - 1) / blockResDim.y);
    convertAndNormalizeKernel<<<gridDimRes, blockResDim>>>(d_input, d_output, width, height);
}