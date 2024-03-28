

// 采用opencv 实现图像的处理转换操作

void ImageCroppingCuda(GpuInfo & src, GpuInfo & dst, int xStart, int yStart, float scaleRatio, void* stream)
{
        //cuda流
        cudaStream_t stream_t = stream ? (cudaStream_t)stream : 0;
        //OpenCV cuda流
        cv::cuda::Stream stream_cv = cv::cuda::StreamAccessor::wrapStream(stream_t);
        int dstWidth = src.W / scaleRatio;
        int dstHeight = src.H / scaleRatio;
        float xOffset = -(float)xStart / (float)src.W * dstWidth;
        float yOffset = -(float)yStart / (float)src.H * dstHeight;
        cv::Mat scale_mat = (Mat_<double>(2, 3) << 1 / scaleRatio, 0, xOffset, 0, 1 / scaleRatio, yOffset);

        cv::cuda::GpuMat* scaleRes;
        if (dst.W != src.W || dst.H != src.H)
        {
                if (dst.gpuptr)
                {
                        cudaFreeAsync(((cv::cuda::GpuMat*)dst.gpuptr)->data, stream_t);
                        dst.gpuptr = nullptr;
                }
        }
        if (!dst.gpuptr)
        {
                unsigned char* memory = nullptr;
                cudaMallocAsync((void**)&memory, src.H * src.W * 4, stream_t);
                scaleRes = new cv::cuda::GpuMat(src.H, src.W, CV_8UC4, (void*)memory, src.W * 4);
                dst.gpuptr = scaleRes;
                dst.H = src.H;
                dst.W = src.W;
                dst.stream = stream;
        }
        else
        {
                scaleRes = (cv::cuda::GpuMat*)dst.gpuptr;
        }
        cv::cuda::warpAffine(*(cv::cuda::GpuMat*)src.gpuptr, *(cv::cuda::GpuMat*)scaleRes, scale_mat, Size(src.W, src.H), INTER_CUBIC, 0, Scalar(0, 0, 0, 255), stream_cv);
}


// 采用CUDA核函数进行问题的解决操作，从而实现计算和转换 

__global__ void drawPointsKernel(cv::cuda::PtrStepSz<uchar4>src, Geometry::Point2d* srcPoints, const int size)
{
        int i = threadIdx.x + blockDim.x * blockIdx.x;
        int j = threadIdx.y + blockDim.y * blockIdx.y;
        if (j >= src.rows || i >= src.cols)
        {
                return;
        }
        for (int m = 0; m < size; m++)
        {
                if (i< srcPoints[m].x + 5 && i>srcPoints[m].x - 5 && j > srcPoints[m].y - 5 && j < srcPoints[m].y + 5)
                {
                        src(j, i).x = 0;
                        src(j, i).y = 0;
                        src(j, i).z = 255;
                }
                
        }
}


void DrawPointsCuda(GpuInfo & src, oneDArrayPoint2d srcPoints, void* stream)
{
        cudaStream_t stream_t;
        if (stream != nullptr)
        {
                stream_t = (cudaStream_t)stream;
        }
        else
        {
                stream_t = 0;
        }
        Geometry::Point2d* srcpoints;
        cudaMallocAsync((void**)&srcpoints, sizeof(Geometry::Point2d) * srcPoints.oneDSize, stream_t);
        cudaMemcpyAsync(srcpoints, srcPoints.oneDptr, sizeof(Geometry::Point2d) * srcPoints.oneDSize, cudaMemcpyHostToDevice, stream_t);
        dim3 threadsPerBlock(32, 32);
        uint block_num_vertical = (src.H + threadsPerBlock.y - 1) / threadsPerBlock.y;
        uint block_num_horizontal = (src.W + threadsPerBlock.x - 1) / threadsPerBlock.x;
        dim3 numBlocks(block_num_horizontal, block_num_vertical);
        drawPointsKernel << < numBlocks, threadsPerBlock, 0, stream_t >> > ((*(cv::cuda::GpuMat*)src.gpuptr), srcpoints, srcPoints.oneDSize);

}