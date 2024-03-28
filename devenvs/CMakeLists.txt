cmake_minimum_required(VERSION 3.5)
project(MyCudaProject)

# 添加C++11支持
set(CMAKE_CXX_STANDARD 11)

# 查找OpenCV库
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# 查找 CUDA
find_package(CUDA REQUIRED)
cuda_add_executable(MyCudaProject warpScheduler.cu)
target_link_libraries(MyCudaProject ${CUDA_LIBRARIES})

#添加 TensorRT
# add TensorRT8
set(SAMPLES_COMMON_SOURCES "/home/wxcwxc/clibs/TensorRT-8.6.1.6/samples/common/logger.cpp")
include_directories(/home/wxcwxc/clibs/TensorRT-8.6.1.6/include)
include_directories(/home/wxcwxc/clibs/TensorRT-8.6.1.6/samples/common)
set(TENSORRT_LIB_PATH "/home/wxcwxc/clibs/TensorRT-8.6.1.6/lib")
file(GLOB LIBS "${TENSORRT_LIB_PATH}/*.so")

target_link_libraries(MyCudaProject ${LIBS}  ${OpenCV_LIBS})


# cuda_add_executable(Myrun 0315interface.cu)
# target_link_libraries(Myrun ${CUDA_LIBRARIES})
# target_link_libraries(Myrun ${LIBS}  ${OpenCV_LIBS})

# 设置输出的路径信息
set(MY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${MY_OUTPUT_DIRECTORY})
set(LIBRARY_OUTPUT_PATH ${MY_OUTPUT_DIRECTORY})



