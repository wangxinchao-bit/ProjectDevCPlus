cmake_minimum_required(VERSION 3.0)
project(srModel)

set(CMAKE_CXX_STANDARD 11)


#  Add The Library For Opencv
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


# Add The Library for CUDA 
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Add The Library for TensorRT 
set(SAMPLES_COMMON_SOURCES "/home/wxcwxc/clibs/TensorRT-8.6.1.6/samples/common/logger.cpp")
include_directories(/home/wxcwxc/clibs/TensorRT-8.6.1.6/include)
include_directories(/home/wxcwxc/clibs/TensorRT-8.6.1.6/samples/common)
set(TENSORRT_LIB_PATH "/home/wxcwxc/clibs/TensorRT-8.6.1.6/lib")
file(GLOB LIBS "${TENSORRT_LIB_PATH}/*.so")


include_directories(includes)

set(CUDA_SOURCE_FILES "src/CudaKernels.cu")
# cuda_add_executable(testModel ${CUDA_SOURCE_FILES} ...)

file(GLOB SOURCES "src/*.cpp")
cuda_add_executable(testModel ${SOURCES}  ${CUDA_SOURCE_FILES})

target_link_libraries(testModel ${LIBS}  ${OpenCV_LIBS})
target_link_libraries(testModel ${CUDA_LIBRARIES})


# 设置输出的路径信息
set(MY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${MY_OUTPUT_DIRECTORY})
set(LIBRARY_OUTPUT_PATH ${MY_OUTPUT_DIRECTORY})
