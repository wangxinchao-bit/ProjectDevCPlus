cmake_minimum_required(VERSION 3.5)
project(ModelInfer)

# 添加C++11支持
set(CMAKE_CXX_STANDARD 11)

# 查找OpenCV库
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


# add TensorRT8
set(SAMPLES_COMMON_SOURCES "/home/wxcwxc/clibs/TensorRT-8.6.1.6/samples/common/logger.cpp")
add_executable(ModelInfer testRT.cpp ${SAMPLES_COMMON_SOURCES})

include_directories(/home/wxcwxc/clibs/TensorRT-8.6.1.6/include)
include_directories(/home/wxcwxc/clibs/TensorRT-8.6.1.6/samples/common)
set(TENSORRT_LIB_PATH "/home/wxcwxc/clibs/TensorRT-8.6.1.6/lib")
file(GLOB LIBS "${TENSORRT_LIB_PATH}/*.so")

# ADD CUDA
# add CUDA
set(CUDA_LIBRARIES "/usr/local/cuda/lib64")
set(CUDA_INCLUDE_DIRS "/usr/local/cuda/include")
message("CUDA_LIBRARIES:${CUDA_LIBRARIES}")
message("CUDA_INCLUDE_DIRS:${CUDA_INCLUDE_DIRS}")
include_directories(${CUDA_INCLUDE_DIRS})
file(GLOB CUDA_LIBS "${CUDA_LIBRARIES}/*.so")

target_link_libraries(ModelInfer ${OpenCV_LIBS} ${LIBS} ${CUDA_LIBS})

# 设置输出目录的变量为与CMakeLists.txt文件相同级别的目录
set(MY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)
# 设置可执行文件输出路径
set(EXECUTABLE_OUTPUT_PATH ${MY_OUTPUT_DIRECTORY})
# 设置库文件输出路径
set(LIBRARY_OUTPUT_PATH ${MY_OUTPUT_DIRECTORY})
