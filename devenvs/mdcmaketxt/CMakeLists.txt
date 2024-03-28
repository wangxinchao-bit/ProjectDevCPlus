# cmake_minimum_required(VERSION 3.10)
# project(ctensorr)

# set(CMAKE_CXX_STANDARD 11)
# set(CUDA_INCLUDE_DIRS "/usr/local/cuda/include")
# set(CUDA_LIBRARIES "/usr/local/cuda/lib64")
# set(TenssorRT_INLCUDE_DIRS "/home/wxcwxc/clibs/TensorRT-8.6.1.6/include")
# set(TensorRT_LIBRARIES "/home/wxcwxc/clibs/TensorRT-8.6.1.6/lib")


# include_directories(${CUDA_INCLUDE_DIRS})
# include_directories(${TenssorRT_INLCUDE_DIRS})

# add_executable(my_tensorrt_app main.cpp)
# # 链接TensorRT和CUA依赖库
# target_link_libraries(my_tensorrt_app ${TensorRT_LIBRARIES} ${CUDA_LIBRARIES})


cmake_minimum_required(VERSION 3.13)
project(TensorRT_test)
set(CMAKE_CXX_STANDARD 11)
 
set(SAMPLES_COMMON_SOURCES "/home/wxcwxc/clibs/TensorRT-8.6.1.6/samples/common/logger.cpp")
# add_executable(TensorRT_test testSim.cpp ${SAMPLES_COMMON_SOURCES})
add_executable(TensorRT_test 0305opencvTest.cpp ${SAMPLES_COMMON_SOURCES})
set(OpenCV_INCLUDE_DIRS "/usr/local/include/opencv4")
# add OpenCV
# find_package(OpenCV REQUIRED)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
 
# add TensorRT8
include_directories(/home/wxcwxc/clibs/TensorRT-8.6.1.6/include)
include_directories(/home/wxcwxc/clibs/TensorRT-8.6.1.6/samples/common)
set(TENSORRT_LIB_PATH "/home/wxcwxc/clibs/TensorRT-8.6.1.6/lib")
file(GLOB LIBS "${TENSORRT_LIB_PATH}/*.so")
 
# add CUDA
set(CUDA_LIBRARIES "/usr/local/cuda/lib64")
set(CUDA_INCLUDE_DIRS "/usr/local/cuda/include")
message("CUDA_LIBRARIES:${CUDA_LIBRARIES}")
message("CUDA_INCLUDE_DIRS:${CUDA_INCLUDE_DIRS}")
include_directories(${CUDA_INCLUDE_DIRS})
file(GLOB CUDA_LIBS "${CUDA_LIBRARIES}/*.so")
 
# link
target_link_libraries(TensorRT_test ${LIBS} ${CUDA_LIBS} ${OpenCV_LIBS})

