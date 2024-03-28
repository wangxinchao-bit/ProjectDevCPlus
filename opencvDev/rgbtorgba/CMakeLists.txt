
cmake_minimum_required(VERSION 3.5)
project(rgb_to_rgba)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(rgb_to_rgba 0305opencvTest.cpp)
target_link_libraries(rgb_to_rgba ${OpenCV_LIBS})


# 设置输出目录的变量为与CMakeLists.txt文件相同级别的目录
set(MY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)
# 设置可执行文件输出路径
set(EXECUTABLE_OUTPUT_PATH ${MY_OUTPUT_DIRECTORY})
# 设置库文件输出路径
set(LIBRARY_OUTPUT_PATH ${MY_OUTPUT_DIRECTORY})
