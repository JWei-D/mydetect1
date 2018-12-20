cmake_minimum_required(VERSION 2.8)

project(mydetec1)

list(APPEND CMAKE_MODULE_PATH ~/caffe/cmake/Modules)

#include_directories("~/mydetec1/data/" "~/mydetec1/data/")
set(OpenCV_DIR /usr/share/OpenCV)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

#link_libraries("/opt/OpenBLAS/lib/libopenblas.so")
link_libraries("/usr/lib/libblas.so.3")

add_executable(${PROJECT_NAME} "main.cpp")
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
add_library(mtcnn SHARED mtcnn.cpp)
target_link_libraries(${PROJECT_NAME} mtcnn)
add_library(pbox SHARED pBox.cpp)
target_link_libraries(${PROJECT_NAME} pbox)
add_library(network SHARED network.cpp)
target_link_libraries(${PROJECT_NAME} network)
#add_library(libtest SHARED test.cpp)
#target_link_libraries(${PROJECT_NAME} libtest)
