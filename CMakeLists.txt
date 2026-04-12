cmake_minimum_required(VERSION 3.20)
project(cctv_capstone)

set(CMAKE_CXX_STANDARD 17)

# OpenCV
set(OpenCV_DIR "C:/cctvcapstone/opencv/opencv/build")
set(OpenCV_ARCH x64)
set(OpenCV_RUNTIME vc16)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# ONNX Runtime
set(ONNXRUNTIME_DIR "C:/cctvcapstone/onnxruntime/lib/onnxruntime-win-x64-gpu-1.24.4")
include_directories(${ONNXRUNTIME_DIR}/include)
link_directories(${ONNXRUNTIME_DIR}/lib)

# CUDA
set(CUDA_DIR "C:/cctvcapstone/cuda")
include_directories(${CUDA_DIR}/include)
link_directories(${CUDA_DIR}/lib/x64)

# 실행 파일
add_executable(cctv_capstone main.cpp)
target_link_libraries(cctv_capstone
    ${OpenCV_LIBS}
    onnxruntime
)