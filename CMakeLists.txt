cmake_minimum_required(VERSION 3.8)

set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-10.2)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-10.2/bin/nvcc)
project(sphconv LANGUAGES CXX CUDA)

find_package(Threads REQUIRED)
link_libraries(Threads::Threads)

set(CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}/lib/python3.7/site-packages/torch/share/cmake/Torch/;${CMAKE_PREFIX_PATH}")
message("Cmake prefix  =" ${CMAKE_PREFIX_PATH})

# add_compile_definitions(CUTLASS_ARCH_WMMA_ENABLED)

### nvcc fatal : Unknown option 'Wall
# This issue is also addressed in README.md. It occurs when torch version >= 1.4

# If you are using linux, you can simply type locate Caffe2Targets.cmake in
# terminal to see the location of the file. For instance, the location could be
# "~/anaconda3/pkgs/pytorch-1.5.0-py3.7_cuda10.1.243_cudnn7.6.3_0/lib/python3.7/site-packages/torch/share/cmake/Caffe2/Caffe2Targets.cmake"
# in your machine. Edit the file for the pytorch version you will use.
#
# There are two lines INTERFACE_COMPILE_OPTIONS
# "-Wall;-Wextra;-Wno-unused-parameter;-Wno-missing-field-initializers;-Wno-write-strings;-Wno-unknown-pragmas;-Wno-missing-braces;-fopenmp"
# in the file. Search for them and comment out those two lines.


###  No rule to make target '/usr/local/cuda/lib64/libnvToolsExt.so', needed by '../../sphconv/sphconv_cuda.cpython-37m-x86_64-linux-gnu.so'.  Stop.
# So basically the path for libculibos.a is hardcoded in
# /path/to/libtorch/share/cmake/Caffe2/Caffe2Targets.cmake as
# /usr/local/cuda/lib64/libculibos.a

# As part of the CMake calls, the file
# /path/to/libtorch/share/cmake/Caffe2/public/cuda.cmake sets the value of
# ${CUDA_TOOLKIT_ROOT_DIR} to your cuda home, which in my case was
# /usr/local/cuda-9.0

# After changing the hardcoded path in
# /path/to/libtorch/share/cmake/Caffe2/Caffe2Targets.cmake from
# /usr/local/cuda/lib64/libculibos.a to
# ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libculibos.a I am able to compile the
# example-app and run it successfully.

find_package(Torch REQUIRED)
find_package(PythonLibs REQUIRED)
# find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")
# see https://github.com/pytorch/pytorch/issues/38122
# add_subdirectory(third_party/pybind11)
find_package(pybind11 CONFIG REQUIRED)

include_directories(${TORCH_INCLUDE_DIRS})


# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include_directories(${PYTHON_INCLUDE_DIRS})

# add_subdirectory(third_party/cutlass)

add_subdirectory(src/sphconv)
add_subdirectory(src/utils)



