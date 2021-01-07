cmake_minimum_required(VERSION 3.8)

project(sphconv LANGUAGES CXX CUDA)

find_package(Threads REQUIRED)
link_libraries(Threads::Threads)

set(CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}/lib/python3.7/site-packages/torch/share/cmake/Torch/;${CMAKE_PREFIX_PATH}")
message("Cmake prefix  =" ${CMAKE_PREFIX_PATH})

find_package(Torch REQUIRED)
# find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")
# see https://github.com/pytorch/pytorch/issues/38122
# add_subdirectory(third_party/pybind11)
find_package(pybind11 CONFIG REQUIRED)

include_directories(${TORCH_INCLUDE_DIRS})


# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

add_subdirectory(sphconv/)



