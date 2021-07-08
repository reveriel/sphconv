
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -gencode=arch=compute_75,code=sm_75 -Xcompiler -rdynamic -lineinfo ")

# https://docs.nvidia.com/cuda/cuda-memcheck/index.html#cuda-memcheck-tool-examples
# add debug info


pybind11_add_module(sphconv_cuda
    all.cpp
    rules.cu
    rule_conv.cu
    tensor.cu
)
# target_compile_options(sphconv_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
# target_compile_options(sphconv_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -rdynamic>)

target_include_directories(sphconv_cuda PRIVATE ${PROJECT_SOURCE_DIR}/include/
    PRIVATE  ${PROJECT_SOURCE_DIR}/third_party/cutlass/include/ )
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")
message(STATUS "TORCH_PYTHON_LIBRARY: ${TORCH_PYTHON_LIBRARY}")
message(STATUS "TORCH_LIBRARIES: ${TORCH_LIBRARIES}")
target_link_libraries(sphconv_cuda PRIVATE ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})
target_compile_definitions(sphconv_cuda PRIVATE VERSION_INFO=${SPHCONV_VERSION_INFO})

add_executable(test_gemm test_gemm.cu)
target_link_libraries(test_gemm PRIVATE ${CUDA_cusparse_LIBRARY} ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY} )
target_include_directories(test_gemm PRIVATE ${PROJECT_SOURCE_DIR}/include/
    PRIVATE
      ${PROJECT_SOURCE_DIR}/third_party/cutlass/include/
      ${PROJECT_SOURCE_DIR}/third_party/cutlass/tools/util/include/
     )
