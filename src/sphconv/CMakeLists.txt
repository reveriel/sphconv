
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

pybind11_add_module(sphconv_cuda
    all.cpp
    rules.cu
    rule_conv.cu
    tensor.cu
    # indice_conv.cu
    # indice.cu
    # to_dense.cu
)

target_include_directories(sphconv_cuda PRIVATE ${PROJECT_SOURCE_DIR}/include/)

find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")
message(STATUS "TORCH_PYTHON_LIBRARY: ${TORCH_PYTHON_LIBRARY}")
target_link_libraries(sphconv_cuda PRIVATE ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})

target_compile_definitions(sphconv_cuda PRIVATE VERSION_INFO=${SPHCONV_VERSION_INFO})


# target_link_libraries(sphconv_cuda ${CUDA_CUDART} pybind11::module)

# find_package(Catch2 REQUIRED)

# Target_link_libraries(feature PRIVATE Catch2::Catch2 sphconv_cuda "${TORCH_LIBRARIES}" )
