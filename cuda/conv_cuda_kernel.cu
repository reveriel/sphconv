
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

template <typename ElementType>
__global__ void sphconv_cuda_forward_kernel(
    const
)