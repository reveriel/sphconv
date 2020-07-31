
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include "to_dense.cu.h"
#include "debug_utils.h"

using torch::RestrictPtrTraits;

namespace sphconv {

const int H_BLOCK = 4, W_BLOCK = 8;

torch::Tensor
to_dense(torch::Tensor feature,
         torch::Tensor depth,
         torch::Tensor thick,
         int D)
{
  int N = feature.size(0);
  int C = feature.size(1);
  int H = feature.size(3);
  int W = feature.size(4);

  auto options = torch::TensorOptions().dtype(feature.dtype()).device(feature.device());
  torch::Tensor buffer = torch::zeros({N, C, D, H, W}, options);

  // choose C_BLOCK
  int C_BLOCK = 4;
  if (C > 4 && C <= 8) {
    C_BLOCK = 8;
  } else if (C > 8 && C <= 16) {
     C_BLOCK = 16;
  } else if (C > 16) {
    C_BLOCK = 32;
  }

  dim3 grid_size, block_size;
  grid_size = dim3(divUp(H, H_BLOCK), divUp(W, W_BLOCK), 1);
  block_size = dim3(H_BLOCK, W_BLOCK, C_BLOCK);

  to_dense_kernel<int32_t><<<grid_size, block_size>>>(
      feature.packed_accessor32<float, 5, RestrictPtrTraits>(),
      depth.packed_accessor32<int32_t, 4, RestrictPtrTraits>(),
      thick.packed_accessor32<int32_t, 3, RestrictPtrTraits>(),
      buffer.packed_accessor32<float, 5, RestrictPtrTraits>(),
      N, C, H, W);

  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  return buffer;
}

torch::Tensor
to_dense_backward(torch::Tensor d_featureOut,
                  torch::Tensor depth,
                  torch::Tensor thick,
                  int T)
{

  int N = d_featureOut.size(0);
  int C = d_featureOut.size(1);
  int H = d_featureOut.size(3);
  int W = d_featureOut.size(4);

  // choose C_BLOCK
  int C_BLOCK = 4;
  if (C > 4 && C <= 8) {
    C_BLOCK = 8;
  } else if (C > 8 && C <= 16) {
     C_BLOCK = 16;
  } else if (C > 16) {
    C_BLOCK = 32;
  }

  dim3 grid_size, block_size;
  grid_size = dim3(divUp(H, H_BLOCK), divUp(W, W_BLOCK), 1);
  block_size = dim3(H_BLOCK, W_BLOCK, C_BLOCK);

  auto options = torch::TensorOptions().dtype(d_featureOut.dtype()).device(d_featureOut.device());
  auto d_feature = torch::zeros({N,C,T,H,W} , options);

  to_dense_backward_kernel<int32_t><<<grid_size, block_size>>>(
      d_featureOut.packed_accessor32<float, 5, RestrictPtrTraits>(),
      depth.packed_accessor32<int32_t, 4, RestrictPtrTraits>(),
      thick.packed_accessor32<int32_t, 3, RestrictPtrTraits>(),
      d_feature.packed_accessor32<float, 5, RestrictPtrTraits>(),
      N, C, H, W);

  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  return d_feature;
}

} // namespace sphconv
