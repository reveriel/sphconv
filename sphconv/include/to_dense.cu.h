#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

using torch::RestrictPtrTraits;

namespace sphconv {

template <typename Index>
__global__ void to_dense_kernel(
    const torch::PackedTensorAccessor32<float , 5, RestrictPtrTraits>
  feature,
    const torch::PackedTensorAccessor32<Index , 4, RestrictPtrTraits>
  depth,
    const torch::PackedTensorAccessor32<Index , 3, RestrictPtrTraits>
  thick,
    torch::PackedTensorAccessor32<float , 5, RestrictPtrTraits>
  buffer,
  int N, int C,
  int H, int W
)
{
  for (int x = threadIdx.x + blockDim.x * blockIdx.x;
       x < H; x += blockDim.x * gridDim.x)
  {
    for (int y = threadIdx.y + blockDim.y * blockIdx.y;
         y < W; y += blockDim.y * gridDim.y)
    {

      for (int b = 0; b < N; b++)
      {
        for (int t = 0; t < thick[b][x][y]; t++)
        {
          Index z = depth[b][t][x][y];

          for (int c = threadIdx.z; c < C; c += blockDim.z)
          {
            buffer[b][c][z][x][y] = feature[b][c][t][x][y];
          }
        }
      }
    }
  }
}


template<typename Index>
__global__ void to_dense_backward_kernel(
    const torch::PackedTensorAccessor32<float , 5, RestrictPtrTraits>
  d_featureOut,
    const torch::PackedTensorAccessor32<Index , 4, RestrictPtrTraits>
  depth,
    const torch::PackedTensorAccessor32<Index , 3, RestrictPtrTraits>
  thick,
    torch::PackedTensorAccessor32<float , 5, RestrictPtrTraits>
  d_feature,
  int N, int C,
  int H, int W
)
{
  for (int x = threadIdx.x + blockDim.x * blockIdx.x;
       x < H; x += blockDim.x * gridDim.x)
  {
    for (int y = threadIdx.y + blockDim.y * blockIdx.y;
         y < W; y += blockDim.y * gridDim.y)
    {

      for (int b = 0; b < N; b++)
      {
        for (int t = 0; t < thick[b][x][y]; t++)
        {

          Index z = depth[b][t][x][y];

          for (int c = threadIdx.z; c < C; c += blockDim.z)
          {
            d_feature[b][c][t][x][y] = d_featureOut[b][c][z][x][y];
          }

        }
      }
    }
  }
}

} // namespace sphconv
